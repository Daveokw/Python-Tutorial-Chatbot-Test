import os
import time
import json
import pickle
import threading
import requests
import faiss
import numpy as np
import streamlit as st
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import gdown

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env (set OPENAI_API_KEY=sk-...)")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")  
GDRIVE_INDEX_PKL_ID  = os.getenv("GDRIVE_INDEX_PKL_ID")
GDRIVE_CHUNKS_ID     = os.getenv("GDRIVE_CHUNKS_ID")  

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PICKLE = os.path.join(DATA_DIR, "chunks.pkl")
STATUS_JSON = os.path.join(DATA_DIR, "build_status.json")

W3_PYTHON_ROOT = "https://www.w3schools.com/python/"
SITEMAP_URL = "https://www.w3schools.com/sitemap.xml"

@st.cache_resource
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

embed_model = load_embed_model()

INDEX = None
CHUNKS = []

def save_status(status: dict):
    try:
        with open(STATUS_JSON, "w", encoding="utf-8") as f:
            json.dump(status, f)
    except Exception:
        pass

def load_status():
    if not os.path.exists(STATUS_JSON):
        return {"running": False, "stage": "idle", "progress": 0, "message": "", "timestamp": time.time()}
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"running": False, "stage": "idle", "progress": 0, "message": "", "timestamp": time.time()}

if not os.path.exists(STATUS_JSON):
    save_status({"running": False, "stage": "idle", "progress": 0, "message": "idle", "timestamp": time.time()})


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "iframe"]):
        s.decompose()
    for sel in ("header", "footer", "nav", "aside"):
        for el in soup.select(sel):
            el.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def get_urls_from_sitemap_filtered(sitemap_url: str, path_filter="/python/"):
    urls = []
    try:
        r = requests.get(sitemap_url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        save_status({"running": False, "stage": "error", "progress": 0, "message": f"sitemap fetch failed: {e}", "timestamp": time.time()})
        return urls

    soup = BeautifulSoup(r.content, "xml")
    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        if path_filter.lower() in url.lower():
            if any(x in url.lower() for x in ["tryit.asp", "signup", "login", "profile", "shop", "certificate", "forum", "about", "news"]):
                continue
            urls.append(url)
    return urls

def allowed_by_robots_simple(url: str):
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = requests.get(robots_url, timeout=8)
        if r.status_code != 200:
            return True
        txt = r.text.lower()
        if "disallow: /" in txt:
            return False
        return True
    except Exception:
        return False

def background_build(pages, batch_size=32):
    """
    pages: list of {"url","text"}
    This function runs in a background thread: chunk -> embed -> build faiss -> save files -> update status.
    """
    status = {"running": True, "stage": "chunking", "progress": 0, "message": "Splitting pages into chunks", "timestamp": time.time()}
    save_status(status)

    # chunk pages
    chunks = []
    for p in pages:
        for block in p["text"].split("\n\n"):
            t = block.strip()
            if len(t) >= 120:
                chunks.append({"url": p["url"], "text": t})
    if not chunks:
        save_status({"running": False, "stage": "error", "progress": 0, "message": "No chunks extracted", "timestamp": time.time()})
        return

    total = len(chunks)
    status.update({"stage":"embedding", "progress":0, "message": f"Embedding {total} chunks", "timestamp": time.time()})
    save_status(status)

    # embed
    embeddings_list = []
    texts = [c["text"] for c in chunks]
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            embs = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            save_status({"running": False, "stage": "error", "progress": int(i/total*100), "message": f"Embedding failed: {e}", "timestamp": time.time()})
            return
        embeddings_list.append(embs)
        status["progress"] = int((i+len(batch))/total*100)
        status["message"] = f"Embedded {i+len(batch)}/{total} chunks"
        save_status(status)

    X = np.vstack(embeddings_list).astype("float32")
    status.update({"stage":"index", "progress": 0, "message":"Building FAISS index", "timestamp": time.time()})
    save_status(status)

    # build faiss
    try:
        dim = X.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(X)
        faiss.write_index(idx, INDEX_FAISS)
        with open(INDEX_PKL, "wb") as f:
            pickle.dump(chunks, f)
        with open(CHUNKS_PICKLE, "wb") as f:
            pickle.dump(chunks, f)
    except Exception as e:
        save_status({"running": False, "stage": "error", "progress": 0, "message": f"FAISS build/save failed: {e}", "timestamp": time.time()})
        return

    save_status({"running": False, "stage": "done", "progress": 100, "message": f"Index built ({len(chunks)} chunks).", "timestamp": time.time()})
    return

_build_thread = None

def start_background_build_from_sitemap(max_pages=500, batch_size=32):
    """
    Start a background build using URLs from sitemap (filtered to /python/).
    """
    status = {"running": True, "stage": "sitemap", "progress": 0, "message": "Fetching sitemap", "timestamp": time.time()}
    save_status(status)

    urls = get_urls_from_sitemap_filtered(SITEMAP_URL, path_filter="/python/")
    if not urls:
        status = {"running": True, "stage": "crawl", "progress":0, "message":"Sitemap empty, falling back to crawl", "timestamp": time.time()}
        save_status(status)
        pages = simple_crawl_python(W3_PYTHON_ROOT, max_pages=max_pages)
    else:
        urls = urls[:max_pages]
        pages = []
        session = requests.Session()
        for i, url in enumerate(urls, start=1):
            try:
                r = session.get(url, timeout=15, headers={"User-Agent":"site-qa-bot/1.0"})
                r.raise_for_status()
                text = extract_text(r.text)
                if len(text.strip()) >= 120:
                    pages.append({"url": url, "text": text})
            except Exception:
                pass
            save_status({"running": True, "stage": "fetching", "progress": int(i/len(urls)*10), "message": f"Fetched {i}/{len(urls)} pages", "timestamp": time.time()})
            time.sleep(0.05)

    global _build_thread
    if _build_thread and _build_thread.is_alive():
        return False
    _build_thread = threading.Thread(target=background_build, args=(pages, batch_size), daemon=True)
    _build_thread.start()
    return True

def simple_crawl_python(start_url: str, max_pages: int = 200):
    domain_base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url))
    session = requests.Session()
    visited = set()
    to_visit = [start_url.rstrip("/")]
    pages = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        if not url.lower().startswith(W3_PYTHON_ROOT):
            visited.add(url); continue
        if not allowed_by_robots_simple(url):
            visited.add(url); continue
        try:
            r = session.get(url, timeout=15, headers={"User-Agent":"site-qa-bot/1.0"})
            r.raise_for_status()
        except Exception:
            visited.add(url); continue

        text = extract_text(r.text)
        if len(text.strip()) >= 120:
            pages.append({"url": url, "text": text})
        visited.add(url)

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split('#')[0]
            if href.startswith("javascript:") or href.strip() == "":
                continue
            full = urljoin(url, href)
            if full.startswith(W3_PYTHON_ROOT) and (full not in visited) and (full not in to_visit):
                if not any(x in full.lower() for x in ["tryit.asp", "signup", "login", "profile", "shop", "certificate", "forum", "about", "news"]):
                    to_visit.append(full.rstrip("/"))
        time.sleep(0.15)
    return pages

def download_from_gdrive_if_missing():
    os.makedirs("data", exist_ok=True)
    ok = False
    if GDRIVE_INDEX_FAISS_ID and not os.path.exists("data/index.faiss"):
        url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
        print("Downloading index.faiss from Google Drive...")
        gdown.download(url, "data/index.faiss", quiet=False)
        ok = True
    if GDRIVE_INDEX_PKL_ID and not os.path.exists("data/index.pkl"):
        url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
        print("Downloading index.pkl from Google Drive...")
        gdown.download(url, "data/index.pkl", quiet=False)
        ok = True
    if GDRIVE_CHUNKS_ID and not os.path.exists("data/chunks.pkl"):
        url = f"https://drive.google.com/uc?id={GDRIVE_CHUNKS_ID}"
        print("Downloading chunks.pkl from Google Drive...")
        gdown.download(url, "data/chunks.pkl", quiet=False)
        ok = True
    return ok

def load_index_if_exists():
    global INDEX, CHUNKS
    if os.path.exists(INDEX_FAISS) and os.path.exists(INDEX_PKL):
        try:
            INDEX = faiss.read_index(INDEX_FAISS)
            with open(INDEX_PKL, "rb") as f:
                CHUNKS = pickle.load(f)
            return True
        except Exception:
            return False
    return False

def search_index_local(idx, chks, question: str, k: int = 5):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    res = []
    for i in I[0]:
        if i < len(chks):
            res.append(chks[i])
    return res

st.set_page_config(page_title="W3Schools Python Chatbot", layout="wide")
st.title("🐍 W3Schools Python — Q&A (Streamlit)")

with st.sidebar:
    st.header("Index controls")
    mode = st.radio("Index source", ["Sitemap (recommended)", "Crawl (link-follow)"])
    max_pages = st.slider("Max pages to fetch", min_value=50, max_value=2000, value=500, step=50)
    batch_size = st.selectbox("Embedding batch size", [16, 32, 64], index=1)
    auto_start = st.checkbox("Auto-start build if index missing (first run)", value=True)
    if st.button("🔄 Start/Restart Build (background)"):
        started = False
        if mode.startswith("Sitemap"):
            started = start_background_build_from_sitemap(max_pages=max_pages, batch_size=batch_size)
        else:
            pages = simple_crawl_python(W3_PYTHON_ROOT, max_pages=max_pages)
            threading.Thread(target=background_build, args=(pages, batch_size), daemon=True).start()
            started = True
        if started:
            st.success("Background build started.")
        else:
            st.warning("A build is already running.")

    if st.button("Load index into memory"):
        ok = load_index_if_exists()
        if ok:
            st.success(f"Index loaded — {len(CHUNKS)} chunks.")
        else:
            st.error("Failed to load index (check data/ files).")

    st.markdown("---")
    st.header("Build status")
    status = load_status()
    st.write(f"Stage: **{status.get('stage','idle')}**")
    st.progress(status.get("progress", 0))
    st.write(status.get("message", ""))
    st.markdown("**Build log (last messages)**")
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            sj = json.load(f)
        st.text(json.dumps(sj, indent=2)[:1000])
    except Exception:
        st.text("No log available")

auto_start = st.sidebar.checkbox("Auto-start if index missing", value=True)

if os.getenv("RUN_MODE", "cloud") == "local":
    if auto_start and (not load_status().get("running", False)):
        start_background_build_from_sitemap(max_pages=max_pages, batch_size=batch_size)
else:
    download_from_gdrive_if_missing()
    load_index_if_exists()

st.subheader("Ask a question about W3Schools Python")
q = st.text_input("Type your question here (press Ask):")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Ask") and q.strip():
        if INDEX is None:
            st.error("Index not loaded. Either wait for background build to finish, or click 'Load index into memory'.")
        else:
            with st.spinner("Searching & building prompt..."):
                try:
                    hits = search_index_local(INDEX, CHUNKS, q, k=5)
                    if not hits:
                        st.info("No relevant content found in the Python index.")
                        st.write("I couldn’t find this in the site data.")
                    else:
                        context = "\n\n---\n\n".join([f"URL: {h['url']}\n{h['text']}" for h in hits])
                        system_prompt = ("You are an assistant that MUST answer questions using ONLY the provided W3Schools Python excerpts. "
                                         "If the answer is not present, respond exactly: \"I couldn’t find this in the site data.\"")
                        user_prompt = f"Website excerpts:\n\n{context}\n\nUser question: {q}"
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
                                max_tokens=700,
                                temperature=0.0
                            )
                            answer = resp.choices[0].message.content.strip()
                            st.markdown("### 🧠 Answer")
                            st.write(answer)
                        except Exception as e:
                            st.warning(f"OpenAI call failed: {e}")
                            st.markdown("### Retrieved excerpts (fallback)")
                            for h in hits:
                                st.markdown(f"**{h['url']}**")
                                st.write(h['text'][:1000] + ("..." if len(h['text'])>1000 else ""))
                except Exception as e:
                    st.error(f"Search failed: {e}")

with col2:
    st.markdown("### Index info")
    if INDEX is None:
        st.write("- Index: **not loaded**")
    else:
        st.write(f"- Chunks: **{len(CHUNKS)}**")
        try:
            st.write(f"- FAISS dim: **{INDEX.d}**, nvecs: **{INDEX.ntotal}**")
        except Exception:
            pass

st.markdown("---")
st.caption("This bot indexes ONLY the W3Schools Python section (URLs under /python/). Use sitemap mode for best coverage.")