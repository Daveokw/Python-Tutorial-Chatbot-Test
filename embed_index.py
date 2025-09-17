import os
import time
import pickle
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

ROOT = "https://www.w3schools.com/python/"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PKL = os.path.join(DATA_DIR, "chunks.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 32
MAX_PAGES = 800
CRAWL_DELAY = 0.12
USER_AGENT = "site-qa-bot/1.0"

def fetch_url(url, session=None, timeout=15):
    s = session or requests.Session()
    try:
        r = s.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def extract_main_with_trafilatura(html, url):
    try:
        main = trafilatura.extract(html, output_format="text")
    except Exception:
        main = None
    soup = BeautifulSoup(html, "html.parser")
    code_blocks = []
    for pre in soup.find_all("pre"):
        txt = pre.get_text().strip()
        if txt:
            code_blocks.append("```\n" + txt + "\n```")
    headings = []
    for h in soup.find_all(["h1","h2","h3"]):
        headings.append(h.get_text().strip())
    combined = ""
    if main and len(main.strip()) > 50:
        combined = main.strip()
    else:
        combined = soup.get_text(separator="\n").strip()
    if code_blocks:
        combined += "\n\n" + "\n\n".join(code_blocks)
    return combined

def clean_url(u):
    return u.rstrip("/")

def get_links(html, base_url, allowed_prefix):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if not href or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        if full.startswith(allowed_prefix) and (not any(x in full.lower() for x in ["tryit.asp","signup","login","profile","shop","certificate","forum","about","news"])):
            links.add(clean_url(full))
    return links

def simple_crawl(start_url, max_pages=500):
    print("Starting crawl:", start_url)
    session = requests.Session()
    visited = set()
    to_visit = [clean_url(start_url)]
    pages = []
    allowed_base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url)) + "/python"
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        html = fetch_url(url, session=session)
        if not html:
            visited.add(url)
            continue
        text = extract_main_with_trafilatura(html, url)
        if text and len(text.strip()) > 120:
            pages.append({"url": url, "text": text})
            print("Collected:", url)
        visited.add(url)
        links = get_links(html, url, allowed_base)
        for l in links:
            if l not in visited and l not in to_visit:
                to_visit.append(l)
        time.sleep(CRAWL_DELAY)
    print("Crawl complete. Pages:", len(pages))
    return pages

def chunk_page_text(url, text, min_chunk_chars=700, overlap_chars=150):
    """
    Create larger chunks (~min_chunk_chars) with overlap so each chunk has
    enough context (paragraphs + code examples).
    """
    text = text.strip()
    if not text:
        return []
    # split by paragraphs (double newline)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks = []
    cur = ""
    for b in blocks:
        if cur:
            cand = cur + "\n\n" + b
        else:
            cand = b
        if len(cand) >= min_chunk_chars:
            # push chunk (keep some overlap)
            chunks.append({"url": url, "text": cand})
            # start new cur as last overlap_chars of cand
            if overlap_chars > 0:
                cur = cand[-overlap_chars:]
            else:
                cur = ""
        else:
            cur = cand
    # final
    if cur and len(cur) >= 120:
        chunks.append({"url": url, "text": cur})
    # safeguard: if still no chunks, add the whole page
    if not chunks and len(text) >= 120:
        chunks = [{"url": url, "text": text}]
    return chunks

def build_index(pages, embed_model_name=EMBED_MODEL, batch_size=EMBED_BATCH):
    print(f"Embedding {sum(len(p['text'])>0 for p in pages)} pages -> chunking ...")
    chunks = []
    for p in pages:
        c = chunk_page_text(p["url"], p["text"])
        chunks.extend(c)
    print("Total chunks:", len(chunks))
    if len(chunks) == 0:
        raise RuntimeError("No chunks to embed")

    model = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]
    emb_batches = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_batches.append(embs)
    X = np.vstack(emb_batches).astype("float32")
    dim = X.shape[1]
    print("Embedding dim:", dim)
    idx = faiss.IndexFlatL2(dim)
    idx.add(X)
    # save
    faiss.write_index(idx, INDEX_FAISS)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(chunks, f)
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(chunks, f)
    print("Index and metadata saved to data/")

def main():
    pages = simple_crawl(ROOT, max_pages=MAX_PAGES)
    if not pages:
        print("No pages found — exiting")
        return
    build_index(pages)

if __name__ == "__main__":
    main()
