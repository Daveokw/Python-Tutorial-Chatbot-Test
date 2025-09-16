import os
import time
import pickle
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PICKLE = os.path.join(DATA_DIR, "chunks.pkl")

W3_PYTHON_ROOT = "https://www.w3schools.com/python/"

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script","style","noscript","iframe"]):
        s.decompose()
    for sel in ("header","footer","nav","aside"):
        for el in soup.select(sel):
            el.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def fetch_sitemap_urls(sitemap_url):
    try:
        r = requests.get(sitemap_url, timeout=20)
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.content, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]

def simple_crawl_python(start_url: str, max_pages: int = 500):
    session = requests.Session()
    visited = set()
    to_visit = [start_url.rstrip("/")]
    pages = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = session.get(url, timeout=15, headers={"User-Agent":"site-qa-bot/1.0"})
            r.raise_for_status()
        except Exception:
            visited.add(url)
            continue
        text = extract_text(r.text)
        if len(text.strip()) > 120:
            pages.append({"url": url, "text": text})
        visited.add(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split('#')[0]
            if href.startswith("javascript:") or not href:
                continue
            full = urljoin(url, href)
            if full.startswith(W3_PYTHON_ROOT) and full not in visited and full not in to_visit:
                if not any(x in full.lower() for x in ["tryit.asp","signup","login","profile","shop","certificate","forum","about","news"]):
                    to_visit.append(full.rstrip("/"))
        time.sleep(0.12)
    return pages

def get_python_urls_from_sitemap_or_crawl(max_pages=1000):
    candidates = [
        "https://www.w3schools.com/sitemap.xml",
        "https://www.w3schools.com/sitemap_index.xml",
        "https://www.w3schools.com/sitemap/sitemap.xml",
        "https://www.w3schools.com/sitemap-index.xml",
        "https://www.w3schools.com/robots.txt",
    ]
    found = []
    for s in candidates:
        try:
            if s.endswith("robots.txt"):
                r = requests.get(s, timeout=10)
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line.split(":",1)[1].strip()
                            urls = fetch_sitemap_urls(sitemap_url)
                            for u in urls:
                                if "/python/" in u.lower():
                                    found.append(u)
            else:
                urls = fetch_sitemap_urls(s)
                for u in urls:
                    if "/python/" in u.lower():
                        found.append(u)
        except Exception:
            pass
        if found:
            break
    found = list(dict.fromkeys(found))
    if found:
        return found[:max_pages]
    pages = simple_crawl_python(W3_PYTHON_ROOT, max_pages=max_pages)
    return [p["url"] for p in pages]

def main(max_pages=500, batch_size=32, embed_model_name="all-MiniLM-L6-v2"):
    print("Gathering python URLs (sitemap or crawl)...")
    urls = get_python_urls_from_sitemap_or_crawl(max_pages=max_pages)
    print(f"Found {len(urls)} python urls (lim {max_pages})")

    session = requests.Session()
    pages = []
    for i, url in enumerate(urls, 1):
        try:
            r = session.get(url, timeout=20, headers={"User-Agent":"site-qa-bot/1.0"})
            r.raise_for_status()
            text = extract_text(r.text)
            if len(text.strip()) >= 120:
                pages.append({"url": url, "text": text})
                print(f"[{i}] added {url}")
        except Exception as e:
            print(f"[{i}] skip {url} ({e})")
        time.sleep(0.08)

    # chunk
    chunks = []
    for p in pages:
        for block in p["text"].split("\n\n"):
            t = block.strip()
            if len(t) >= 120:
                chunks.append({"url": p["url"], "text": t})
    print("Total chunks:", len(chunks))
    if not chunks:
        print("No chunks extracted. Exiting.")
        return

    # embed
    model = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(embs)
        print(f"Embedded {i+len(batch)}/{len(texts)}")
    X = np.vstack(embeddings).astype("float32")

    # build faiss
    dim = X.shape[1]
    print("Embedding dim:", dim)
    idx = faiss.IndexFlatL2(dim)
    idx.add(X)
    print("Index built. saving to disk...")
    faiss.write_index(idx, INDEX_FAISS)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(chunks, f)
    print("Saved index and metadata to data/")

if __name__ == "__main__":
    main(max_pages=500, batch_size=32)