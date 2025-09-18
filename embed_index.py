import os
import sys
import time
import argparse
import pickle
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("ERROR: sentence-transformers is required. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
except Exception:
    print("ERROR: faiss (faiss-cpu) is required. Install with: pip install faiss-cpu")
    sys.exit(1)

import numpy as np

ROOT = "https://docs.python.org/3/tutorial/"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PKL = os.path.join(DATA_DIR, "chunks.pkl")

USER_AGENT = "site-qa-bot/1.0"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MIN_CHUNK_CHARS = 800
DEFAULT_OVERLAP_CHARS = 150

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def fetch_url_text(url, timeout=15):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def extract_main(html):
    """
    Prefer trafilatura for main extraction; fall back to BeautifulSoup.
    Preserve code blocks (pre).
    """
    if not html:
        return ""
    text = None
    if trafilatura:
        try:
            main = trafilatura.extract(html)
            if main and len(main.strip()) > 50:
                text = main.strip()
        except Exception:
            text = None

    if text is None:
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript", "iframe"]):
            s.decompose()
        for sel in ("header", "footer", "nav", "aside"):
            for el in soup.select(sel):
                el.decompose()
        parts = []
        for tag in soup.find_all(["h1","h2","h3","p","li","pre","code"]):
            t = tag.get_text(separator="\n").strip()
            if t:
                parts.append(t)
        text = "\n\n".join(parts).strip()

    try:
        soup2 = BeautifulSoup(html, "html.parser")
        codes = []
        for pre in soup2.find_all("pre"):
            c = pre.get_text().strip()
            if c and len(c) > 10:
                codes.append("```\n" + c + "\n```")
        if codes:
            if text:
                text = text + "\n\n" + "\n\n".join(codes)
            else:
                text = "\n\n".join(codes)
    except Exception:
        pass

    return text or ""

def get_links(html, base_url, allowed_prefix):
    """
    Extract same-site links that start with allowed_prefix.
    """
    links = set()
    if not html:
        return links
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].strip()
        if not href:
            continue
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        if full.startswith(allowed_prefix):
            if any(x in full.lower() for x in ["print.html", "genindex.html", "py-modindex.html"]):
                continue
            links.add(full.rstrip("/"))
    return links

def simple_crawl(start_url, max_pages=300, politeness=0.12):
    allowed_prefix = "{uri.scheme}://{uri.netloc}/3/tutorial".format(uri=urlparse(start_url))
    visited = set()
    to_visit = [start_url.rstrip("/")]
    pages = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        html = fetch_url_text(url)
        if not html:
            visited.add(url)
            continue
        text = extract_main(html)
        if text and len(text.strip()) >= 120:
            pages.append({"url": url, "text": text})
            print("Collected:", url)
        visited.add(url)
        links = get_links(html, url, allowed_prefix)
        for l in links:
            if l not in visited and l not in to_visit and len(visited) + len(to_visit) < max_pages:
                to_visit.append(l)
        time.sleep(politeness)
    return pages

def chunk_page_text(url, text, min_chunk_chars=DEFAULT_MIN_CHUNK_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS):
    """
    Create larger overlapping chunks from page text.
    """
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks = []
    cur = ""
    for b in blocks:
        cand = (cur + "\n\n" + b).strip() if cur else b
        if len(cand) >= min_chunk_chars:
            chunks.append({"url": url, "text": cand})
            cur = cand[-overlap_chars:] if overlap_chars > 0 else ""
        else:
            cur = cand
    if cur and len(cur) >= 120:
        chunks.append({"url": url, "text": cur})
    if not chunks and len(text) >= 120:
        chunks = [{"url": url, "text": text}]
    return chunks

def build_index_from_pages(pages, embed_model_name=DEFAULT_MODEL, batch_size=32, min_chunk_chars=DEFAULT_MIN_CHUNK_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS):
    all_chunks = []
    for p in pages:
        cs = chunk_page_text(p["url"], p["text"], min_chunk_chars, overlap_chars)
        all_chunks.extend(cs)
    print("Total chunks:", len(all_chunks))
    if not all_chunks:
        raise RuntimeError("No chunks produced.")
    model = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in all_chunks]
    emb_arrays = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_arrays.append(embs)
    X = np.vstack(emb_arrays).astype("float32")
    dim = X.shape[1]
    print("Embedding dim:", dim)
    idx = faiss.IndexFlatL2(dim)
    idx.add(X)
    faiss.write_index(idx, INDEX_FAISS)
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(all_chunks, f)
    print("Saved index:", INDEX_FAISS)
    print("Saved metadata:", INDEX_PKL, CHUNKS_PKL)
    return idx, all_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=300, help="Max pages to fetch/crawl")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--min-chunk-chars", type=int, default=DEFAULT_MIN_CHUNK_CHARS, help="Minimum characters per chunk")
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS, help="Overlap characters between chunks")
    args = parser.parse_args()

    print("Starting crawl of:", ROOT)
    pages = simple_crawl(ROOT, max_pages=args.max_pages)
    if not pages:
        print("No pages were discovered. Exiting.")
        return
    print(f"Building index from {len(pages)} pages (model={args.model}, batch={args.batch_size})")
    build_index_from_pages(pages, embed_model_name=args.model, batch_size=args.batch_size, min_chunk_chars=args.min_chunk_chars, overlap_chars=args.overlap_chars)
    print("Done.")

if __name__ == "__main__":
    main()
