import os
import pickle
import time
import gdown
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# ------------------------
# Config / env
# ------------------------
load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
GDRIVE_CHUNKS_ID     = os.getenv("GDRIVE_CHUNKS_ID")  

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PICKLE = os.path.join(DATA_DIR, "chunks.pkl")

# ------------------------
# Helpers: download index from Google Drive (if missing)
# ------------------------
def download_from_gdrive_if_missing():
    """Download index.faiss and index.pkl from Google Drive using file IDs in env vars."""
    downloaded = False
    if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
        url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
        st.info("Downloading vector index (this may take a moment)...")
        gdown.download(url, INDEX_FAISS, quiet=False)
        downloaded = True
    if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
        url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
        st.info("Downloading index metadata (this may take a moment)...")
        gdown.download(url, INDEX_PKL, quiet=False)
        downloaded = True
    if GDRIVE_CHUNKS_ID and not os.path.exists(CHUNKS_PICKLE):
        url = f"https://drive.google.com/uc?id={GDRIVE_CHUNKS_ID}"
        st.info("Downloading index metadata (this may take a moment)...")
        gdown.download(url, CHUNKS_PICKLE, quiet=False)
        downloaded = True
    return downloaded

# ------------------------
# Cached resource loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    # Small extractive QA model that runs on CPU; swap for a different HF model if you prefer.
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource(show_spinner=False)
def load_index_and_chunks():
    """
    Loads FAISS index and chunks (list of dicts {url,text}) from disk.
    Returns (index, chunks) or (None, None) if missing/corrupt.
    """
    try:
        if not (os.path.exists(INDEX_FAISS) and os.path.exists(INDEX_PKL)):
            return None, None
        idx = faiss.read_index(INDEX_FAISS)
        with open(INDEX_PKL, "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    except Exception:
        return None, None

# ------------------------
# Search & answer helpers
# ------------------------
embed_model = load_embed_model()
qa_pipeline = load_qa_pipeline()

def search_index(idx, chunks, question, k=4):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for idx_i in I[0]:
        if idx_i < len(chunks):
            results.append(chunks[idx_i])
    return results

def build_context_from_hits(hits, max_chars=3000):
    pieces = []
    sources = []
    for h in hits:
        txt = h.get("text","").strip()
        url = h.get("url","")
        if txt:
            piece = f"{txt}"
            pieces.append(piece)
            if url:
                sources.append(url)
    context = "\n\n".join(pieces)
    # truncate to a reasonable size for the QA model
    if len(context) > max_chars:
        context = context[:max_chars]
    return context, list(dict.fromkeys(sources))

def answer_with_hf(question, context):
    if not context.strip():
        return None
    try:
        res = qa_pipeline(question=question, context=context, topk=1)
        # pipeline returns dict (single) or list if topk>1
        if isinstance(res, list):
            res = res[0]
        return res.get("answer", "").strip()
    except Exception:
        return None

# ------------------------
# App UI (minimal)
# ------------------------
st.set_page_config(page_title="W3Schools Python Chatbot", layout="centered")
st.title("🐍 W3Schools Python — Q&A")

# Ensure index present (download if needed), then load
with st.spinner("Preparing index..."):
    # Try download from Google Drive if missing
    download_from_gdrive_if_missing()
    INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.warning("Index not available. If you uploaded index files to Google Drive, set the GDRIVE_INDEX_FAISS_ID and GDRIVE_INDEX_PKL_ID environment variables. Otherwise run the index builder locally and upload the files.")
    # Still show a simple input but disable Ask until index is available
    query = st.text_input("Ask a question about W3Schools Python (index not loaded yet):", value="")
    st.stop()

# Input area (single box)
query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Retrieving and answering..."):
        try:
            hits = search_index(INDEX, CHUNKS, query, k=5)
            context, sources = build_context_from_hits(hits, max_chars=3000)
            answer = answer_with_hf(query, context)
            if answer:
                st.markdown("### 🧠 Answer")
                st.write(answer)
            else:
                st.info("I couldn't find a confident answer in the indexed content.")
                # show retrieved excerpts as fallback
                st.markdown("### Retrieved excerpts")
                for h in hits:
                    url = h.get("url","")
                    st.write(f"- [{url}]({url})" if url else "- (no url)")
                    st.write(h.get("text","")[:800] + ("..." if len(h.get("text",""))>800 else ""))
            # show sources (compact)
            if sources:
                st.markdown("### 📌 Sources")
                for s in sources:
                    st.write(f"- [{s}]({s})")
        except Exception:
            st.error("An error occurred while searching or answering. Please try again.")