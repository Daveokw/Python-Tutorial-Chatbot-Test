# app.py
import os
import pickle
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

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ------------------------
# Helpers: download index from Google Drive (silent)
# ------------------------
def download_from_gdrive_if_missing():
    """
    Download index files from Google Drive using gdown if they are missing locally.
    This runs silently (no UI messages).
    """
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
            gdown.download(url, INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
            gdown.download(url, INDEX_PKL, quiet=True)
    except Exception:
        # ignore download errors here; load_index_and_chunks will handle missing files
        pass

# ------------------------
# Cached resource loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_gen_pipeline():
    """
    Use a small FLAN-T5 model for text-to-text generation.
    It produces natural answers given a prompt + context.
    """
    # Use the small variant for reasonable CPU usage. Swap for a larger model if you have more resources.
    return pipeline("text2text-generation", model="google/flan-t5-small", device_map=None)

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
gen_pipeline = load_gen_pipeline()

def search_index(idx, chunks, question, k=4):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for idx_i in I[0]:
        if idx_i < len(chunks):
            results.append(chunks[idx_i])
    return results

def build_context_from_hits(hits, max_chars=2500):
    """
    Merge retrieved chunks into a single context string, trimming to max_chars.
    """
    pieces = []
    for h in hits:
        txt = h.get("text","").strip()
        if txt:
            pieces.append(txt)
    context = "\n\n".join(pieces)
    if len(context) > max_chars:
        context = context[:max_chars]
    return context

def answer_with_generation(question, context, max_length=200):
    """
    Use FLAN-T5-style text2text generation to produce a natural answer grounded in context.
    If context is empty, returns None.
    """
    if not context.strip():
        return None
    # Build a clear prompt instructing the model to answer using only the context
    prompt = (
        "Answer the user's question using ONLY the information from the context below. "
        "If the answer is not present in the context, reply: \"I couldn't find this in the site data.\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        out = gen_pipeline(prompt, max_length=max_length, do_sample=False)
        # out is a list of dicts [{'generated_text': '...'}]
        text = out[0].get("generated_text","").strip()
        return text
    except Exception:
        return None

# ------------------------
# Streamlit UI (minimal)
# ------------------------
st.set_page_config(page_title="W3Schools Python Chatbot", layout="centered")
st.title("🐍 W3Schools Python — Q&A")

# silent attempt to ensure index files exist (no UI messages)
download_from_gdrive_if_missing()

# load index
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.warning("Index not available. Please upload index files to Google Drive and set GDRIVE_INDEX_FAISS_ID and GDRIVE_INDEX_PKL_ID as secrets, or run the index builder locally.")
    # still display a very minimal input but don't attempt to answer without the index
    st.text_input("Ask a question about W3Schools Python (index not loaded):")
    st.stop()

# main input / ask button
query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        hits = search_index(INDEX, CHUNKS, query, k=6)
        context = build_context_from_hits(hits, max_chars=2500)
        answer = answer_with_generation(query, context, max_length=220)
        if answer:
            # simple clean-up: if model returns the fallback phrase, show it exactly
            if "couldn't find" in answer.lower() or "could not find" in answer.lower():
                st.info("I couldn't find this in the site data.")
            else:
                st.markdown("### 🧠 Answer")
                st.write(answer)
        else:
            st.info("I couldn't find a confident answer in the indexed content.")
