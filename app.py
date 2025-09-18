import os, re, pickle
import streamlit as st
import gdown
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from dotenv import load_dotenv

try:
    import trafilatura
except Exception:
    trafilatura = None

load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")  
GDRIVE_INDEX_PKL_ID  = os.getenv("GDRIVE_INDEX_PKL_ID")     
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")   

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PKL = os.path.join(DATA_DIR, "chunks.pkl")

def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}", INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}", INDEX_PKL, quiet=True)
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource(show_spinner=False)
def load_gen(model_name=HF_MODEL):
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    try:
        return pipeline("text2text-generation", model=model_name, device=device)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_index():
    try:
        if not (os.path.exists(INDEX_FAISS) and os.path.exists(INDEX_PKL)):
            return None, None
        idx = faiss.read_index(INDEX_FAISS)
        with open(INDEX_PKL, "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    except Exception:
        return None, None

BOILERPLATE = ["W3Schools","©","Privacy","Terms","Search","Get Certified","Sign In","Menu"]
def clean_text(s):
    if not s:
        return ""
    out = "\n".join([ln.strip() for ln in s.splitlines() if ln.strip()])
    for b in BOILERPLATE:
        out = out.replace(b, "")
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

embed_model = load_embed_model()
extractive = load_extractive()
gen = load_gen()

def search_index(idx, chunks, question, k=6):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

def pick_best_candidate(candidates):
    if not candidates:
        return None
    best = {}
    for c in candidates:
        key = re.sub(r'\s+',' ', c['answer'].strip().lower())
        key = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','', key)
        if not key:
            continue
        prev = best.get(key)
        if not prev:
            best[key] = c
        else:
            if c['score'] > prev['score']:
                best[key] = c
    chosen = sorted(best.values(), key=lambda x: -x['score'])
    return chosen[0] if chosen else None

def extract_candidates(hits, question, keep_top=8):
    cand = []
    for h, dist in hits:
        text = clean_text(h.get("text",""))
        if not text or len(text)<60:
            continue
        try:
            res = extractive(question=question, context=text)
            score = float(res.get("score",0.0))
            ans = (res.get("answer") or "").strip()
            if ans:
                cand.append({"answer":ans,"score":score,"source":h.get("url",""),"text":text,"dist":dist})
        except Exception:
            continue
    cand = sorted(cand, key=lambda x: -x['score'])[:keep_top]
    return cand

def get_concise(question, candidates):
    best = pick_best_candidate(candidates)
    if best:
        a = best['answer'].strip()
        a = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', a)
        if not re.search(r'[.!?]$', a):
            a = a + '.'
        return a, best.get("source","")
    return None, None

def synthesize_expanded(question, candidates):
    if not candidates or gen is None:
        return None
    facts = []
    srcs = []
    used = set()
    for c in candidates[:6]:
        a = c.get("answer","").strip()
        k = re.sub(r'\s+',' ', a.lower())
        if not k or k in used: continue
        used.add(k)
        facts.append("- " + a)
        s = c.get("source")
        if s and s not in srcs: srcs.append(s)
    if not facts:
        return None
    prompt = (
        "You are a helpful assistant. Using ONLY the facts below, write a clear friendly answer. "
        "Produce 1) one-line definition, 2) 3 short bullets 'what makes it special', 3) 3 short bullets 'what you can build', 4) a tiny python example (if relevant).\n\n"
        f"Question: {question}\n\nFacts:\n{chr(10).join(facts)}\n\nAnswer:"
    )
    try:
        out = gen(prompt, max_length=280, do_sample=False)
        text = out[0].get("generated_text","").strip()
        return text, srcs[:2]
    except Exception:
        return None, srcs[:2]

st.set_page_config(page_title="Python Tutorial Chatbot", layout="centered")
st.title("🐍 Python Tutorial Chatbot")

download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index()

if INDEX is None or CHUNKS is None:
    st.error("Index files missing. Run embed_index.py locally, upload data/index.faiss and data/index.pkl, or set GDRIVE_INDEX_FAISS_ID & GDRIVE_INDEX_PKL_ID.")
    st.stop()

q = st.text_input("Ask a question about the Python tutorial:", "")
if st.button("Ask") and q.strip():
    with st.spinner("Searching..."):
        hits = search_index(INDEX, CHUNKS, q, k=8)
        candidates = extract_candidates(hits, q, keep_top=8)
        concise, src = get_concise(q, candidates)
        if concise:
            st.markdown("### Answer")
            st.write(concise)
            if src:
                st.markdown(f"**Source:** [{src}]({src})")
            if st.button("Show expanded answer"):
                expanded, sources = synthesize_expanded(q, candidates)
                if expanded:
                    st.markdown("### Expanded Answer")
                    st.write(expanded)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
                else:
                    st.info("No expanded answer available.")
        else:
            if candidates:
                expanded, sources = synthesize_expanded(q, candidates)
                if expanded:
                    st.markdown("### Answer (from summarization)")
                    st.write(expanded)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
                else:
                    st.info("No confident answer found.")
            else:
                st.info("No relevant content found in the index.")
