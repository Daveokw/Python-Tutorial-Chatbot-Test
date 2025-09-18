# app.py (fixed)
import os
import re
import pickle
import requests
import gdown
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# Make trafilatura optional (cloud-friendly)
try:
    import trafilatura
except Exception:
    trafilatura = None

load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
GDRIVE_CHUNKS_PKL_ID = os.getenv("GDRIVE_CHUNKS_ID")  # optional

HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # default small for cloud-safety

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_PKL = os.path.join(DATA_DIR, "chunks.pkl")

# ---------------- Silent download ----------------
def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
            gdown.download(url, INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
            gdown.download(url, INDEX_PKL, quiet=True)
        if GDRIVE_CHUNKS_PKL_ID and not os.path.exists(CHUNKS_PKL):
            url = f"https://drive.google.com/uc?id={GDRIVE_CHUNKS_PKL_ID}"
            gdown.download(url, CHUNKS_PKL, quiet=True)
    except Exception:
        pass

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
    # DistilBERT-based QA pipeline (CPU safe)
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_gen_pipeline(model_name=HF_MODEL):
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline("text2text-generation", model=model_name, device=device)

@st.cache_resource(show_spinner=False)
def load_index_and_chunks():
    try:
        if not (os.path.exists(INDEX_FAISS) and os.path.exists(INDEX_PKL)):
            return None, None
        idx = faiss.read_index(INDEX_FAISS)
        with open(INDEX_PKL, "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    except Exception:
        return None, None

# ---------------- Cleaning ----------------
BOILERPLATE_PATTERNS = [
    r"\bGet Certified\b", r"\bSign In\b", r"\bTryit Editor\b", r"\bSearch\b",
    r"\bSpaces\b", r"\bMenu\b", r"\bAbout\b", r"\bContact\b", r"\bSubscribe\b",
    r"W3Schools", r"©", r"Privacy", r"Terms", r"Home\b", r"Next\b", r"Prev\b",
    r"Advertisement", r"©\s*\d{4}", r"Table of Contents", r"Contents", r"Jump to",
    r"Related Topics", r"Try it Yourself", r"Get Certified", r"Sign In"
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\r\n", "\n", s)
    lines = []
    for ln in (line.strip() for line in s.splitlines()):
        if not ln:
            continue
        if len(ln) < 30 and (ln.isupper() or len(ln.split()) <= 3):
            continue
        if BOILERPLATE_RE.search(ln):
            continue
        lines.append(ln)
    out = "\n".join(lines)
    out = re.sub(r"\n{2,}", "\n\n", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# ---------------- Search & QA flow ----------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
summarizer = load_summarizer()
gen_pipeline = load_gen_pipeline()

def search_index(idx, chunks, question, k=6):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

# improved dedupe + pick best candidate
def normalize_answer(a: str) -> str:
    s = re.sub(r'\s+', ' ', a.strip().lower())
    s = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s)
    return s

def pick_best_candidate(candidates):
    if not candidates:
        return None
    best = {}
    for c in candidates:
        key = normalize_answer(c["answer"])
        if not key:
            continue
        prev = best.get(key)
        if not prev:
            best[key] = c
        else:
            if c["score"] > prev["score"] + 1e-6 or (len(c.get("text","")) > len(prev.get("text","")) and c["score"] >= prev["score"] - 1e-6):
                best[key] = c
    unique = sorted(best.values(), key=lambda x: (-x["score"], x["dist"]))
    return unique[0] if unique else None

def extract_candidates(hits_with_dist, question, keep_top=8):
    cand = []
    for h, dist in hits_with_dist:
        text = clean_text(h.get("text",""))
        if not text or len(text) < 60:
            continue
        try:
            res = extractive_qa(question=question, context=text)
            score = float(res.get("score", 0.0))
            ans = res.get("answer","").strip()
            if ans:
                cand.append({
                    "answer": ans,
                    "score": score,
                    "source": h.get("url",""),
                    "text": text,
                    "dist": dist
                })
        except Exception:
            continue
    return sorted(cand, key=lambda x: (-x["score"], x["dist"]))[:keep_top]

def fallback_summarize_top_chunks(candidates_or_hits):
    if summarizer is None:
        return None
    texts = []
    for c in candidates_or_hits[:3]:
        if isinstance(c, tuple):
            texts.append(clean_text(c[0].get("text","")))
        else:
            texts.append(c.get("text",""))
    joined = "\n\n".join([t for t in texts if t])
    if not joined:
        return None
    try:
        out = summarizer(joined, max_length=120, min_length=40, do_sample=False)
        return out[0].get("summary_text","").strip()
    except Exception:
        return None

# concise default answer (direct)
def get_concise_answer(question, candidates):
    best = pick_best_candidate(candidates)
    if best:
        one = best["answer"].strip()
        one = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', one)  # simple de-dup
        if not re.search(r'[.!?]$', one):
            one = one + '.'
        return one, best.get("source", "")
    return None, ""

# expanded answer via generator (only when user requests)
def synthesize_expanded(question, candidates, max_len=280):
    if not candidates:
        return None, []
    facts = []
    sources = []
    used = set()
    for c in candidates[:6]:
        a = c.get("answer","").strip()
        key = normalize_answer(a)
        if not key or key in used:
            continue
        used.add(key)
        if len(a) >= 8:
            facts.append(f"- {a}")
            s = c.get("source")
            if s and s not in sources:
                sources.append(s)
    if sum(len(f) for f in facts) < 60:
        summ = fallback_summarize_top_chunks(candidates)
        if summ:
            facts = [f"- {summ}"]
    if not facts:
        return None, []
    facts_block = "\n".join(facts)
    prompt = (
        "You are a helpful assistant. Use ONLY the facts below as your main source. "
        "You MAY add very small widely-known background facts (creator/release year) for programming languages if appropriate. "
        "Write a concise, friendly explanation structured as: 1) One-line definition; 2) 3 bullets: what makes it special; 3) 3 bullets: what you can build; 4) a tiny python example (1-3 lines); 5) one closing encouragement.\n\n"
        f"Question: {question}\n\nFacts:\n{facts_block}\n\nAnswer:"
    )
    try:
        out = gen_pipeline(prompt, max_length=max_len, do_sample=False)
        gen = out[0].get("generated_text") or out[0].get("summary_text") or ""
        gen = gen.strip()
        gen = re.sub(r'\s{2,}', ' ', gen)
        return gen, sources[:2]
    except Exception:
        return None, sources[:2]

# ---------------- URL-on-demand QA ----------------
def fetch_main_from_url(url):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"site-qa-bot/1.0"})
        r.raise_for_status()
        html = r.text
        if trafilatura:
            main = trafilatura.extract(html)
            if main and len(main.strip()) > 50:
                return main
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        return text
    except Exception:
        return None

def fetch_and_answer_url(url, question):
    text = fetch_main_from_url(url)
    if not text or len(text.strip()) < 80:
        return None, []
    clean = clean_text(text)
    try:
        res = extractive_qa(question=question, context=clean)
        ans = res.get("answer","").strip()
        score = float(res.get("score",0.0))
        if ans and score >= 0.05:
            cand = [{"answer": ans, "score": score, "source": url, "text": clean, "dist": 0.0}]
            concise, src = get_concise_answer(question, cand)
            return concise, [src] if src else []
        summ = None
        if summarizer:
            try:
                s = summarizer(clean, max_length=160, min_length=60, do_sample=False)
                summ = s[0].get("summary_text","").strip()
            except Exception:
                summ = None
        if summ:
            cand = [{"answer": summ, "score": 0.01, "source": url, "text": clean, "dist": 0.0}]
            concise, src = get_concise_answer(question, cand)
            return concise, [src] if src else []
    except Exception:
        pass
    return None, []

# ---------------- UI ----------------
st.set_page_config(page_title="W3Schools Python Chatbot", layout="centered")
st.title("🐍 W3Schools Python — Q&A")

download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.warning("Index not available. Upload index files and set GDRIVE_INDEX_FAISS_ID & GDRIVE_INDEX_PKL_ID.")
    st.stop()

show_dev = st.checkbox("Developer: show candidates & raw chunks", value=False)

url_input = st.text_input("Optional: specific URL (leave blank to use index):", value="", key="url_input")
query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        # URL branch (direct answer from page)
        if url_input.strip():
            concise, sources = fetch_and_answer_url(url_input.strip(), query)
            if concise:
                st.markdown("### 🧠 Answer (from provided URL)")
                st.write(concise)
                if sources:
                    st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
            else:
                st.info("No confident answer found on that page.")
        else:
            # Index retrieval branch
            hits = search_index(INDEX, CHUNKS, query, k=8)
            candidates = extract_candidates(hits, query, keep_top=8)

            # Developer view
            if show_dev:
                st.subheader("Candidates (dev)")
                st.write(candidates)
                st.subheader("Top raw chunks (dev)")
                for h, d in hits[:4]:
                    st.write(h.get("url",""))
                    st.write(h.get("text","")[:400] + ("..." if len(h.get("text",""))>400 else ""))

            # default concise answer
            concise, top_src = get_concise_answer(query, candidates)
            if concise:
                st.markdown("### 🧠 Answer")
                st.write(concise)
                if top_src:
                    st.markdown("**Source:** " + f"[{top_src}]({top_src})")
                # show button to expand
                if st.button("Show expanded answer"):
                    expanded, sources = synthesize_expanded(query, candidates, max_len=420)
                    if expanded:
                        st.markdown("### 🧾 Expanded Answer")
                        st.markdown(expanded)
                        if sources:
                            st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
                    else:
                        st.info("No expanded answer could be generated.")
            else:
                raw_summ = fallback_summarize_top_chunks(hits)
                if raw_summ:
                    st.markdown("### 🧠 Answer (fallback summarization)")
                    st.write(raw_summ)
                else:
                    st.info("I couldn't find a confident answer in the indexed content.")
