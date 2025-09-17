# app.py
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
import trafilatura

# ---------------- Config ----------------
load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")  # set to "-small" if limited

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ---------------- Silent download ----------------
def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
            gdown.download(url, INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
            gdown.download(url, INDEX_PKL, quiet=True)
    except Exception:
        pass

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
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
    r"Advertisement", r"©\s*\d{4}", r"Table of Contents", r"Contents", r"Jump to", r"Related Topics",
    r"Previous", r"Next", r"Skip to content", r"Show more", r"Log in", r"Sign up", r"Try it Yourself"
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

# ---------------- Search & QA pipeline ----------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
summarizer = load_summarizer()
gen_pipeline = load_gen_pipeline()

def search_index(idx, chunks, question, k=8):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

# improved dedupe helpers
def normalize_answer(a: str) -> str:
    s = re.sub(r'\s+', ' ', a.strip().lower())
    s = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s)
    return s

def pick_best_variant(candidates_list):
    best = {}
    for c in candidates_list:
        key = normalize_answer(c["answer"])
        if not key:
            continue
        prev = best.get(key)
        if not prev:
            best[key] = c
        else:
            if (c["score"] > prev["score"] + 1e-6) or (len(c.get("text","")) > len(prev.get("text","")) and c["score"] >= prev["score"] - 1e-6):
                best[key] = c
    out = sorted(best.values(), key=lambda x: (-x["score"], x["dist"]))
    return out

def extract_answers_from_hits(hits_with_dist, question, keep_top=8):
    raw_candidates = []
    for h, dist in hits_with_dist:
        text = clean_text(h.get("text",""))
        if not text or len(text) < 60:
            continue
        try:
            res = extractive_qa(question=question, context=text)
            score = float(res.get("score", 0.0))
            ans = res.get("answer","").strip()
            if ans:
                raw_candidates.append({
                    "answer": ans,
                    "score": score,
                    "source": h.get("url",""),
                    "text": text,
                    "dist": dist
                })
        except Exception:
            continue
    if not raw_candidates:
        return []
    unique = pick_best_variant(raw_candidates)
    return unique[:keep_top]

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

def synthesize_answer_rich(question, candidates, max_len=420):
    """
    Build a structured prompt that asks the generator to expand and produce
    a friendly, informative answer. Allows a small set of safe background facts
    for programming languages (creator, release year).
    Returns (answer_text, top_sources)
    """
    if not candidates:
        return None, []

    # build short deduped facts (prefer variants that are longer / higher score)
    facts = []
    sources = []
    for c in candidates[:6]:
        a = c.get("answer","").strip()
        if not a or len(a) < 8:
            continue
        facts.append(f"- {a}")
        src = c.get("source")
        if src and src not in sources:
            sources.append(src)
    # fallback: if facts are extremely short, use summarizer on top texts
    if sum(len(f) for f in facts) < 60:
        summ = None
        try:
            if summarizer:
                # summarizer expects a single long text
                joined = "\n\n".join([c.get("text","") for c in candidates[:3]])
                if joined and len(joined) > 200:
                    out = summarizer(joined, max_length=140, min_length=60, do_sample=False)
                    summ = out[0].get("summary_text","").strip()
        except Exception:
            summ = None
        if summ:
            facts = [f"- {summ}"]

    if not facts:
        return None, sources[:2]

    facts_block = "\n".join(facts)

    # Add a short knowledge whitelist for safe background facts for 'python'
    background_guidance = (
        "You may add only very small, widely-known background facts such as the creator and first release year "
        "for programming languages (for example: 'Python was created by Guido van Rossum and first released in 1991'). "
        "Do NOT invent other facts or details not implied by the facts below."
    )

    prompt = (
        "You are a helpful, friendly AI tutor. Use ONLY the facts below as your main source. "
        + background_guidance
        + " Produce a clear answer with this structure:\n\n"
        "1) One-line definition/summary.\n"
        "2) 'What makes it special?' (3 bullets)\n"
        "3) 'What can you build?' (3 bullets)\n"
        "4) A tiny example in ```python``` if relevant (1-4 lines).\n"
        "5) One encouraging closing sentence.\n\n"
        f"Question: {question}\n\nFacts:\n{facts_block}\n\nAnswer:"
    )

    try:
        out = gen_pipeline(prompt, max_length=max_len, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text, sources[:2]
    except Exception:
        return None, sources[:2]

# ---------------- URL-on-demand QA ----------------
def fetch_main_from_url(url):
    """Fetch page and extract main content via trafilatura (fallback to basic text)."""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "site-qa-bot/1.0"})
        r.raise_for_status()
        html = r.text
        main = trafilatura.extract(html)
        if main and len(main.strip()) > 50:
            # include code blocks from HTML as well
            return main
        # fallback: simple soup extraction without soup import-heavy logic to keep deps small
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        return text
    except Exception:
        return None

def fetch_and_answer_url(url, question):
    """Fetch a single URL and answer the question from its content."""
    text = fetch_main_from_url(url)
    if not text or len(text.strip()) < 80:
        return None, []
    clean = clean_text(text)
    # try extractive QA directly on the page
    try:
        res = extractive_qa(question=question, context=clean)
        ans = res.get("answer","").strip()
        score = float(res.get("score", 0.0))
        # if score is reasonably high, synthesize a structured answer that includes this fact
        if ans and score >= 0.05:
            # create a pseudo-candidate
            cand = [{"answer": ans, "score": score, "source": url, "text": clean, "dist": 0.0}]
            out, sources = synthesize_answer_rich(question, cand, max_len=320)
            return out, sources
        # else fallback to summarizer over the whole page then synthesize
        summ = None
        if summarizer:
            try:
                s = summarizer(clean, max_length=160, min_length=60, do_sample=False)
                summ = s[0].get("summary_text","").strip()
            except Exception:
                summ = None
        if summ:
            cand = [{"answer": summ, "score": 0.01, "source": url, "text": clean, "dist": 0.0}]
            out, sources = synthesize_answer_rich(question, cand, max_len=320)
            return out, sources
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

# Optional URL field (if provided, answer directly from the URL)
url_input = st.text_input("Optional: Enter a specific URL to answer from (leave blank to use site index):", value="", key="url_input")
query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        if url_input.strip():
            answer, sources = fetch_and_answer_url(url_input.strip(), query)
            if answer:
                if "couldn't find" in answer.lower():
                    st.info("I couldn't find this in the page data.")
                else:
                    st.markdown("### 🧠 Answer (from provided URL)")
                    st.markdown(answer)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
            else:
                st.info("No confident answer found on that page.")
        else:
            try:
                hits = search_index(INDEX, CHUNKS, query, k=10)
                candidates = extract_answers_from_hits(hits, query, keep_top=8)
                answer, top_sources = None, []
                if candidates:
                    answer, top_sources = synthesize_answer_rich(query, candidates, max_len=420)
                else:
                    raw_summ = fallback_summarize_top_chunks(hits)
                    if raw_summ:
                        pseudo = [{"answer": raw_summ, "score": 0.01, "source": ""}]
                        answer, top_sources = synthesize_answer_rich(query, pseudo, max_len=420)
                if show_dev:
                    st.subheader("Candidates (dev)")
                    st.write(candidates)
                    st.subheader("Top raw chunks (dev)")
                    for h, d in hits[:4]:
                        st.write(h.get("url",""))
                        st.write(h.get("text","")[:400] + ("..." if len(h.get("text",""))>400 else ""))
                if answer:
                    if "couldn't find" in answer.lower():
                        st.info("I couldn't find this in the site data.")
                    else:
                        st.markdown("### 🧠 Answer")
                        st.markdown(answer)
                        if top_sources:
                            st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in top_sources))
                else:
                    st.info("I couldn't find a confident answer in the indexed content.")
            except Exception:
                st.error("An error occurred. Please try again.")
