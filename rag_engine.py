"""
rag_engine.py — ITA Technician Assistant

Core retrieval + generation logic, decoupled from the UI (app.py) so it can
be unit-tested or reused in a CLI/API context.

Retrieval strategy: hybrid BM25 (sparse/lexical) + TF-IDF-LSA (dense/semantic),
fused with min-max normalized weighted scores, then a lightweight lexical
rerank pass on the fused top-N to push exact-term matches (model numbers,
IP addresses, button names) above near-miss semantic matches.

Generation strategy: provider-agnostic. Checks for an available LLM backend
in this priority order:
  1. Anthropic API (ANTHROPIC_API_KEY env var)
  2. OpenAI-compatible API (OPENAI_API_KEY env var)
  3. Local Ollama (if a local Ollama server is reachable)
  4. Extractive fallback — no LLM call at all; returns the top retrieved
     chunks directly, formatted as a cited answer. This means the assistant
     is fully functional with zero API keys and zero cost, which matters
     for a small business evaluating this before committing to any provider.
"""

import os
import re
import json
import pickle

import numpy as np
import faiss

BASE_DIR = os.path.dirname(__file__)
FAISS_PATH = os.path.join(BASE_DIR, "index", "dense.faiss")
BM25_PATH = os.path.join(BASE_DIR, "index", "bm25.pkl")
META_PATH = os.path.join(BASE_DIR, "index", "meta.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "index", "tfidf_svd.pkl")

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
TOP_K_RETRIEVE = 8   # candidates pulled from each retriever before fusion
TOP_K_FINAL = 4       # chunks actually passed to generation / shown to user


def simple_tokenize(text: str):
    return text.lower().replace("/", " ").replace("-", " ").split()


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class RagEngine:
    def __init__(self):
        self.faiss_index = faiss.read_index(FAISS_PATH)
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            bundle = pickle.load(f)
            self.vectorizer = bundle["vectorizer"]
            self.svd = bundle["svd"]

    # ---------- Retrieval ----------

    def _dense_search(self, query: str, k: int):
        vec = self.vectorizer.transform([query])
        lsa_vec = self.svd.transform(vec)
        norm = np.linalg.norm(lsa_vec, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        lsa_vec = (lsa_vec / norm).astype("float32")
        scores, idxs = self.faiss_index.search(lsa_vec, k)
        return idxs[0], scores[0]

    def _sparse_search(self, query: str, k: int):
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return top_idx, scores[top_idx]

    def _lexical_rerank_boost(self, query: str, chunk_text: str) -> float:
        """Small boost for chunks that literally contain query terms verbatim
        (model numbers, IPs, exact error strings) — dense/BM25 fusion alone
        can under-rank these relative to topically-similar-but-less-precise
        chunks."""
        q_terms = set(simple_tokenize(query))
        c_terms = set(simple_tokenize(chunk_text))
        if not q_terms:
            return 0.0
        overlap = len(q_terms & c_terms) / len(q_terms)
        return overlap

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL):
        dense_idx, dense_scores = self._dense_search(query, TOP_K_RETRIEVE)
        sparse_idx, sparse_scores = self._sparse_search(query, TOP_K_RETRIEVE)

        dense_norm = _minmax_norm(np.asarray(dense_scores, dtype="float32"))
        sparse_norm = _minmax_norm(np.asarray(sparse_scores, dtype="float32"))

        fused = {}
        for i, idx in enumerate(dense_idx):
            if idx == -1:
                continue
            fused[idx] = fused.get(idx, 0.0) + DENSE_WEIGHT * dense_norm[i]
        for i, idx in enumerate(sparse_idx):
            fused[idx] = fused.get(idx, 0.0) + SPARSE_WEIGHT * sparse_norm[i]

        # lexical rerank boost on the fused candidate set
        candidates = []
        for idx, score in fused.items():
            chunk = self.chunks[idx]
            boost = self._lexical_rerank_boost(query, chunk["text"])
            final_score = score + 0.15 * boost
            candidates.append((idx, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        results = []
        for idx, score in top:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    # ---------- Generation ----------

    # Ollama connection settings — override via env vars if needed.
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

    STEP_FORMAT_INSTRUCTIONS = (
        "You are a field-support assistant for ITA technicians who are usually mid-job, "
        "on a ladder, or on the phone with a customer. They do NOT want paragraphs. "
        "Answer using ONLY the manual excerpts provided. Respond in EXACTLY this format, "
        "with no extra commentary before or after:\n\n"
        "ISSUE: <one short line restating the problem>\n"
        "VENDOR: <vendor/product this applies to>\n"
        "STEPS:\n"
        "1. <short, single action per line>\n"
        "2. <short, single action per line>\n"
        "(as many numbered steps as needed, each one a single concrete action)\n"
        "WARNING: <only include this line if there's a genuine safety/data-loss risk, "
        "e.g. power/shock hazard or a step that wipes configuration — omit the line entirely "
        "if there is none>\n"
        "IF_STILL_STUCK: <one short line on what to check or escalate next>\n\n"
        "Keep every line short. No fluff, no restating the question, no markdown headers."
    )

    def _build_prompt(self, query: str, contexts: list) -> str:
        context_block = "\n\n".join(
            f"[Source: {c['doc_title']} — {c['section_title']}]\n{c['text']}"
            for c in contexts
        )
        return (
            f"{self.STEP_FORMAT_INSTRUCTIONS}\n\n"
            f"MANUAL EXCERPTS:\n{context_block}\n\n"
            f"TECHNICIAN QUESTION: {query}"
        )

    def _try_anthropic(self, prompt: str):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import httpx
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []))
        except Exception as e:
            print(f"[rag_engine] Anthropic call failed: {e}")
            return None

    def _try_ollama(self, prompt: str):
        try:
            import httpx
            resp = httpx.post(
                f"{self.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                # local first-token latency can be slow on a cold model load
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("response")
        except Exception as e:
            print(f"[rag_engine] Ollama call failed ({self.OLLAMA_HOST}, "
                  f"model={self.OLLAMA_MODEL}): {e}")
            return None

    # ---------- Step parsing (turns either LLM output or raw chunk text
    # into a structured dict the UI can render as step cards) ----------

    def _parse_structured_llm_output(self, text: str):
        """Parse the ISSUE/VENDOR/STEPS/WARNING/IF_STILL_STUCK format above.
        Returns None if the model didn't follow the format, so the caller
        can fall back to a looser parse."""
        issue = re.search(r"ISSUE:\s*(.+)", text)
        vendor = re.search(r"VENDOR:\s*(.+)", text)
        warning = re.search(r"WARNING:\s*(.+)", text)
        escalate = re.search(r"IF_STILL_STUCK:\s*(.+)", text)
        steps_block = re.search(r"STEPS:\s*(.+?)(?:WARNING:|IF_STILL_STUCK:|$)", text, re.DOTALL)

        if not steps_block:
            return None

        step_lines = re.findall(r"\d+\.\s*(.+)", steps_block.group(1))
        step_lines = [s.strip() for s in step_lines if s.strip()]
        if not step_lines:
            return None

        return {
            "issue": issue.group(1).strip() if issue else None,
            "vendor": vendor.group(1).strip() if vendor else None,
            "steps": step_lines,
            "warning": warning.group(1).strip() if warning else None,
            "escalate": escalate.group(1).strip() if escalate else None,
        }

    def _parse_chunk_as_steps(self, chunk: dict):
        """Best-effort structuring of a raw manual chunk (extractive fallback,
        no LLM available). Looks for numbered steps or Q/A pairs already
        present in the manual text; otherwise returns it as a short info card."""
        text = chunk["text"]

        image_paths = chunk.get("image_paths", [])

        numbered = re.findall(r"(?:^|\n)\s*\d+\.\s*(.+)", text)
        if len(numbered) >= 2:
            return {
                "type": "steps",
                "title": chunk["section_title"],
                "vendor": chunk["vendor"],
                "steps": [s.strip() for s in numbered],
                "image_paths": image_paths,
            }

        qa_pairs = re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)", text, re.DOTALL)
        if qa_pairs:
            return {
                "type": "qa",
                "title": chunk["section_title"],
                "vendor": chunk["vendor"],
                "pairs": [(q.strip(), a.strip()) for q, a in qa_pairs],
                "image_paths": image_paths,
            }

        return {
            "type": "info",
            "title": chunk["section_title"],
            "vendor": chunk["vendor"],
            "text": text.strip(),
            "image_paths": image_paths,
        }

    def answer(self, query: str):
        contexts = self.retrieve(query, top_k=TOP_K_FINAL)
        if not contexts:
            return {"mode": "none"}, []

        prompt = self._build_prompt(query, contexts)

        raw = self._try_anthropic(prompt)
        source_backend = "anthropic" if raw else None
        if raw is None:
            raw = self._try_ollama(prompt)
            source_backend = "ollama" if raw else None

        if raw is not None:
            parsed = self._parse_structured_llm_output(raw)
            if parsed is not None:
                parsed["mode"] = "llm_steps"
                parsed["backend"] = source_backend
                return parsed, contexts
            # model responded but didn't follow the format — show raw text
            return {"mode": "llm_raw", "text": raw, "backend": source_backend}, contexts

        # no LLM backend available at all — structure the raw chunks instead
        cards = [self._parse_chunk_as_steps(c) for c in contexts]
        return {"mode": "extractive", "cards": cards}, contexts


if __name__ == "__main__":
    # Quick manual smoke test
    engine = RagEngine()
    test_queries = [
        "How do I factory reset an Araknis switch?",
        "Sonos speaker won't show up during setup, what do I check?",
        "Josh Core has power but won't show up in the app",
        "What's the default IP for a Lutron hub troubleshooting?",
    ]
    for q in test_queries:
        print("=" * 70)
        print("Q:", q)
        result, ctx = engine.answer(q)
        print("-" * 70)
        print(json.dumps(result, indent=2)[:800])
        print("\n[Sources used]:", [f"{c['doc_title']} / {c['section_title']}" for c in ctx])