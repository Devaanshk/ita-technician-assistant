# ITA Technician Assistant

A retrieval-augmented (RAG) support assistant built over Innovative Technologies
& Associates' core vendor product lines — **Lutron, Sonos, Araknis Networks,
and Josh.ai**. Ask a plain-English support question, get an answer grounded
in the actual manual content, with sources cited.

Built as a working prototype to demonstrate how ITA's 24/7 monitoring,
service/maintenance, and field-install teams could get instant, cited answers
instead of digging through PDFs mid-call.

## Live Demo

[Open ITA Technician Assistant](https://ita-technician-assistant.onrender.com)

## Why this matters for ITA specifically

- ITA installs and supports **4+ major product lines** (Lutron, Sonos, Josh.ai,
  Araknis, plus James Loudspeaker, IC Realtime, Honeywell, Coastal Source).
  Techs on an emergency on-call ticket at 11pm shouldn't be searching PDFs.
- The company advertises **24/7 monitoring** and **emergency on-call** as core
  services — response speed is a direct differentiator, and this tool cuts the
  "find the right manual section" step to seconds.
- HTA certification and CEDIA membership already signal a company that invests
  in technical rigor — this is a natural extension of that positioning, not a
  bolt-on gimmick.

## Architecture

```
manuals/*.txt          →  ingest.py         →  index/chunks.jsonl
                                                    │
                                          build_index.py
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              │                                           │
                    index/dense.faiss                              index/bm25.pkl
              (TF-IDF → LSA → FAISS, semantic)              (BM25, lexical/exact-match)
                              │                                           │
                              └─────────────────┬─────────────────────────┘
                                                 │
                                          rag_engine.py
                                    (hybrid fusion + lexical rerank
                                     + provider-agnostic generation)
                                                 │
                                             app.py
                                        (Gradio UI, cited answers)
```

**Retrieval is hybrid by design**, matching the reasoning used on the
[SmartFinanceQA](https://github.com/Devaanshk/SmartFinanceQA) project: pure
semantic search misses exact terminology ("AN-310-SW-R-8-POE", "192.168.20.254"),
and pure keyword search misses paraphrased questions ("why won't my speaker
show up" vs. the manual's "speaker not detected"). Fusing both, plus a small
lexical-overlap rerank pass on the fused candidates, covers both query styles.

**Retrieval runs fully offline** — no external embedding API calls. The dense
index is TF-IDF + Truncated SVD (LSA) rather than a downloaded transformer
model, which means zero per-query embedding cost and no proprietary manual
content ever leaves the local machine during retrieval. (This was originally
built with `sentence-transformers` for the dense index — see
[Swapping in a stronger embedding model](#swapping-in-a-stronger-embedding-model)
below for how to switch back once deployed somewhere with normal internet access.)

**Generation is provider-agnostic** and tries, in order:
1. Anthropic API (`ANTHROPIC_API_KEY` env var)
2. Local Ollama server (`llama3.2:1b` by default — matches the model already
   used for local LLM experimentation)
3. Extractive fallback — if no LLM is configured, the app returns the top
   retrieved manual excerpts directly, structured into the same step-card
   format. **The tool is 100% functional with zero API keys and zero
   per-query cost** — a generation backend is an upgrade, not a requirement.

**Every answer renders as step cards, not paragraphs.** Technicians mid-job
don't want prose — the LLM is prompted to return a strict `ISSUE / VENDOR /
STEPS / WARNING / IF_STILL_STUCK` format, which the UI parses into numbered
step cards with a vendor badge and an amber warning banner for anything
safety- or data-loss-relevant (shock hazard, a reset that wipes config,
etc.). If a model ignores the format, the raw answer still displays instead
of failing silently. In extractive mode (no LLM at all), the app parses
numbered steps and Q/A pairs that are already in the manual text into the
same card layout, so the UI is consistent regardless of which backend
answered.

### Using your local Ollama install

```bash
# 1. Make sure the Ollama server is running (usually auto-starts after install)
ollama serve &

# 2. Pull a model if you haven't already
ollama pull llama3.2:1b

# 3. Just launch the app — no flags needed, it auto-detects Ollama on
#    localhost:11434 as long as ANTHROPIC_API_KEY isn't set
python3 app.py
```

To use a different model or a remote Ollama host:
```bash
export OLLAMA_MODEL=llama3.2:3b        # or whatever `ollama list` shows
export OLLAMA_HOST=http://localhost:11434
python3 app.py
```

## Running it

```bash
pip install -r requirements.txt
python3 ingest.py        # chunk the manuals
python3 build_index.py   # build BM25 + FAISS/LSA indexes
python3 app.py            # launch the Gradio UI on http://localhost:7860
```

Optional — for synthesized (non-extractive) answers:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 app.py
```

## Extending the corpus

Add a new vendor manual by dropping a `.txt` file into `manuals/` with the
same header format:

```
MANUAL: <title>
VENDOR: <vendor name>
PRODUCT LINE: <product line>
DOC TYPE: Technician Reference Manual

SECTION 1: <title>
<content>

SECTION 2: <title>
<content>
```

Then re-run `ingest.py` and `build_index.py`. No code changes needed —
chunking, indexing, and retrieval all key off the `SECTION N:` structure.

## Swapping in a stronger embedding model

The current dense index (TF-IDF + LSA) was a deliberate choice for a fully
offline prototype. For production, `build_index.py`'s dense-index block can
be swapped for a real sentence-transformer (e.g. `all-MiniLM-L6-v2`) with
~15 lines of changes — the FAISS index, hybrid fusion logic in
`rag_engine.py`, and the rest of the pipeline are unaffected, since both
approaches just produce a normalized vector per chunk.

## Deploying

Same pattern as prior projects — push to a Hugging Face Space (Gradio SDK)
for a shareable public demo link, no separate hosting needed.

## Roadmap ideas (not built yet, worth discussing)

- Add a feedback thumbs-up/down per answer to flag manual gaps
- Auto-ingest the actual PDF manuals ITA already has on hand (via the `pdf`
  extraction pipeline) instead of hand-written reference docs
- Track which questions get asked most — surfaces which product lines
  generate the most support load, which is useful business data on its own
- Wire into the existing customer/ticket system so techs get this inline
  during a call instead of as a separate tool
