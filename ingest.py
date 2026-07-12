"""
ingest_pdf.py — ITA Technician Assistant, multimodal PDF ingestion

Replaces/extends ingest.py for real vendor PDFs (as opposed to the hand-
written SECTION-tagged .txt corpus). Handles three observations from the
actual manuals:

  1. Figures are frequently VECTOR-DRAWN diagrams, not embedded raster
     images. `page.get_images()` misses these entirely. Fix: rasterize the
     bounding-box region of every image-type block with `get_pixmap(clip=...)`.
     This works uniformly for vector diagrams AND embedded bitmap photos.

  2. Caption conventions differ per vendor and can't be regex-matched:
       - Araknis:  explicit "Figure N. Caption" text below the image
       - Josh.ai:  no caption at all; the image sits inline between bullets
       - Lutron:   a step/section header above a cluster of diagrams, no
                   per-image numbering
     Fix: POSITIONAL association. Sort all page blocks (text + image) by
     reading order (top-to-bottom, left-to-right) and treat the nearest
     preceding non-boilerplate text block as the image's context, rather
     than pattern-matching a specific caption style.

  3. No manual uses a consistent machine-parseable "SECTION N:" header like
     the synthetic corpus did. Fix: detect section headings via a font-size
     heuristic (the largest/boldest span near the top of a text block is
     treated as a heading) instead of vendor-specific regex.

Output: a chunks.jsonl compatible with the existing Chunk schema, extended
with an `image_paths` field, plus the extracted figure PNGs on disk.
"""

import os
import re
import json
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(__file__)
MANUALS_DIR = os.path.join(BASE_DIR, "manuals_pdf")
IMAGES_DIR = os.path.join(BASE_DIR, "manuals_images")
OUTPUT_PATH = os.path.join(BASE_DIR, "index", "chunks.jsonl")

# Boilerplate patterns seen across headers/footers on every page of every
# vendor manual — these must never be treated as section headings or as
# the "caption" for a figure.
BOILERPLATE_PATTERNS = [
    r"^-Return to Table of Contents-$",
    r"^©\s?\d{4}",
    r"^Product Manual$",
    r"^\d{1,4}$",  # bare page numbers
    r"White Paper",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

CHUNK_WORD_LIMIT = 220
CHUNK_OVERLAP = 40
FIGURE_ROW_GAP_PX = 15       # merge image blocks into one figure if within this y-gap
FIGURE_CLIP_PADDING = 6      # px padding around rasterized figure bbox
HEADING_SIZE_PERCENTILE = 0.85  # spans in the top 15% of font sizes seen = heading


@dataclass
class Chunk:
    chunk_id: str
    vendor: str
    product_line: str
    doc_title: str
    section_title: str
    source_file: str
    text: str
    image_paths: List[str] = field(default_factory=list)


def is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_RE.search(text.strip()))


TOP_MARGIN_PX = 55    # running headers (logo + doc title) live above this
BOTTOM_MARGIN_PX = 60  # footers (page number, copyright, TOC link) live below page_height - this


def get_page_blocks(page):
    """Return text and image blocks in reading order, each tagged with
    bbox and (for text) the max font size present, used later for heading
    detection."""
    raw = page.get_text("dict")
    page_height = page.rect.height
    blocks = []
    for b in raw["blocks"]:
        if b["type"] == 0:  # text
            spans = [s for l in b["lines"] for s in l["spans"]]
            text = "".join(s["text"] for s in spans).strip()
            if not text or is_boilerplate(text):
                continue
            # skip running headers/footers by position, not just regex —
            # catches glued-together header text like
            # "Araknis Networks Wireless Access PointProduct ManualMounting"
            # that free-text patterns won't match
            y0 = b["bbox"][1]
            if y0 < TOP_MARGIN_PX or y0 > page_height - BOTTOM_MARGIN_PX:
                continue
            max_size = max((s["size"] for s in spans), default=0)
            blocks.append({
                "type": "text", "bbox": b["bbox"], "text": text, "max_size": max_size,
            })
        elif b["type"] == 1:  # image (raster OR the bbox of a vector drawing region)
            blocks.append({"type": "image", "bbox": b["bbox"]})
    blocks.sort(key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))
    return blocks


def cluster_figures(blocks):
    """Merge adjacent image blocks (e.g. Figure 6A/6B/6C side by side) into
    single figure groups, and note the nearest preceding text block as
    context for each group."""
    figures = []
    i = 0
    last_text_idx = None
    while i < len(blocks):
        b = blocks[i]
        if b["type"] == "text":
            last_text_idx = i
            i += 1
            continue
        # start of an image cluster
        cluster = [b]
        j = i + 1
        while j < len(blocks) and blocks[j]["type"] == "image":
            prev_bottom = cluster[-1]["bbox"][3]
            gap = blocks[j]["bbox"][1] - prev_bottom
            if gap < FIGURE_ROW_GAP_PX or blocks[j]["bbox"][1] < prev_bottom:
                cluster.append(blocks[j])
                j += 1
            else:
                break
        xs = [c["bbox"][0] for c in cluster] + [c["bbox"][2] for c in cluster]
        ys = [c["bbox"][1] for c in cluster] + [c["bbox"][3] for c in cluster]
        context_text = blocks[last_text_idx]["text"] if last_text_idx is not None else ""
        figures.append({
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "context_text": context_text,
        })
        i = j
    return figures


def detect_heading(blocks) -> Optional[str]:
    """Pick the block with the largest font size as the page's heading,
    used as section_title when a vendor doesn't use numbered SECTION
    headers."""
    text_blocks = [b for b in blocks if b["type"] == "text"]
    if not text_blocks:
        return None
    sizes = [b["max_size"] for b in text_blocks]
    threshold = statistics.quantiles(sizes, n=100)[int(HEADING_SIZE_PERCENTILE * 100) - 1] \
        if len(sizes) > 1 else sizes[0]
    candidates = [b for b in text_blocks if b["max_size"] >= threshold]
    candidates.sort(key=lambda b: b["bbox"][1])
    return candidates[0]["text"][:120] if candidates else text_blocks[0]["text"][:120]


def rasterize_figure(page, bbox, out_path, dpi=200):
    pad = FIGURE_CLIP_PADDING
    clip = fitz.Rect(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    clip = clip & page.rect  # clamp to page bounds
    pix = page.get_pixmap(clip=clip, dpi=dpi)
    pix.save(out_path)


def split_long_text(text: str, limit: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= limit:
        return [text]
    out, start = [], 0
    while start < len(words):
        end = min(start + limit, len(words))
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return out


def ingest_pdf(filepath: str, vendor: str, product_line: str, doc_title: str) -> List[Chunk]:
    fname = os.path.basename(filepath)
    slug = os.path.splitext(fname)[0]
    img_out_dir = os.path.join(IMAGES_DIR, slug)
    os.makedirs(img_out_dir, exist_ok=True)

    doc = fitz.open(filepath)
    chunks: List[Chunk] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = get_page_blocks(page)
        if not blocks:
            continue

        heading = detect_heading(blocks) or doc_title
        figures = cluster_figures(blocks)

        # Rasterize each figure on this page, collect paths + their context text
        image_paths = []
        figure_context_texts = []
        for fig_idx, fig in enumerate(figures):
            out_path = os.path.join(img_out_dir, f"p{page_idx+1}_fig{fig_idx+1}.png")
            try:
                rasterize_figure(page, fig["bbox"], out_path)
                image_paths.append(out_path)
                if fig["context_text"]:
                    figure_context_texts.append(fig["context_text"])
            except Exception as e:
                print(f"[ingest_pdf] figure rasterize failed p{page_idx+1}: {e}")

        # Page body text = all text blocks, in reading order
        body_text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
        if not body_text.strip() and not image_paths:
            continue

        sub_texts = split_long_text(body_text, CHUNK_WORD_LIMIT, CHUNK_OVERLAP)
        for sub_idx, sub_text in enumerate(sub_texts):
            cid = f"{fname}::p{page_idx+1}::part{sub_idx}"
            # Only attach images to the chunk whose text actually contains
            # the figure's context text (avoids dumping every page image
            # onto every sub-chunk of a long page)
            attached = [
                p for p, ctx in zip(image_paths, figure_context_texts + [""] * (len(image_paths) - len(figure_context_texts)))
                if not ctx or ctx[:40] in sub_text
            ] if sub_idx == 0 else []
            if sub_idx == 0 and not attached:
                attached = image_paths  # fallback: single-chunk pages keep all figures

            chunks.append(Chunk(
                chunk_id=cid,
                vendor=vendor,
                product_line=product_line,
                doc_title=doc_title,
                section_title=heading,
                source_file=fname,
                text=sub_text.strip(),
                image_paths=attached,
            ))

    doc.close()
    return chunks


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Vendor metadata per source file — extend as more PDFs are added
    VENDOR_MAP = {
        "araknis_network.pdf": ("Araknis Networks (Snap One)", "Wireless Access Point", "AN-100/300 Product Manual"),
        "josh_ai_lutron_white_paper.pdf": ("Josh.ai", "Voice Control", "Lutron Programming for Voice Control"),
        "lutron_caseta.pdf": ("Lutron", "Caseta Wireless", "Caseta Wireless Advanced Installation Guide"),
        "sonos.pdf": ("Sonos", "Sonos System", "Sonos Product Manual"),
    }

    all_chunks: List[Chunk] = []
    if not os.path.isdir(MANUALS_DIR):
        print(f"[ingest_pdf] no {MANUALS_DIR} found, nothing to do")
        return

    for fname in sorted(os.listdir(MANUALS_DIR)):
        if not fname.endswith(".pdf"):
            continue
        vendor, product_line, doc_title = VENDOR_MAP.get(
            fname, ("Unknown", "", fname)
        )
        filepath = os.path.join(MANUALS_DIR, fname)
        print(f"[ingest_pdf] ingesting {fname} ...")
        chunks = ingest_pdf(filepath, vendor, product_line, doc_title)
        print(f"  -> {len(chunks)} chunks, "
              f"{sum(len(c.image_paths) for c in chunks)} figures attached")
        all_chunks.extend(chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"[ingest_pdf] wrote {len(all_chunks)} total chunks -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()