"""
Step 1: Ingest & segmentation (TXT → Chapters → Sentences)

- Uses ONLY blank lines to split chapters (web-novel friendly).
- No scanning for 'CHAPTER' or headings.
- Robust sentence splitting with sanitizer + pysbd (fallback included).

Public API:
  load_text(path)
  split_chapters(text, min_blanklines=1)
  split_sentences(chapter_text)
  ingest_plaintext(path, min_blanklines=1)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict
import pysbd

_SEGMENTER = pysbd.Segmenter(language="en", clean=True)

@dataclass
class Chapter:
    id: int
    title: str
    text: str
    sentences: List[str]


def load_text(path: str) -> str:
    """Read UTF-8 text and normalize newlines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return re.sub(r"\r\n?", "\n", raw).strip()


def split_chapters(text: str, min_blanklines: int = 1) -> List[Dict[str, str]]:
    """
    Split manuscript into chapters by consecutive blank lines only.

    Args:
      min_blanklines: number of empty lines that separate chapters.
                      1 → single blank line means a chapter break.
    """
    gap_n = max(1, min_blanklines) + 1
    parts = [b.strip() for b in re.split(rf"\n{{{gap_n},}}", text) if b.strip()]
    chunks = [{"title": "", "text": p} for p in parts]
    # Normalize titles (purely index-based)
    for i, ch in enumerate(chunks, start=1):
        ch["title"] = f"Chapter {i}"
    return chunks


def _sanitize_for_segmentation(text: str) -> str:
    """
    Remove troublesome control characters that can break pysbd, preserving \n and \t.
    Also normalizes odd Unicode paragraph separators and whitespace.
    """
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", " ", text)
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(chapter_text: str) -> List[str]:
    """Sentence segmentation with pysbd + robust sanitizer and fallback."""
    clean = _sanitize_for_segmentation(chapter_text)
    try:
        sentences = _SEGMENTER.segment(clean)
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", clean)
        sentences = [p.strip() for p in parts if len(p.strip()) >= 2]

    out = []
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) >= 2:
            out.append(s)
    return out


def ingest_plaintext(path: str, min_blanklines: int = 1) -> List[Chapter]:
    """End-to-end ingestion pipeline using blank-line-only chapter splits."""
    text = load_text(path)
    raw_chapters = split_chapters(text, min_blanklines=min_blanklines)
    chapters: List[Chapter] = []
    for i, ch in enumerate(raw_chapters, start=1):
        sentences = split_sentences(ch["text"])
        chapters.append(Chapter(id=i, title=ch["title"], text=ch["text"], sentences=sentences))
    return chapters