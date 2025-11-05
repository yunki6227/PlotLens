"""
Step 2 (ML path): Entities + Coreference + Alias Clustering

- spaCy NER (en_core_web_sm by default)
- fastcoref (optional): if present, we map pronouns to their entity clusters
- Alias clustering joins close surface forms

Public API:
    build_entity_catalog(chapters, model_name="en_core_web_sm") -> dict
"""

from __future__ import annotations
import os
import re
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ---------------------------
# Data classes
# ---------------------------

@dataclass
class Mention:
    chapter: int
    sent_idx: int
    text: str
    ent_type: str  # "PERSON" | "LOC" | "ORG" | "MISC"


@dataclass
class Entity:
    id: str
    ent_type: str
    canonical: str
    aliases: List[str] = field(default_factory=list)
    first_chapter: int = 10**9
    last_chapter: int = -1
    mentions: List[Mention] = field(default_factory=list)

# ---------------------------
# Utilities
# ---------------------------

_PUNCT_RE = re.compile(r"[^\w\s'-]")

def _norm_alias(s: str) -> str:
    s = s.strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _ent_priority(ent_type: str) -> int:
    order = {"PERSON": 0, "LOC": 1, "ORG": 2, "MISC": 3}
    return order.get(ent_type, 9)

# ---------------------------
# ML components (lazy init)
# ---------------------------

_SPACY_NLP = None
_COREF = None
_COREF_OK = False

def _init_models(model_name: str = "en_core_web_sm"):
    """Initialize spaCy and optional fastcoref once."""
    global _SPACY_NLP, _COREF, _COREF_OK

    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load(model_name)

        # Try fastcoref; it's optional
        try:
            from fastcoref import FCoref
            _COREF = FCoref()
            _COREF_OK = True
        except Exception:
            _COREF = None
            _COREF_OK = False

# ---------------------------
# Extraction
# ---------------------------

def _spacy_ner_mentions(chapters) -> List[Mention]:
    """Run NER chapter-by-chapter and produce Mention objects."""
    mentions: List[Mention] = []
    for ch in chapters:
        doc = _SPACY_NLP(ch.text)
        # Build sentence list with start/end token indices to map ent → sent_idx
        sent_bounds = []
        for si, s in enumerate(doc.sents):
            sent_bounds.append((si, s.start, s.end))

        for ent in doc.ents:
            label = ent.label_
            if label in ("PERSON", "ORG", "GPE", "LOC"):
                ent_type = "PERSON" if label == "PERSON" else ("LOC" if label in ("GPE", "LOC") else "ORG")
                # Find sentence index containing this entity
                sent_idx = 0
                for si, st, en in sent_bounds:
                    if ent.start >= st and ent.end <= en:
                        sent_idx = si
                        break
                mentions.append(Mention(chapter=ch.id, sent_idx=sent_idx, text=ent.text, ent_type=ent_type))
    return mentions


def _apply_coref(chapters, mentions: List[Mention]) -> List[Mention]:
    """
    If fastcoref is available, try to map pronouns to their cluster heads (nearest named span).
    We *add* pronoun mentions as MISC tied to the chapter/sentence; clustering will merge them.
    """
    if not _COREF_OK:
        return mentions

    extra: List[Mention] = []
    for ch in chapters:
        # fastcoref wants raw text; it returns clusters within the doc
        res = _COREF.predict(
            texts=[ch.text],
            max_dist=1000,  # generous for web-novel paragraphs
            doc_stride=128,
        )[0]

        # res.get_clusters_as_strings() returns clusters of strings;
        # res.clusters gives token span indices. We'll attach pronoun strings by sentence.
        # We’ll record pronoun-like spans (lowercase pronouns) as MISC mentions.
        # Named spans are already handled by NER; clustering later will merge via string similarity.

        # Build a naive sentence index by splitting on punctuation,
        # but better is using spaCy’s sent indices again:
        doc = _SPACY_NLP(ch.text)
        sent_map = []
        for si, s in enumerate(doc.sents):
            sent_map.append((si, s.start_char, s.end_char))

        # Gather candidate pronoun strings from coref clusters
        for cl in res.get_clusters_as_strings():
            # cl is a list of surface strings that coref considers equivalent
            for surface in cl:
                surf = surface.strip()
                # very rough pronoun filter
                if re.fullmatch(r"(he|him|his|she|her|hers|they|them|their|it|its|i|me|my|we|us|our)", surf.lower()):
                    # find sentence index by char position (approximate: first occurrence)
                    pos = ch.text.find(surface)
                    sent_idx = 0
                    if pos >= 0:
                        for si, stc, enc in sent_map:
                            if stc <= pos < enc:
                                sent_idx = si
                                break
                    extra.append(Mention(chapter=ch.id, sent_idx=sent_idx, text=surf, ent_type="MISC"))

    return mentions + extra

# ---------------------------
# Alias clustering
# ---------------------------

def _cluster_aliases(mentions: List[Mention]) -> List[Entity]:
    """
    Merge mentions into entities using fuzzy rules:
      - exact normalized match
      - difflib similarity >= 0.86
      - prefer longer multi-token canonical
    Pronouns (MISC) will merge to the best-scoring existing PERSON entity.
    """
    groups: List[Entity] = []

    def add(m: Mention):
        nonlocal groups
        m_norm = _norm_alias(m.text)
        target_type = "PERSON" if m.ent_type in ("PERSON", "MISC") else m.ent_type

        best_i = -1
        best_score = 0.0
        for i, g in enumerate(groups):
            if g.ent_type != target_type:
                continue
            candidates = [g.canonical] + g.aliases
            score = max(difflib.SequenceMatcher(None, _norm_alias(c), m_norm).ratio() for c in candidates) if candidates else 0.0
            if score > best_score:
                best_score = score
                best_i = i

        if best_i >= 0 and (best_score >= 0.86 or m_norm in [_norm_alias(a) for a in groups[best_i].aliases + [groups[best_i].canonical]]):
            g = groups[best_i]
            g.mentions.append(m)
            g.first_chapter = min(g.first_chapter, m.chapter)
            g.last_chapter = max(g.last_chapter, m.chapter)
            if m.ent_type != "MISC" and _norm_alias(m.text) != _norm_alias(g.canonical):
                if m.text not in g.aliases:
                    g.aliases.append(m.text)
            if len(m.text.split()) > len(g.canonical.split()) and m.ent_type != "MISC":
                g.canonical = m.text
        else:
            ent_id = f"E{len(groups)+1}"
            g = Entity(id=ent_id, ent_type=target_type, canonical=m.text,
                       aliases=[], mentions=[m],
                       first_chapter=m.chapter, last_chapter=m.chapter)
            groups.append(g)

    for m in mentions:
        add(m)

    # dedupe aliases
    for g in groups:
        seen = set()
        uniq = []
        for a in [g.canonical] + g.aliases:
            k = _norm_alias(a)
            if k not in seen:
                seen.add(k)
                if a != g.canonical:
                    uniq.append(a)
        g.aliases = uniq
    return groups

# ---------------------------
# Public API
# ---------------------------

def build_entity_catalog(chapters, model_name: str = "en_core_web_sm") -> Dict[str, Any]:
    """
    Build an ML-driven entity catalog. Uses spaCy NER; uses fastcoref if available.

    Returns:
      {
        "entities": [{ id, type, canonical, aliases, first_chapter, last_chapter, mentions: [...] }]
      }
    """
    _init_models(model_name=model_name)
    ner_mentions = _spacy_ner_mentions(chapters)
    all_mentions = _apply_coref(chapters, ner_mentions)
    entities = _cluster_aliases(all_mentions)
    entities.sort(key=lambda e: (_ent_priority(e.ent_type), e.first_chapter, e.id))
    return {
        "entities": [
            {
                "id": e.id,
                "type": e.ent_type,
                "canonical": e.canonical,
                "aliases": e.aliases,
                "first_chapter": e.first_chapter,
                "last_chapter": e.last_chapter,
                "mentions": [
                    {"chapter": m.chapter, "sent_idx": m.sent_idx, "text": m.text, "ent_type": m.ent_type}
                    for m in e.mentions
                ],
            } for e in entities
        ]
    }