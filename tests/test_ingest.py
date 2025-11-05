import os
from engine.ingest import load_text, split_chapters, split_sentences, ingest_plaintext

SAMPLE_MINI = """First block of text.
Still the same chapter.

Second block starts here.
And continues.

Third block here."""

def test_load_text_normalizes_newlines(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_bytes(b"Hello\r\nWorld\r\n")
    text = load_text(str(p))
    assert text == "Hello\nWorld"

def test_split_chapters_blankline_basic():
    chunks = split_chapters(SAMPLE_MINI, min_blanklines=1)
    assert len(chunks) == 3
    assert [c["title"] for c in chunks] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert "First block" in chunks[0]["text"]
    assert "Second block" in chunks[1]["text"]

def test_split_chapters_blankline_requires_more_gaps():
    txt = "A\n\nB\n\n\nC\n\n\nD"
    c1 = split_chapters(txt, min_blanklines=1)
    c2 = split_chapters(txt, min_blanklines=2)
    assert len(c1) == 4
    assert len(c2) == 3

def test_split_sentences_handles_abbrevs():
    sents = split_sentences("He met Mr. Kim. Dr. Cho waved.")
    assert sents == ["He met Mr. Kim.", "Dr. Cho waved."]

def test_split_sentences_handles_control_chars():
    dirty = "1.\x1c1 He looks around. It is dark.\nAnother line."
    sents = split_sentences(dirty)
    joined = " ".join(sents)
    assert "He looks around." in joined
    assert "It is dark." in joined

def test_ingest_plaintext_end_to_end(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(SAMPLE_MINI, encoding="utf-8")
    chapters = ingest_plaintext(str(p), min_blanklines=1)
    assert len(chapters) == 3
    assert chapters[0].id == 1
    assert any("First block" in s or "First block" in chapters[0].text for s in chapters[0].sentences)

def test_real_file_if_present():
    sample_path = os.path.join("data", "samples", "solo_leveling.txt")
    if os.path.exists(sample_path):
        chs = ingest_plaintext(sample_path, min_blanklines=1)
        assert len(chs) >= 10
        total_sents = sum(len(c.sentences) for c in chs)
        assert total_sents > 200