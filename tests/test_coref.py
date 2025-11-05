from engine.ingest import ingest_plaintext
from engine.coref import build_entity_catalog

SAMPLE = (
    "Sung Jinwoo entered the dungeon. Jinwoo looked around. "
    "The Hunter Association warned Jinwoo. "
    "He nodded and followed Yoo Jinho."
)

def test_build_entity_catalog_ml(tmp_path):
    # Create two tiny chapters (blank line split)
    p = tmp_path / "mini.txt"
    p.write_text(SAMPLE.replace(". ", ".\n\n"), encoding="utf-8")
    chapters = ingest_plaintext(str(p), min_blanklines=1)

    catalog = build_entity_catalog(chapters, model_name="en_core_web_sm")
    assert "entities" in catalog
    ents = catalog["entities"]
    assert len(ents) >= 1

    # We expect at least one PERSON entity representing Jinwoo
    joined = " ".join(e["canonical"] + " " + " ".join(e["aliases"]) for e in ents)
    assert ("Jinwoo" in joined) or ("Sung Jinwoo" in joined)
    print(ents)
    # Mentions should capture chapter indices
    for e in ents:
        assert e["first_chapter"] <= e["last_chapter"]