"""
CLI to inspect big TXT ingestion.

Usage:
  python3 -m app.cli data/samples/solo_leveling.txt --blankline --min-blank 1
"""
import argparse
from engine.ingest import ingest_plaintext

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to TXT manuscript")
    p.add_argument("--blankline", action="store_true", help="Use blank-line splitting")
    p.add_argument("--min-blank", type=int, default=1, help="Empty lines needed for break")
    args = p.parse_args()

    strategy = "blankline" if args.blankline else "auto"
    chapters = ingest_plaintext(args.path, strategy=strategy, min_blanklines=args.min_blank)

    total_chars = sum(len(c.text) for c in chapters)
    total_sents = sum(len(c.sentences) for c in chapters)

    print(f"Chapters: {len(chapters)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total sentences: {total_sents:,}")
    print("First 3 titles:", [c.title for c in chapters[:3]])

if __name__ == "__main__":
    main()