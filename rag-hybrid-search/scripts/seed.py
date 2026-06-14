"""Seed the index with the bundled sample documentation corpus."""
import sys
from app.ingestion.indexer import index_path


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "recursive"
    result = index_path("data/sample_docs", strategy=strategy, recreate=True)
    print("Seeded:", result)


if __name__ == "__main__":
    main()
