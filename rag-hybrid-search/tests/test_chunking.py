from app.ingestion.loaders import LoadedDocument, Section
from app.ingestion.chunking import fixed_size, recursive


def _doc():
    return LoadedDocument("d.md", [
        Section("A" * 2500, "d.md", "Intro"),
        Section("Short section.", "d.md", "Notes"),
    ])


def test_fixed_size_overlap():
    chunks = fixed_size(_doc(), size=1000, overlap=100)
    assert all(c.chunking_strategy == "fixed" for c in chunks)
    assert all(c.char_count <= 1000 for c in chunks)
    assert len(chunks) >= 3


def test_recursive_preserves_headings():
    chunks = recursive(_doc(), size=1000, overlap=100)
    assert any(c.section_heading == "Notes" for c in chunks)
    assert all(c.chunking_strategy == "recursive" for c in chunks)
