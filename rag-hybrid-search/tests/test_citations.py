from app.models import RetrievedChunk, Chunk
from app.generation.citations import parse_citations


def _rc(i, text):
    return RetrievedChunk(chunk=Chunk(id=str(i), text=text, source_document="d.md",
                                      chunk_index=i, chunking_strategy="recursive",
                                      char_count=len(text)))


def test_parse_citations_maps_markers():
    chunks = [_rc(0, "Port 8000."), _rc(1, "Uses RRF.")]
    answer = "The API uses port 8000 [1]. Retrieval is hybrid via RRF [2]."
    cits = parse_citations(answer, chunks)
    markers = sorted(c.marker for c in cits)
    assert markers == [1, 2]
    assert cits[0].chunk_id == "0"
