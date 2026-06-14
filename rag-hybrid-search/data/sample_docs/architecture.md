# System Architecture

The RAG pipeline indexes chunks using the text-embedding-3-small model and stores
them in Amazon OpenSearch. The same index backs both dense k-NN search and BM25
keyword search, so the dense and sparse indexes stay in sync.

## Deduplication
Before a chunk is indexed, near-duplicate detection runs. Any chunk whose cosine
similarity exceeds 0.95 against an already-indexed chunk is skipped.

## Hybrid Retrieval
Dense and sparse result lists are merged with Reciprocal Rank Fusion, then the top
candidates are reranked by a cross-encoder before generation.
