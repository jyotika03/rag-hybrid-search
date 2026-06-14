from typing import Optional, Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source_document: str
    chunk_index: int
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    chunking_strategy: str
    char_count: int
    embedding: Optional[list[float]] = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    marker: int
    chunk_id: str
    source_document: str
    claim: str
    supported: Optional[bool] = None
    verifier_reasoning: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    hybrid: bool = True
    verify_citations: bool = True
    chunking_strategy: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    answered: bool
    citations: list[Citation]
    confidence: ConfidenceBreakdown
    retrieved: list[RetrievedChunk]
    notes: Optional[str] = None


class IngestRequest(BaseModel):
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    reindex: bool = False


class DocumentInfo(BaseModel):
    source_document: str
    chunk_count: int
    strategies: list[str]
