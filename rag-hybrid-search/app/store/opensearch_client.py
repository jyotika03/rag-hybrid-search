"""OpenSearch client + index management.

Amazon OpenSearch Service is the managed store. A single index holds both the
dense vector (knn_vector) and the analyzed text field, so the same documents
back both dense (k-NN) and sparse (BM25) retrieval. The two indexes therefore
stay in sync by construction.
"""
from __future__ import annotations

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from app.config import get_settings


def _auth():
    s = get_settings()
    if s.opensearch_aws_auth:
        # IAM / SigV4 auth for Amazon OpenSearch Service
        import boto3
        from requests_aws4auth import AWS4Auth

        cred = boto3.Session().get_credentials()
        return AWS4Auth(
            cred.access_key,
            cred.secret_key,
            s.aws_region,
            "es",
            session_token=cred.token,
        )
    return (s.opensearch_user, s.opensearch_password)


def get_client() -> OpenSearch:
    s = get_settings()
    return OpenSearch(
        hosts=[{"host": s.opensearch_host, "port": s.opensearch_port}],
        http_auth=_auth(),
        use_ssl=s.opensearch_use_ssl,
        verify_certs=s.opensearch_verify_certs,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def index_body(dim: int) -> dict:
    return {
        "settings": {
            "index": {"knn": True, "knn.algo_param.ef_search": 100},
            "analysis": {
                "analyzer": {
                    "default": {"type": "standard"}
                }
            },
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "english"},
                "source_document": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "section_heading": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "chunking_strategy": {"type": "keyword"},
                "char_count": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }


def ensure_index(client: OpenSearch | None = None, recreate: bool = False) -> None:
    s = get_settings()
    client = client or get_client()
    exists = client.indices.exists(index=s.opensearch_index)
    if exists and recreate:
        client.indices.delete(index=s.opensearch_index)
        exists = False
    if not exists:
        client.indices.create(index=s.opensearch_index, body=index_body(s.embedding_dim))


def bulk_index(chunks: list[dict], client: OpenSearch | None = None) -> int:
    s = get_settings()
    client = client or get_client()
    actions = [
        {"_index": s.opensearch_index, "_id": c["id"], "_source": c} for c in chunks
    ]
    ok, _ = helpers.bulk(client, actions, refresh=True)
    return ok
