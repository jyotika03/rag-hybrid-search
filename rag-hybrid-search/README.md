# RAG Pipeline with Hybrid Search Over Internal Docs

A production-grade Retrieval-Augmented Generation system that ingests internal
documentation, indexes it with **both dense vector and sparse keyword search**,
retrieves the most relevant context with **Reciprocal Rank Fusion + cross-encoder
reranking**, and generates grounded answers with **verified inline citations** and
a **composite confidence score**.

This repo is wired for **AWS deployment via CloudFormation** onto **ECS Fargate**,
with **Amazon OpenSearch Service** as the managed dense+sparse store.

---

## Architecture

```
            ingest                          ask
  docs --> loader --> chunker --> embed --> OpenSearch (knn_vector + BM25 text)
   md/txt/html/pdf      |  fixed/recursive/semantic        ^          ^
                        +-- dedup (cos>0.95 skip)          | dense    | sparse
                                                           +--+-------+
                                                   RRF fusion (0.7/0.3)
                                                              |
                                                   cross-encoder rerank (top 5)
                                                              |
                                              grounded generation (GPT-4o)
                                                              |
                                       citation parse --> LLM-as-judge verify
                                                              |
                                      confidence (retrieval+coverage+completeness)
```

A single OpenSearch index holds both the `knn_vector` embedding and the analyzed
`text` field, so the dense and sparse indexes **stay in sync by construction**.

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.11 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector + sparse store | Amazon OpenSearch Service (k-NN + BM25) |
| Fusion | Reciprocal Rank Fusion (configurable weights) |
| Reranker | cross-encoder (`ms-marco-MiniLM-L-6-v2`) or LLM-as-judge |
| Generation | GPT-4o (Claude Sonnet swappable) |
| API | FastAPI |
| Dashboard | Streamlit |
| Deploy | Docker, CloudFormation, ECS Fargate |

## What maps to the project brief

| Brief phase | Where |
|---|---|
| Multi-format loader + metadata | `app/ingestion/loaders.py` |
| 3 switchable chunking strategies | `app/ingestion/chunking.py` |
| Embeddings + dual index in sync | `app/ingestion/embeddings.py`, `app/store/opensearch_client.py` |
| Dedup (cos > 0.95) | `app/ingestion/dedup.py` |
| Dense / sparse retrieval | `app/retrieval/dense.py`, `app/retrieval/sparse.py` |
| RRF fusion (weighted) | `app/retrieval/fusion.py` |
| Cross-encoder / LLM reranker | `app/retrieval/rerank.py` |
| Grounded prompt + citations | `app/generation/prompts.py`, `generator.py` |
| Citation verification (LLM-as-judge) | `app/generation/citations.py` |
| Confidence scorer + "I don't know" | `app/generation/confidence.py` |
| Eval framework + chunking comparison | `app/eval/` |
| FastAPI (`/v1/ask`,`/documents`,`/ingest`) | `app/main.py` |
| Query dashboard (hybrid vs dense) | `frontend/streamlit_app.py` |
| Containerized stack + seed | `docker-compose.yml`, `scripts/seed.py` |

## Run locally

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up --build     # OpenSearch + API + Streamlit
# in another shell, seed the bundled sample corpus:
docker compose exec api python scripts/seed.py recursive
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

Ask a question:

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"What port does the API listen on?","hybrid":true}' | jq
```

## Evaluation

```bash
python -m app.eval.run_eval --corpus data/sample_docs              # full suite
python -m app.eval.run_eval --corpus data/sample_docs --compare-chunking
```

Metrics: answer correctness, faithfulness, retrieval relevance, citation accuracy.
The chunking comparison runs the same suite across fixed/recursive/semantic and
reports which strategy wins on each metric.

## Tests

```bash
pip install -r requirements.txt
pytest          # chunking, RRF fusion, BM25 fallback, citation parsing
```

These tests run **offline** (no API keys / no cluster required).

## Deploy to AWS (CloudFormation + ECS Fargate)

`infra/cloudformation.yaml` provisions: VPC (2 public + 2 private subnets, NAT),
ALB, ECS Fargate cluster/service, **Amazon OpenSearch** domain (VPC, encrypted,
IAM-auth), ECR repo, Secrets Manager (API keys), IAM roles, and CloudWatch logs.

```bash
cd infra
export AWS_REGION=us-east-1
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...      # optional
./deploy.sh                               # build, push to ECR, deploy stack
```

The stack output `ApiUrl` is the public ALB endpoint. On AWS the app talks to
OpenSearch over HTTPS with **SigV4/IAM auth** (`OPENSEARCH_AWS_AUTH=true`), so no
basic-auth password is stored.

### Cost note
Amazon OpenSearch (2x `t3.small.search`) and a NAT gateway are the main standing
costs. Tear down with `aws cloudformation delete-stack --stack-name rag-hybrid-search-stack`.

## Implemented vs. simplified

Fully implemented: all retrieval/generation/citation/confidence logic, eval suite,
API, dashboard, Docker, CloudFormation. Simplified for portfolio scope: the
semantic chunker uses greedy sentence-similarity merging (not a trained boundary
model); reranking defaults to a small cross-encoder; the golden dataset ships 5
representative cases (the brief asks for 50+ - extend `app/eval/golden_dataset.json`).
