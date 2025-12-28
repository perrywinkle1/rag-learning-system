# RAG Learning System

A self-improving Retrieval-Augmented Generation (RAG) system that gets smarter with every user interaction.

## What is This?

Traditional RAG systems retrieve documents and generate answers, but they never learn whether their results were actually helpful. This system closes that loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP RAG                          │
│                                                             │
│   User Query ──► Retrieve Docs ──► Generate Answer          │
│        ▲                                  │                 │
│        │                                  ▼                 │
│   Improved    ◄── Train Model ◄── Collect Feedback          │
│   Retrieval                                                 │
└─────────────────────────────────────────────────────────────┘
```

**The Problem:** Static RAG systems don't improve. They retrieve the same quality results whether users find them helpful or not.

**The Solution:** This system collects implicit signals (clicks, dwell time) and explicit feedback (thumbs up/down, ratings), then uses contrastive learning to continuously improve the embedding model.

## Key Features

- **Vector Search**: Semantic similarity search using Qdrant
- **Re-ranking**: Cross-encoder models refine initial retrieval results
- **LLM Generation**: OpenAI or Anthropic for response synthesis with citations
- **Feedback Collection**: Track clicks, ratings, and behavioral signals
- **Continuous Learning**: Contrastive training on feedback pairs
- **A/B Testing**: Validate model improvements before deployment
- **Auto-Rollback**: Revert if new models underperform

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key or Anthropic API key

### 1. Clone the Repository

```bash
git clone https://github.com/perrywinkle1/rag-learning-system.git
cd rag-learning-system
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=your-key-here
```

### 3. Start the System

```bash
docker-compose up --build
```

This starts:
- **API Server** on http://localhost:8000
- **PostgreSQL** (document metadata, feedback storage)
- **Qdrant** (vector store) on http://localhost:6333
- **Kafka** (event streaming for feedback)
- **Redis** (caching)

### 4. Verify It's Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0", "services": {...}}
```

### 5. Open the API Docs

Navigate to http://localhost:8000/docs for interactive Swagger documentation.

## Usage Examples

### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Return Policy",
    "content": "Items can be returned within 30 days for a full refund. Products must be unused and in original packaging.",
    "source_type": "manual"
  }'
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How long do I have to return an item?"
  }'
```

Response:
```json
{
  "response_id": "abc-123",
  "answer": "You have 30 days to return an item for a full refund. The product must be unused and in its original packaging. [1]",
  "sources": [...],
  "confidence": 0.85
}
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "abc-123",
    "type": "thumbs_up"
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Submit a query, get AI-generated answer |
| `/api/v1/query/{id}` | GET | Retrieve a previous response |
| `/api/v1/feedback` | POST | Submit feedback on a response |
| `/api/v1/documents` | POST | Ingest a new document |
| `/api/v1/documents` | GET | List all documents |
| `/api/v1/documents/{id}` | GET | Get document details |
| `/api/v1/documents/{id}` | DELETE | Remove a document |
| `/api/v1/admin/metrics` | GET | System performance metrics |
| `/api/v1/admin/learning/status` | GET | Learning pipeline status |
| `/api/v1/admin/learning/trigger` | POST | Manually trigger training |

## Architecture

```
src/
├── api/            # FastAPI endpoints
├── core/           # RAG components (embedder, retriever, reranker, generator)
├── services/       # Business logic (caching, feedback, documents)
├── learning/       # ML training pipeline
└── models/         # Database models
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Vector Store | Qdrant |
| Database | PostgreSQL |
| Message Queue | Kafka |
| Cache | Redis |
| Embeddings | sentence-transformers |
| Re-ranking | Cross-encoders |
| LLM | OpenAI / Anthropic |

## How the Learning Loop Works

1. **Feedback Collection**: Every interaction generates signals
   - Explicit: thumbs up/down, star ratings
   - Implicit: clicks on sources, time spent reading

2. **Pair Generation**: Create training pairs
   - Positive: query + document user engaged with
   - Negative: query + document user ignored

3. **Contrastive Training**: Fine-tune the embedding model
   - Push positive pairs closer in vector space
   - Push negative pairs further apart

4. **Evaluation**: Compare new model against baseline
   - Metrics: MRR, NDCG, Recall@K

5. **Deployment**: If improved, roll out via A/B test
   - Automatic rollback if metrics degrade

## Running Tests

```bash
docker-compose run --rm app pytest tests/ -v
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://rag:rag_password@postgres:5432/rag_learning` |
| `QDRANT_URL` | Qdrant server URL | `http://qdrant:6333` |
| `REDIS_URL` | Redis connection | `redis://redis:6379` |

## Success Metrics

The system is working when:
- MRR@10 improves >15% after 10,000 interactions
- User satisfaction (thumbs up) exceeds 75%
- Query abandonment decreases >20%

## License

MIT

## Contributing

Contributions welcome! Please read the specs in `/technical` and `/api` before submitting PRs.
