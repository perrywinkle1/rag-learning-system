# API Spec: Closed-Loop RAG Learning System

## Base Path
`/api/v1`

## Authentication

All endpoints require authentication via Bearer token in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

Tokens are obtained via OAuth 2.0 flow or API key exchange. Tokens include:
- `sub`: User/service identifier
- `scope`: Permissions (query, feedback, admin)
- `exp`: Expiration timestamp

## Rate Limits

| Scope | Limit | Window | Burst |
|-------|-------|--------|-------|
| query | 100 | 1 minute | 20 |
| feedback | 500 | 1 minute | 50 |
| admin | 60 | 1 minute | 10 |
| ingest | 1000 | 1 hour | 100 |

Rate limit headers returned on all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

---

## Query Endpoints

### `POST /query`

Submit a query and receive a generated response with sources.

**Request:**
```json
{
  "query": "string (required) - The user's question or search query",
  "session_id": "string (optional) - Session identifier for context continuity",
  "top_k": "number (optional, default: 10) - Number of sources to retrieve",
  "filters": {
    "document_ids": ["string"] ,
    "source_types": ["string"],
    "date_range": {
      "start": "ISO8601",
      "end": "ISO8601"
    },
    "metadata": { "key": "value" }
  },
  "options": {
    "include_scores": "boolean (default: false)",
    "include_embeddings": "boolean (default: false)",
    "stream": "boolean (default: false)"
  }
}
```

**Response 200:**
```json
{
  "response_id": "uuid",
  "answer": "string - Generated response text",
  "sources": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "document_title": "string",
      "content": "string - Chunk text",
      "rank": 1,
      "score": 0.89,
      "metadata": {}
    }
  ],
  "confidence": 0.85,
  "model_info": {
    "retrieval_model": "v1.2.3",
    "generation_model": "gpt-4-turbo",
    "reranker_model": "v1.0.1"
  },
  "latency_ms": 1234,
  "session_id": "uuid"
}
```

**Response 200 (Streaming):**
When `stream: true`, returns Server-Sent Events:
```
event: source
data: {"chunk_id": "uuid", "content": "...", "rank": 1, "score": 0.89}

event: source
data: {"chunk_id": "uuid", "content": "...", "rank": 2, "score": 0.85}

event: answer_start
data: {"response_id": "uuid"}

event: answer_chunk
data: {"text": "Based on the documents, "}

event: answer_chunk
data: {"text": "the answer is..."}

event: answer_end
data: {"confidence": 0.85, "latency_ms": 1234}
```

**Errors:**

| Status | Code | When |
|--------|------|------|
| 400 | INVALID_QUERY | Query empty or exceeds 10,000 chars |
| 400 | INVALID_FILTER | Malformed filter parameters |
| 401 | UNAUTHORIZED | Missing or invalid auth token |
| 403 | FORBIDDEN | Insufficient scope |
| 429 | RATE_LIMITED | Query rate limit exceeded |
| 500 | RETRIEVAL_ERROR | Vector store unavailable |
| 500 | GENERATION_ERROR | LLM generation failed |
| 503 | SERVICE_UNAVAILABLE | System overloaded |

---

### `GET /query/:response_id`

Retrieve a previous response by ID.

**Response 200:**
```json
{
  "response_id": "uuid",
  "query": "string",
  "answer": "string",
  "sources": [...],
  "confidence": 0.85,
  "created_at": "ISO8601",
  "feedback_summary": {
    "thumbs_up": 5,
    "thumbs_down": 1,
    "avg_rating": 4.2
  }
}
```

**Errors:**

| Status | Code | When |
|--------|------|------|
| 404 | NOT_FOUND | Response ID doesn't exist |

---

## Feedback Endpoints

### `POST /feedback`

Submit feedback on a response.

**Request:**
```json
{
  "response_id": "uuid (required)",
  "type": "string (required) - thumbs_up|thumbs_down|rating|click|dwell|copy|share",
  "value": "number (optional) - For rating (1-5) or dwell (ms)",
  "target_chunk_id": "uuid (optional) - For source-specific feedback",
  "reason": {
    "code": "string (optional) - wrong_answer|outdated|irrelevant|incomplete|other",
    "text": "string (optional) - Free-form explanation"
  },
  "client_timestamp": "ISO8601 (optional)"
}
```

**Response 202:**
```json
{
  "feedback_id": "uuid",
  "accepted": true,
  "message": "Feedback recorded"
}
```

**Errors:**

| Status | Code | When |
|--------|------|------|
| 400 | INVALID_TYPE | Unknown feedback type |
| 400 | INVALID_VALUE | Value out of range |
| 404 | RESPONSE_NOT_FOUND | response_id doesn't exist |
| 429 | RATE_LIMITED | Feedback rate limit exceeded |

---

### `POST /feedback/batch`

Submit multiple feedback events in one request.

**Request:**
```json
{
  "events": [
    {
      "response_id": "uuid",
      "type": "click",
      "target_chunk_id": "uuid",
      "client_timestamp": "ISO8601"
    },
    {
      "response_id": "uuid",
      "type": "dwell",
      "value": 45000,
      "client_timestamp": "ISO8601"
    }
  ]
}
```

**Response 202:**
```json
{
  "accepted": 2,
  "rejected": 0,
  "errors": []
}
```

---

## Document Management Endpoints

### `POST /documents`

Ingest a new document into the knowledge base.

**Request:**
```json
{
  "external_id": "string (optional) - Your system's document ID",
  "title": "string (required)",
  "content": "string (required) - Document text",
  "source_type": "string (required) - upload|crawl|api|manual",
  "source_url": "string (optional)",
  "language": "string (default: en) - ISO 639-1 code",
  "metadata": {
    "author": "string",
    "category": "string",
    "tags": ["string"],
    "custom_field": "any"
  },
  "chunking": {
    "strategy": "string (default: semantic) - semantic|fixed|sentence",
    "chunk_size": "number (default: 512) - Target tokens per chunk",
    "chunk_overlap": "number (default: 50) - Overlap tokens"
  }
}
```

**Response 201:**
```json
{
  "document_id": "uuid",
  "external_id": "string",
  "chunks_created": 15,
  "status": "processing",
  "estimated_completion": "ISO8601"
}
```

**Response 202 (Large Document):**
For documents >100KB, processing is async:
```json
{
  "document_id": "uuid",
  "status": "queued",
  "job_id": "uuid",
  "poll_url": "/api/v1/documents/uuid/status"
}
```

**Errors:**

| Status | Code | When |
|--------|------|------|
| 400 | INVALID_CONTENT | Content empty or >10MB |
| 400 | INVALID_LANGUAGE | Unsupported language code |
| 409 | DUPLICATE_DOCUMENT | content_hash already exists |
| 413 | PAYLOAD_TOO_LARGE | Request exceeds 10MB |
| 429 | RATE_LIMITED | Ingest rate limit exceeded |

---

### `GET /documents/:document_id`

Get document metadata and status.

**Response 200:**
```json
{
  "document_id": "uuid",
  "external_id": "string",
  "title": "string",
  "source_type": "string",
  "source_url": "string",
  "language": "en",
  "metadata": {},
  "status": "indexed",
  "chunks_count": 15,
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "embedding_model": "v1.2.3"
}
```

---

### `DELETE /documents/:document_id`

Remove a document and its chunks from the knowledge base.

**Response 204:** No content

**Errors:**

| Status | Code | When |
|--------|------|------|
| 404 | NOT_FOUND | Document doesn't exist |

---

### `GET /documents`

List documents with filtering and pagination.

**Query Parameters:**
- `page` (default: 1)
- `per_page` (default: 20, max: 100)
- `source_type` - Filter by source
- `created_after` - ISO8601 timestamp
- `created_before` - ISO8601 timestamp
- `search` - Full-text search in titles

**Response 200:**
```json
{
  "documents": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8
  }
}
```

---

## Admin Endpoints

### `GET /admin/metrics`

Get system metrics and health indicators.

**Query Parameters:**
- `period` - 1h|24h|7d|30d (default: 24h)
- `metrics` - Comma-separated list (optional)

**Response 200:**
```json
{
  "period": "24h",
  "retrieval": {
    "mrr": 0.72,
    "ndcg_10": 0.68,
    "recall_10": 0.85,
    "avg_latency_ms": 156,
    "p95_latency_ms": 312
  },
  "feedback": {
    "total_events": 15234,
    "thumbs_up_rate": 0.78,
    "thumbs_down_rate": 0.12,
    "avg_rating": 4.1,
    "click_through_rate": 0.45
  },
  "usage": {
    "total_queries": 45678,
    "unique_sessions": 12345,
    "queries_per_session": 3.7
  },
  "system": {
    "vector_store_latency_p95_ms": 45,
    "embedding_latency_p95_ms": 12,
    "generation_latency_p95_ms": 2100,
    "error_rate": 0.002
  },
  "learning": {
    "last_training": "ISO8601",
    "training_samples": 50000,
    "model_version": "v1.2.3",
    "improvement_vs_baseline": 0.08
  }
}
```

---

### `GET /admin/models`

List all model versions with their status.

**Response 200:**
```json
{
  "models": [
    {
      "version": "v1.2.3",
      "type": "embedding",
      "status": "active",
      "traffic_percent": 100,
      "created_at": "ISO8601",
      "metrics": {
        "mrr": 0.72,
        "ndcg_10": 0.68
      }
    },
    {
      "version": "v1.2.4",
      "type": "embedding",
      "status": "canary",
      "traffic_percent": 10,
      "created_at": "ISO8601",
      "metrics": {
        "mrr": 0.74,
        "ndcg_10": 0.71
      }
    }
  ]
}
```

---

### `POST /admin/models/:version/deploy`

Deploy a model version to production.

**Request:**
```json
{
  "strategy": "string (required) - immediate|canary|blue_green",
  "traffic_percent": "number (for canary, default: 10)",
  "auto_promote": "boolean (default: true) - Auto-promote if metrics improve",
  "rollback_threshold": "number (default: 0.95) - Rollback if < baseline * threshold"
}
```

**Response 200:**
```json
{
  "deployment_id": "uuid",
  "status": "deploying",
  "model_version": "v1.2.4",
  "strategy": "canary",
  "traffic_percent": 10,
  "started_at": "ISO8601"
}
```

---

### `POST /admin/models/:version/rollback`

Rollback to a previous model version.

**Request:**
```json
{
  "reason": "string (required) - Explanation for rollback",
  "target_version": "string (optional) - Specific version to rollback to"
}
```

**Response 200:**
```json
{
  "rollback_id": "uuid",
  "from_version": "v1.2.4",
  "to_version": "v1.2.3",
  "status": "rolling_back",
  "started_at": "ISO8601"
}
```

---

### `GET /admin/learning/status`

Get current learning pipeline status.

**Response 200:**
```json
{
  "status": "active",
  "current_job": {
    "job_id": "uuid",
    "type": "training",
    "started_at": "ISO8601",
    "progress": 0.65,
    "eta": "ISO8601"
  },
  "last_completed": {
    "job_id": "uuid",
    "completed_at": "ISO8601",
    "result": "success",
    "model_version": "v1.2.3"
  },
  "queue": {
    "pending_feedback_events": 15234,
    "training_pairs_available": 45678
  },
  "schedule": {
    "next_training": "ISO8601",
    "frequency": "daily"
  }
}
```

---

### `POST /admin/learning/pause`

Pause the learning pipeline.

**Response 200:**
```json
{
  "status": "paused",
  "paused_at": "ISO8601",
  "message": "Learning pipeline paused. Feedback collection continues."
}
```

---

### `POST /admin/learning/resume`

Resume the learning pipeline.

**Response 200:**
```json
{
  "status": "active",
  "resumed_at": "ISO8601",
  "message": "Learning pipeline resumed."
}
```

---

### `POST /admin/learning/trigger`

Manually trigger a training job.

**Request:**
```json
{
  "force": "boolean (default: false) - Run even if min samples not met",
  "config_override": {
    "learning_rate": 0.0001,
    "epochs": 10
  }
}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_start": "ISO8601"
}
```

---

### `GET /admin/experiments`

List A/B experiments.

**Response 200:**
```json
{
  "experiments": [
    {
      "id": "uuid",
      "name": "embedding-v1.2.4-test",
      "status": "running",
      "control": "v1.2.3",
      "treatment": "v1.2.4",
      "traffic_percent": 10,
      "started_at": "ISO8601",
      "metrics": {
        "control_mrr": 0.72,
        "treatment_mrr": 0.74,
        "p_value": 0.08,
        "samples": {
          "control": 5432,
          "treatment": 603
        }
      }
    }
  ]
}
```

---

### `POST /admin/experiments`

Create a new A/B experiment.

**Request:**
```json
{
  "name": "string (required)",
  "control_model": "string (required) - Version ID",
  "treatment_model": "string (required) - Version ID",
  "traffic_percent": "number (default: 10)",
  "primary_metric": "string (default: mrr)",
  "min_samples": "number (default: 1000)",
  "max_duration_hours": "number (default: 168)"
}
```

**Response 201:**
```json
{
  "experiment_id": "uuid",
  "status": "running",
  "started_at": "ISO8601"
}
```

---

### `POST /admin/experiments/:id/stop`

Stop an experiment early.

**Request:**
```json
{
  "winner": "string (optional) - control|treatment|none",
  "reason": "string (required)"
}
```

**Response 200:**
```json
{
  "experiment_id": "uuid",
  "status": "stopped",
  "winner": "treatment",
  "stopped_at": "ISO8601"
}
```

---

## Webhooks

### Webhook Configuration

Register webhooks via API or dashboard to receive events.

**Supported Events:**
- `model.deployed` - New model version deployed
- `model.rollback` - Model rolled back
- `experiment.completed` - A/B test finished
- `alert.triggered` - Metric alert fired
- `learning.completed` - Training job finished

**Webhook Payload:**
```json
{
  "event": "model.deployed",
  "timestamp": "ISO8601",
  "data": {
    "model_version": "v1.2.4",
    "strategy": "canary",
    "traffic_percent": 10
  },
  "signature": "sha256=..."
}
```

Verify webhook authenticity by computing HMAC-SHA256 of the raw body using your webhook secret.

---

## Versioning Strategy

- URL versioning: `/api/v1/`, `/api/v2/`
- Deprecation: 6-month notice before removal
- Breaking changes trigger new major version
- Non-breaking additions to existing versions allowed
- Sunset header indicates deprecation: `Sunset: Sat, 01 Jan 2026 00:00:00 GMT`

---

## SDKs

Official SDKs available:
- Python: `pip install rag-learning-client`
- TypeScript/JS: `npm install @rag-learning/client`
- Go: `go get github.com/org/rag-learning-go`

Example (Python):
```python
from rag_learning import Client

client = Client(api_key="your-api-key")

# Query
response = client.query("What is the refund policy?")
print(response.answer)

# Submit feedback
client.feedback(
    response_id=response.id,
    type="thumbs_up"
)
```

Example (TypeScript):
```typescript
import { RAGClient } from '@rag-learning/client';

const client = new RAGClient({ apiKey: 'your-api-key' });

// Query with streaming
const stream = await client.query({
  query: 'What is the refund policy?',
  options: { stream: true }
});

for await (const event of stream) {
  if (event.type === 'answer_chunk') {
    process.stdout.write(event.text);
  }
}
```
