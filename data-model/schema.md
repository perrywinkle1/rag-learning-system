# Data Model: Closed-Loop RAG Learning System

## Overview

This specification defines the data model for a self-improving RAG system. The model covers four domains: Documents (knowledge base), Interactions (queries and responses), Feedback (learning signals), and Models (versioned artifacts). The schema supports both operational queries and analytical learning pipelines.

## Entities

### Document Domain

#### documents
Primary storage for knowledge base content.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Unique document identifier |
| external_id | VARCHAR(255) | UNIQUE, INDEX | Source system reference |
| title | VARCHAR(500) | NOT NULL | Document title |
| content | TEXT | NOT NULL | Full document content |
| content_hash | CHAR(64) | INDEX | SHA-256 for deduplication |
| source_type | VARCHAR(50) | NOT NULL | Origin (upload, crawl, api) |
| source_url | VARCHAR(2048) | | Original URL if applicable |
| mime_type | VARCHAR(100) | | Content type |
| language | CHAR(2) | DEFAULT 'en' | ISO 639-1 code |
| metadata | JSONB | | Flexible attributes |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | Ingestion time |
| updated_at | TIMESTAMP | NOT NULL | Last modification |
| deleted_at | TIMESTAMP | | Soft delete marker |

#### document_chunks
Chunked segments for retrieval.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Unique chunk identifier |
| document_id | UUID | FK(documents), INDEX | Parent document |
| chunk_index | INT | NOT NULL | Order within document |
| content | TEXT | NOT NULL | Chunk text content |
| token_count | INT | NOT NULL | Token length |
| start_char | INT | | Character offset start |
| end_char | INT | | Character offset end |
| metadata | JSONB | | Chunk-specific metadata |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |

#### chunk_embeddings
Vector representations of chunks.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Unique embedding identifier |
| chunk_id | UUID | FK(document_chunks), UNIQUE | Source chunk |
| model_version | VARCHAR(50) | NOT NULL, INDEX | Embedding model version |
| embedding | VECTOR(384) | NOT NULL | Dense vector (dimension varies) |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |

### Interaction Domain

#### sessions
User interaction sessions.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Session identifier |
| user_id | UUID | INDEX | Anonymized user reference |
| client_type | VARCHAR(50) | | web, api, mobile |
| user_agent | VARCHAR(500) | | Client information |
| ip_hash | CHAR(64) | | Hashed IP for abuse detection |
| started_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |
| ended_at | TIMESTAMP | | Session close time |
| metadata | JSONB | | Session context |

#### queries
Individual search/question requests.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Query identifier |
| session_id | UUID | FK(sessions), INDEX | Parent session |
| query_text | TEXT | NOT NULL | Raw user query |
| query_normalized | TEXT | | Preprocessed query |
| query_hash | CHAR(64) | INDEX | For duplicate detection |
| query_embedding | VECTOR(384) | | Query vector |
| filters | JSONB | | Applied filters |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | |

#### responses
Generated responses to queries.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Response identifier |
| query_id | UUID | FK(queries), UNIQUE | Source query |
| response_text | TEXT | NOT NULL | Generated answer |
| confidence | FLOAT | CHECK(0..1) | Model confidence score |
| model_version | VARCHAR(50) | NOT NULL, INDEX | Generation model |
| retrieval_model_version | VARCHAR(50) | NOT NULL, INDEX | Embedding model used |
| reranker_model_version | VARCHAR(50) | | Reranker version |
| latency_ms | INT | | Total processing time |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |

#### response_sources
Documents cited in responses.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | |
| response_id | UUID | FK(responses), INDEX | Parent response |
| chunk_id | UUID | FK(document_chunks) | Cited chunk |
| rank | INT | NOT NULL | Position in results |
| retrieval_score | FLOAT | | Vector similarity score |
| rerank_score | FLOAT | | Re-ranker score |
| was_used | BOOLEAN | DEFAULT true | Included in generation |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |

### Feedback Domain

#### feedback_events
All feedback signals (implicit and explicit).

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Event identifier |
| response_id | UUID | FK(responses), INDEX | Target response |
| session_id | UUID | FK(sessions), INDEX | Source session |
| event_type | VARCHAR(50) | NOT NULL, INDEX | thumbs_up, click, dwell, etc. |
| event_value | FLOAT | | Numeric value (rating, time) |
| target_chunk_id | UUID | FK(document_chunks) | For source-specific feedback |
| reason_code | VARCHAR(50) | | Categorized negative feedback |
| reason_text | TEXT | | Free-form feedback |
| client_timestamp | TIMESTAMP | | Client-reported time |
| server_timestamp | TIMESTAMP | NOT NULL, DEFAULT NOW(), INDEX | Server receive time |
| processed | BOOLEAN | DEFAULT false, INDEX | Consumed by learning |
| metadata | JSONB | | Additional context |

#### feedback_aggregates
Rolled-up feedback metrics per response.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| response_id | UUID | PK, FK(responses) | |
| thumbs_up_count | INT | DEFAULT 0 | |
| thumbs_down_count | INT | DEFAULT 0 | |
| avg_rating | FLOAT | | Average star rating |
| total_dwell_ms | INT | DEFAULT 0 | Cumulative reading time |
| click_count | INT | DEFAULT 0 | Source link clicks |
| copy_count | INT | DEFAULT 0 | Response copy events |
| share_count | INT | DEFAULT 0 | |
| computed_score | FLOAT | | Weighted aggregate |
| updated_at | TIMESTAMP | NOT NULL | |

#### training_pairs
Curated pairs for model training.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | |
| query_id | UUID | FK(queries), INDEX | Source query |
| positive_chunk_id | UUID | FK(document_chunks) | Relevant document |
| negative_chunk_id | UUID | FK(document_chunks) | Non-relevant document |
| label_source | VARCHAR(50) | | feedback, click, manual |
| confidence | FLOAT | CHECK(0..1) | Label confidence |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |
| used_in_training | TIMESTAMP | | When used for training |

### Model Domain

#### model_versions
Registry of trained models.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | VARCHAR(50) | PK | Version identifier |
| model_type | VARCHAR(50) | NOT NULL | embedding, reranker, query_expansion |
| base_model | VARCHAR(100) | | Parent model if fine-tuned |
| artifact_path | VARCHAR(500) | NOT NULL | S3 path to weights |
| config | JSONB | NOT NULL | Training configuration |
| training_data_start | TIMESTAMP | | Training data range start |
| training_data_end | TIMESTAMP | | Training data range end |
| training_samples | INT | | Number of training examples |
| created_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |
| created_by | VARCHAR(100) | | Job ID or user |

#### model_evaluations
Evaluation metrics for each model.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | |
| model_version | VARCHAR(50) | FK(model_versions), INDEX | |
| eval_dataset | VARCHAR(100) | NOT NULL | Dataset name |
| mrr | FLOAT | | Mean Reciprocal Rank |
| ndcg_10 | FLOAT | | NDCG@10 |
| recall_10 | FLOAT | | Recall@10 |
| recall_20 | FLOAT | | Recall@20 |
| precision_10 | FLOAT | | Precision@10 |
| map | FLOAT | | Mean Average Precision |
| latency_p50_ms | FLOAT | | 50th percentile latency |
| latency_p95_ms | FLOAT | | 95th percentile latency |
| evaluated_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |

#### model_deployments
Deployment history and current state.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | |
| model_version | VARCHAR(50) | FK(model_versions) | |
| environment | VARCHAR(50) | NOT NULL | production, staging, canary |
| traffic_percent | INT | CHECK(0..100) | Traffic allocation |
| status | VARCHAR(50) | NOT NULL | deploying, active, draining, rolled_back |
| deployed_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |
| deactivated_at | TIMESTAMP | | When replaced/rolled back |
| deployed_by | VARCHAR(100) | | Actor (job or user) |
| rollback_reason | TEXT | | If rolled back, why |

#### ab_experiments
A/B test configurations and results.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | |
| name | VARCHAR(100) | NOT NULL | |
| control_model | VARCHAR(50) | FK(model_versions) | Baseline model |
| treatment_model | VARCHAR(50) | FK(model_versions) | Challenger model |
| traffic_percent | INT | CHECK(0..100) | Treatment allocation |
| metric_primary | VARCHAR(50) | | Primary success metric |
| metric_threshold | FLOAT | | Required improvement |
| min_samples | INT | | Minimum sample size |
| status | VARCHAR(50) | | running, completed, stopped |
| winner | VARCHAR(50) | | control, treatment, inconclusive |
| started_at | TIMESTAMP | NOT NULL, DEFAULT NOW() | |
| ended_at | TIMESTAMP | | |
| results | JSONB | | Statistical analysis |

## Relationships

```
documents 1:N document_chunks
document_chunks 1:N chunk_embeddings
document_chunks 1:N response_sources
document_chunks 1:N training_pairs (as positive or negative)

sessions 1:N queries
queries 1:1 responses
responses 1:N response_sources
responses 1:N feedback_events
responses 1:1 feedback_aggregates

queries 1:N training_pairs
model_versions 1:N model_evaluations
model_versions 1:N model_deployments
model_versions 1:N ab_experiments (as control or treatment)
```

## Indexes

### Primary Access Patterns

| Table | Index | Type | Rationale |
|-------|-------|------|-----------|
| documents | external_id | UNIQUE | External system lookup |
| documents | content_hash | BTREE | Deduplication check |
| documents | (deleted_at) WHERE deleted_at IS NULL | PARTIAL | Active documents only |
| document_chunks | document_id, chunk_index | COMPOSITE | Ordered chunk retrieval |
| chunk_embeddings | chunk_id | UNIQUE | Embedding lookup |
| chunk_embeddings | model_version | BTREE | Model-specific retrieval |
| sessions | user_id, started_at DESC | COMPOSITE | User history |
| queries | session_id, created_at DESC | COMPOSITE | Session timeline |
| queries | query_hash | BTREE | Duplicate detection |
| queries | created_at | BTREE | Time-range queries |
| responses | query_id | UNIQUE | Query-response join |
| responses | retrieval_model_version | BTREE | Model attribution |
| response_sources | response_id, rank | COMPOSITE | Ordered sources |
| feedback_events | response_id, event_type | COMPOSITE | Feedback aggregation |
| feedback_events | server_timestamp | BTREE | Time-range queries |
| feedback_events | (processed) WHERE processed = false | PARTIAL | Unprocessed events |
| training_pairs | query_id | BTREE | Query-based sampling |
| training_pairs | (used_in_training) WHERE used_in_training IS NULL | PARTIAL | Unused pairs |

### Vector Index (Qdrant)

```python
# Collection configuration
{
    "collection_name": "document_chunks",
    "vectors": {
        "size": 384,
        "distance": "Cosine"
    },
    "optimizers_config": {
        "indexing_threshold": 20000,
        "memmap_threshold": 50000
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000
    },
    "payload_schema": {
        "document_id": "keyword",
        "model_version": "keyword",
        "language": "keyword",
        "created_at": "datetime"
    }
}
```

## Migrations

### Migration Order
1. Create enum types
2. documents table
3. document_chunks table
4. chunk_embeddings table
5. sessions table
6. queries table
7. responses table
8. response_sources table
9. feedback_events table
10. feedback_aggregates table
11. training_pairs table
12. model_versions table
13. model_evaluations table
14. model_deployments table
15. ab_experiments table
16. Create indexes
17. Create triggers for feedback_aggregates

### Initial Migration

```sql
-- Migration: 001_initial_schema

-- Enum types
CREATE TYPE feedback_type AS ENUM (
    'thumbs_up', 'thumbs_down', 'rating',
    'click', 'dwell', 'copy', 'share', 'abandon'
);

CREATE TYPE model_type AS ENUM (
    'embedding', 'reranker', 'query_expansion', 'generator'
);

CREATE TYPE deployment_status AS ENUM (
    'deploying', 'active', 'draining', 'rolled_back'
);

CREATE TYPE experiment_status AS ENUM (
    'running', 'completed', 'stopped'
);

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_hash CHAR(64),
    source_type VARCHAR(50) NOT NULL,
    source_url VARCHAR(2048),
    mime_type VARCHAR(100),
    language CHAR(2) DEFAULT 'en',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP
);

CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_active ON documents(id) WHERE deleted_at IS NULL;

-- Document chunks
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    start_char INT,
    end_char INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON document_chunks(document_id, chunk_index);

-- Chunk embeddings (PostgreSQL with pgvector)
CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(chunk_id, model_version)
);

CREATE INDEX idx_embeddings_model ON chunk_embeddings(model_version);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    client_type VARCHAR(50),
    user_agent VARCHAR(500),
    ip_hash CHAR(64),
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_sessions_user ON sessions(user_id, started_at DESC);

-- Queries
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    query_text TEXT NOT NULL,
    query_normalized TEXT,
    query_hash CHAR(64),
    query_embedding VECTOR(384),
    filters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_queries_session ON queries(session_id, created_at DESC);
CREATE INDEX idx_queries_hash ON queries(query_hash);
CREATE INDEX idx_queries_time ON queries(created_at);

-- Responses
CREATE TABLE responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL UNIQUE REFERENCES queries(id),
    response_text TEXT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50) NOT NULL,
    retrieval_model_version VARCHAR(50) NOT NULL,
    reranker_model_version VARCHAR(50),
    latency_ms INT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_responses_retrieval_model ON responses(retrieval_model_version);

-- Response sources
CREATE TABLE response_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES document_chunks(id),
    rank INT NOT NULL,
    retrieval_score FLOAT,
    rerank_score FLOAT,
    was_used BOOLEAN DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sources_response ON response_sources(response_id, rank);

-- Feedback events
CREATE TABLE feedback_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES responses(id),
    session_id UUID REFERENCES sessions(id),
    event_type feedback_type NOT NULL,
    event_value FLOAT,
    target_chunk_id UUID REFERENCES document_chunks(id),
    reason_code VARCHAR(50),
    reason_text TEXT,
    client_timestamp TIMESTAMP,
    server_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    processed BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_feedback_response ON feedback_events(response_id, event_type);
CREATE INDEX idx_feedback_time ON feedback_events(server_timestamp);
CREATE INDEX idx_feedback_unprocessed ON feedback_events(id) WHERE processed = false;

-- Feedback aggregates
CREATE TABLE feedback_aggregates (
    response_id UUID PRIMARY KEY REFERENCES responses(id),
    thumbs_up_count INT DEFAULT 0,
    thumbs_down_count INT DEFAULT 0,
    avg_rating FLOAT,
    total_dwell_ms INT DEFAULT 0,
    click_count INT DEFAULT 0,
    copy_count INT DEFAULT 0,
    share_count INT DEFAULT 0,
    computed_score FLOAT,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Training pairs
CREATE TABLE training_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL REFERENCES queries(id),
    positive_chunk_id UUID NOT NULL REFERENCES document_chunks(id),
    negative_chunk_id UUID NOT NULL REFERENCES document_chunks(id),
    label_source VARCHAR(50),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    used_in_training TIMESTAMP
);

CREATE INDEX idx_training_query ON training_pairs(query_id);
CREATE INDEX idx_training_unused ON training_pairs(id) WHERE used_in_training IS NULL;

-- Model versions
CREATE TABLE model_versions (
    id VARCHAR(50) PRIMARY KEY,
    model_type model_type NOT NULL,
    base_model VARCHAR(100),
    artifact_path VARCHAR(500) NOT NULL,
    config JSONB NOT NULL,
    training_data_start TIMESTAMP,
    training_data_end TIMESTAMP,
    training_samples INT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- Model evaluations
CREATE TABLE model_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL REFERENCES model_versions(id),
    eval_dataset VARCHAR(100) NOT NULL,
    mrr FLOAT,
    ndcg_10 FLOAT,
    recall_10 FLOAT,
    recall_20 FLOAT,
    precision_10 FLOAT,
    map FLOAT,
    latency_p50_ms FLOAT,
    latency_p95_ms FLOAT,
    evaluated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evaluations_model ON model_evaluations(model_version);

-- Model deployments
CREATE TABLE model_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL REFERENCES model_versions(id),
    environment VARCHAR(50) NOT NULL,
    traffic_percent INT CHECK (traffic_percent >= 0 AND traffic_percent <= 100),
    status deployment_status NOT NULL,
    deployed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    deactivated_at TIMESTAMP,
    deployed_by VARCHAR(100),
    rollback_reason TEXT
);

CREATE INDEX idx_deployments_active ON model_deployments(environment, status)
    WHERE status = 'active';

-- A/B experiments
CREATE TABLE ab_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    control_model VARCHAR(50) NOT NULL REFERENCES model_versions(id),
    treatment_model VARCHAR(50) NOT NULL REFERENCES model_versions(id),
    traffic_percent INT CHECK (traffic_percent >= 0 AND traffic_percent <= 100),
    metric_primary VARCHAR(50),
    metric_threshold FLOAT,
    min_samples INT,
    status experiment_status DEFAULT 'running',
    winner VARCHAR(50),
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP,
    results JSONB
);

-- Trigger for feedback aggregation
CREATE OR REPLACE FUNCTION update_feedback_aggregate()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO feedback_aggregates (response_id, updated_at)
    VALUES (NEW.response_id, NOW())
    ON CONFLICT (response_id) DO UPDATE SET
        thumbs_up_count = feedback_aggregates.thumbs_up_count +
            CASE WHEN NEW.event_type = 'thumbs_up' THEN 1 ELSE 0 END,
        thumbs_down_count = feedback_aggregates.thumbs_down_count +
            CASE WHEN NEW.event_type = 'thumbs_down' THEN 1 ELSE 0 END,
        click_count = feedback_aggregates.click_count +
            CASE WHEN NEW.event_type = 'click' THEN 1 ELSE 0 END,
        copy_count = feedback_aggregates.copy_count +
            CASE WHEN NEW.event_type = 'copy' THEN 1 ELSE 0 END,
        share_count = feedback_aggregates.share_count +
            CASE WHEN NEW.event_type = 'share' THEN 1 ELSE 0 END,
        total_dwell_ms = feedback_aggregates.total_dwell_ms +
            CASE WHEN NEW.event_type = 'dwell' THEN COALESCE(NEW.event_value, 0)::INT ELSE 0 END,
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_feedback_aggregate
AFTER INSERT ON feedback_events
FOR EACH ROW EXECUTE FUNCTION update_feedback_aggregate();
```

## Queries

### Frequent Queries

```sql
-- Get response with sources for display (expected: <10ms)
SELECT
    r.id,
    r.response_text,
    r.confidence,
    r.latency_ms,
    json_agg(json_build_object(
        'chunk_id', rs.chunk_id,
        'rank', rs.rank,
        'score', rs.rerank_score,
        'content', dc.content,
        'doc_title', d.title
    ) ORDER BY rs.rank) as sources
FROM responses r
JOIN response_sources rs ON rs.response_id = r.id
JOIN document_chunks dc ON dc.id = rs.chunk_id
JOIN documents d ON d.id = dc.document_id
WHERE r.id = $1
GROUP BY r.id;

-- Get unprocessed feedback for learning (expected: <50ms)
SELECT
    fe.id,
    fe.response_id,
    fe.event_type,
    fe.event_value,
    fe.target_chunk_id,
    q.query_text,
    q.query_embedding
FROM feedback_events fe
JOIN responses r ON r.id = fe.response_id
JOIN queries q ON q.id = r.query_id
WHERE fe.processed = false
ORDER BY fe.server_timestamp
LIMIT 1000;

-- Generate training pairs from positive feedback (expected: <100ms)
SELECT
    q.id as query_id,
    q.query_text,
    rs_pos.chunk_id as positive_chunk_id,
    rs_neg.chunk_id as negative_chunk_id
FROM feedback_events fe
JOIN responses r ON r.id = fe.response_id
JOIN queries q ON q.id = r.query_id
JOIN response_sources rs_pos ON rs_pos.response_id = r.id AND rs_pos.chunk_id = fe.target_chunk_id
JOIN response_sources rs_neg ON rs_neg.response_id = r.id AND rs_neg.rank > rs_pos.rank
WHERE fe.event_type IN ('thumbs_up', 'click')
  AND fe.server_timestamp > NOW() - INTERVAL '7 days'
  AND rs_neg.chunk_id != rs_pos.chunk_id;

-- Model performance comparison (expected: <20ms)
SELECT
    mv.id as model_version,
    mv.model_type,
    me.mrr,
    me.ndcg_10,
    me.recall_10,
    me.latency_p50_ms,
    COUNT(DISTINCT r.id) as responses_served
FROM model_versions mv
LEFT JOIN model_evaluations me ON me.model_version = mv.id
LEFT JOIN responses r ON r.retrieval_model_version = mv.id
WHERE mv.model_type = 'embedding'
GROUP BY mv.id, me.mrr, me.ndcg_10, me.recall_10, me.latency_p50_ms
ORDER BY me.ndcg_10 DESC NULLS LAST;
```

### Write Patterns

| Pattern | Frequency | Consistency |
|---------|-----------|-------------|
| Insert feedback event | High (100/s) | Eventual (async) |
| Update feedback aggregate | High (100/s) | Eventual (trigger) |
| Insert query/response | Medium (50/s) | Strong |
| Insert training pairs | Low (batch) | Eventual |
| Update model deployment | Rare (<1/day) | Strong |
| Bulk chunk embeddings | Batch (daily) | Eventual |

### Partitioning Strategy

```sql
-- Partition feedback_events by month for retention management
CREATE TABLE feedback_events (
    -- ... columns ...
) PARTITION BY RANGE (server_timestamp);

CREATE TABLE feedback_events_2024_01 PARTITION OF feedback_events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- Continue for each month

-- Partition queries by month
CREATE TABLE queries (
    -- ... columns ...
) PARTITION BY RANGE (created_at);
```

## Data Retention

| Table | Retention | Archive Strategy |
|-------|-----------|------------------|
| documents | Indefinite | None |
| document_chunks | Indefinite | None |
| chunk_embeddings | Per model version | Delete on model deprecation |
| sessions | 90 days | Export to cold storage |
| queries | 90 days | Anonymize, export for analysis |
| responses | 90 days | Export with queries |
| feedback_events | 1 year | Export for retraining |
| training_pairs | Indefinite | Archive after training |
| model_versions | Indefinite | None (small) |
| model_evaluations | Indefinite | None (small) |

## Backup Strategy

```yaml
backup:
  full:
    schedule: "0 2 * * 0"  # Weekly Sunday 2 AM
    retention: 4 weeks
  incremental:
    schedule: "0 2 * * *"  # Daily 2 AM
    retention: 7 days
  wal_archiving:
    enabled: true
    retention: 7 days

  priority_tables:
    - documents
    - document_chunks
    - model_versions
    - training_pairs
```
