# Technical Spec: Closed-Loop RAG Learning System Architecture

## Summary

This document specifies the technical architecture for a self-improving RAG system. The system comprises five core subsystems: Ingestion, Retrieval, Generation, Feedback Collection, and Learning Pipeline. These subsystems work together to create a continuous improvement loop where user interactions drive retrieval quality enhancements.

## Architecture

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   Web App    │  │   API SDK    │  │ Admin Panel  │                      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                      │
└─────────┼─────────────────┼─────────────────┼──────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Rate Limiting │ Auth │ Request Routing │ Response Caching           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            SERVICE LAYER                                    │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Query Service  │  │ Feedback Service│  │  Admin Service  │             │
│  │                 │  │                 │  │                 │             │
│  │ • Parse query   │  │ • Collect signals│  │ • Metrics      │             │
│  │ • Orchestrate   │  │ • Validate      │  │ • Controls     │             │
│  │ • Format resp   │  │ • Queue events  │  │ • A/B config   │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            CORE LAYER                                       │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │    Retriever    │  │    Generator    │  │  Learning Engine│             │
│  │                 │  │                 │  │                 │             │
│  │ • Embed query   │  │ • Prompt build  │  │ • Feature eng   │             │
│  │ • Vector search │  │ • LLM inference │  │ • Model training│             │
│  │ • Re-rank       │  │ • Citation gen  │  │ • Evaluation    │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Vector Store │  │  Doc Store  │  │ Event Store │  │ Model Store │        │
│  │  (Qdrant)   │  │(PostgreSQL) │  │  (Kafka)    │  │    (S3)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Subsystem Details

#### 1. Ingestion Subsystem

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                           │
│                                                                   │
│  Documents ──▶ Chunker ──▶ Embedder ──▶ Indexer ──▶ Vector Store │
│                  │            │            │                      │
│                  ▼            ▼            ▼                      │
│              Metadata    Model Version  Index Config              │
│                  │            │            │                      │
│                  └────────────┴────────────┘                      │
│                              │                                    │
│                              ▼                                    │
│                         Doc Store                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Query Processing Pipeline

```
Query ──▶ Preprocessor ──▶ Embedder ──▶ Retriever ──▶ Re-ranker ──▶ Generator
              │                │            │            │             │
              ▼                ▼            ▼            ▼             ▼
          Expansion      Query Vector   Candidates   Top-K Docs   Response
          Synonyms                      (k=100)      (k=10)       + Citations
```

#### 3. Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEARNING PIPELINE                                │
│                                                                          │
│  Event Stream ──▶ Feature   ──▶ Training   ──▶ Validation ──▶ Deployment│
│                   Extraction     Job            (Holdout)      (Canary) │
│                      │            │               │              │       │
│                      ▼            ▼               ▼              ▼       │
│                  Feature       Model          Metrics       New Model   │
│                  Store        Checkpoint       Report       (if better) │
│                                                                          │
│  Rollback Trigger: validation_metric < baseline - threshold             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query Flow (Synchronous)**
   1. Client sends query to API Gateway
   2. Gateway authenticates, rate limits, routes to Query Service
   3. Query Service preprocesses (normalize, expand synonyms)
   4. Retriever embeds query using current embedding model
   5. Vector Store returns top-100 candidates by cosine similarity
   6. Re-ranker scores and selects top-10 documents
   7. Generator builds prompt with context and generates response
   8. Response returned with citations and metadata

2. **Feedback Flow (Asynchronous)**
   1. Client sends feedback event to Feedback Service
   2. Feedback Service validates and enriches with session context
   3. Event published to Kafka topic `feedback-events`
   4. Learning Engine consumes events in micro-batches
   5. Features extracted and stored in Feature Store

3. **Learning Flow (Batch + Online)**
   1. Scheduled job triggers training pipeline (daily)
   2. Feature Store provides training data (query, doc, label triples)
   3. Model trained on contrastive loss
   4. Validation against holdout set
   5. If metrics improve: staged rollout via A/B test
   6. If A/B winner: promote to primary model
   7. Previous model retained for rollback

## Interface Definition

### Public API

```typescript
interface RAGSystemAPI {
  // Core query endpoint
  query(request: QueryRequest): Promise<QueryResponse>;

  // Feedback collection
  submitFeedback(feedback: FeedbackEvent): Promise<void>;

  // Document management
  ingestDocument(doc: Document): Promise<IngestResult>;
  deleteDocument(docId: string): Promise<void>;

  // Admin operations
  getMetrics(timeRange: TimeRange): Promise<MetricsSnapshot>;
  getLearningStatus(): Promise<LearningStatus>;
  rollbackModel(version: string): Promise<void>;
  pauseLearning(): Promise<void>;
  resumeLearning(): Promise<void>;
}

interface QueryRequest {
  query: string;
  sessionId: string;
  topK?: number;           // default: 10
  filters?: FilterCriteria;
  includeScores?: boolean; // default: false
}

interface QueryResponse {
  responseId: string;
  answer: string;
  sources: Source[];
  confidence: number;
  latencyMs: number;
  modelVersion: string;
}

interface Source {
  docId: string;
  chunkId: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
}

interface FeedbackEvent {
  responseId: string;
  sessionId: string;
  type: 'thumbs_up' | 'thumbs_down' | 'rating' | 'click' | 'dwell' | 'copy';
  value?: number;          // for ratings: 1-5
  targetDocId?: string;    // for source-specific feedback
  reason?: string;         // for negative feedback
  timestamp: number;
}
```

### Internal Service Interfaces

```typescript
interface Retriever {
  embed(text: string): Promise<Float32Array>;
  search(embedding: Float32Array, topK: number): Promise<Candidate[]>;
  rerank(query: string, candidates: Candidate[]): Promise<RankedResult[]>;
}

interface LearningEngine {
  ingestFeedback(events: FeedbackEvent[]): Promise<void>;
  trainModel(config: TrainingConfig): Promise<TrainingResult>;
  evaluate(model: Model, testSet: TestSet): Promise<EvalMetrics>;
  deploy(model: Model, strategy: DeployStrategy): Promise<void>;
}
```

### Events Emitted

| Event | Payload | When |
|-------|---------|------|
| `query.completed` | `{responseId, queryHash, latencyMs, sourcesCount}` | After each query |
| `feedback.received` | `{responseId, type, value}` | After feedback submission |
| `model.training.started` | `{jobId, dataPoints, config}` | Training job begins |
| `model.training.completed` | `{jobId, metrics, duration}` | Training job ends |
| `model.deployed` | `{version, strategy, metrics}` | New model goes live |
| `model.rollback` | `{fromVersion, toVersion, reason}` | Rollback triggered |
| `alert.metric.degraded` | `{metric, value, threshold}` | Metric below threshold |

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Qdrant | 1.7+ | Vector similarity search |
| PostgreSQL | 15+ | Document metadata, user data |
| Kafka | 3.5+ | Event streaming for feedback |
| Redis | 7+ | Caching, rate limiting |
| S3/MinIO | - | Model artifact storage |
| sentence-transformers | 2.2+ | Base embedding models |
| PyTorch | 2.0+ | Model training |
| LangChain/LlamaIndex | latest | RAG orchestration |
| Prometheus | 2.45+ | Metrics collection |
| Grafana | 10+ | Dashboards |

## Component Specifications

### Embedding Service

```typescript
interface EmbeddingConfig {
  modelName: string;           // e.g., "all-MiniLM-L6-v2"
  dimensions: number;          // e.g., 384
  maxTokens: number;           // e.g., 512
  batchSize: number;           // e.g., 32
  poolingStrategy: 'mean' | 'cls' | 'max';
}

// Versioned model management
interface EmbeddingModel {
  version: string;
  config: EmbeddingConfig;
  checkpoint: string;          // S3 path
  createdAt: Date;
  metrics: {
    avgLatencyMs: number;
    throughputQps: number;
    validationScore: number;
  };
}
```

### Re-ranker Service

```typescript
interface RerankerConfig {
  modelType: 'cross-encoder' | 'listwise' | 'pairwise';
  modelName: string;
  maxCandidates: number;       // max docs to re-rank
  features: string[];          // ['semantic_sim', 'recency', 'doc_quality', 'user_affinity']
}

interface RerankerInput {
  query: string;
  candidates: Candidate[];
  userContext?: UserContext;
}

interface RerankerOutput {
  ranked: RankedCandidate[];
  scores: Record<string, number>;  // feature contributions
}
```

### Learning Engine

```typescript
interface TrainingConfig {
  objective: 'contrastive' | 'triplet' | 'listwise';
  learningRate: number;
  batchSize: number;
  epochs: number;
  warmupSteps: number;
  validationSplit: number;
  earlyStoppingPatience: number;
  hardNegativeMining: boolean;
  minFeedbackCount: number;    // min events before training
}

interface TrainingResult {
  jobId: string;
  modelVersion: string;
  trainingMetrics: {
    loss: number[];
    learningRate: number[];
  };
  validationMetrics: EvalMetrics;
  duration: number;
  dataPoints: number;
}

interface EvalMetrics {
  mrr: number;
  ndcg: number;
  recall: Record<number, number>;  // recall@k
  precision: Record<number, number>;
  map: number;
}
```

## Error Handling

| Error Condition | Response | Recovery |
|-----------------|----------|----------|
| Vector store unavailable | 503 + retry-after | Circuit breaker, fallback to cached results |
| Embedding service timeout | 504 | Retry with backoff, degrade to keyword search |
| LLM generation failure | 500 + partial response | Return sources without synthesis |
| Feedback ingestion failure | 202 (queued) | Async retry via dead letter queue |
| Training job failure | Alert + no deployment | Keep current model, investigate logs |
| Validation metrics degraded | Auto-rollback | Revert to previous model checkpoint |
| Vector index corruption | Alert + rebuild | Trigger full re-indexing from doc store |

### Circuit Breaker Configuration

```yaml
vector_store:
  failure_threshold: 5
  success_threshold: 3
  timeout_ms: 30000
  half_open_requests: 3

embedding_service:
  failure_threshold: 3
  success_threshold: 2
  timeout_ms: 5000
  half_open_requests: 1

llm_service:
  failure_threshold: 3
  success_threshold: 2
  timeout_ms: 30000
  half_open_requests: 1
```

## Performance Requirements

| Component | Metric | Target | Max Acceptable |
|-----------|--------|--------|----------------|
| Query embedding | Latency p50 | 10ms | 50ms |
| Vector search | Latency p50 | 20ms | 100ms |
| Re-ranking | Latency p50 | 30ms | 150ms |
| LLM generation | Latency p50 | 800ms | 3000ms |
| End-to-end query | Latency p95 | 2000ms | 5000ms |
| Feedback ingestion | Throughput | 1000/s | 500/s |
| Training pipeline | Duration | <2hr | <6hr |
| Model serving | Memory | <4GB | <8GB |

### Scaling Considerations

```yaml
embedding_service:
  replicas: 3
  cpu: 2
  memory: 4Gi
  gpu: optional (T4 for higher throughput)
  autoscaling:
    min: 2
    max: 10
    target_cpu: 70%

vector_store:
  shards: 3
  replicas_per_shard: 2
  memory_per_shard: 8Gi

reranker_service:
  replicas: 2
  cpu: 4
  memory: 8Gi
  gpu: recommended (for cross-encoder)
```

## Security Considerations

### Authentication
- API Gateway: JWT tokens with RSA-256 verification
- Inter-service: mTLS with service mesh
- Admin endpoints: Additional RBAC layer

### Authorization
| Role | Capabilities |
|------|--------------|
| user | query, submitFeedback |
| admin | user + metrics, config, rollback |
| service | internal operations only |

### Data Sensitivity
- Query logs: PII scrubbing before storage
- Feedback: Anonymized user IDs
- Embeddings: No direct PII recovery possible
- Model artifacts: Access-controlled S3 buckets

### Audit Trail
- All admin operations logged with actor, action, timestamp
- Model deployments tracked with full lineage
- Feedback data access requires approval

## Testing Strategy

### Unit Tests
- Embedding service: Vector dimension, normalization
- Re-ranker: Score ordering, feature computation
- Learning engine: Loss computation, gradient flow

### Integration Tests
- Query pipeline: End-to-end with mock data
- Feedback pipeline: Event flow to feature store
- Model deployment: Canary rollout mechanics

### Load Tests
- Query service: 1000 QPS sustained for 30min
- Feedback ingestion: 5000 events/sec burst
- Vector search: 10M vectors, 1000 QPS

### Evaluation Tests
- Retrieval quality on benchmark datasets (MS MARCO, BEIR)
- Regression tests against golden set
- A/B test statistical significance validation

## Migration / Rollout

### Phase 1: Foundation (Week 1-2)
- [ ] Deploy vector store cluster
- [ ] Deploy embedding service with base model
- [ ] Implement query pipeline without learning
- [ ] Set up monitoring and alerting

### Phase 2: Feedback Collection (Week 3-4)
- [ ] Deploy Kafka cluster
- [ ] Implement feedback service
- [ ] Instrument client for implicit signals
- [ ] Add explicit feedback UI

### Phase 3: Learning Pipeline (Week 5-6)
- [ ] Implement feature extraction
- [ ] Build training pipeline
- [ ] Set up model evaluation framework
- [ ] Implement A/B testing infrastructure

### Phase 4: Closed Loop (Week 7-8)
- [ ] Enable automated training jobs
- [ ] Implement canary deployment
- [ ] Add rollback automation
- [ ] Full system integration testing

### Rollback Plan
1. Feature flags disable learning pipeline instantly
2. Model rollback via single API call
3. Vector index can be rebuilt from doc store
4. Full system rollback: Kubernetes deployment rollback

## Configuration

```yaml
# config/rag-system.yaml

retrieval:
  embedding_model: "all-MiniLM-L6-v2"
  vector_store:
    host: "qdrant.internal"
    port: 6333
    collection: "documents"
  search:
    top_k_candidates: 100
    top_k_final: 10
    min_score: 0.5

reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  max_candidates: 50

generation:
  llm_provider: "openai"
  model: "gpt-4-turbo"
  max_tokens: 1024
  temperature: 0.3

learning:
  enabled: true
  min_feedback_for_training: 1000
  training_schedule: "0 2 * * *"  # Daily at 2 AM
  validation_threshold: 0.95      # vs baseline
  auto_rollback: true

ab_testing:
  enabled: true
  traffic_split: 0.1              # 10% to challenger
  min_samples: 500
  significance_level: 0.05

monitoring:
  metrics_retention_days: 90
  alert_channels: ["slack", "pagerduty"]
```
