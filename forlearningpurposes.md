# Understanding the Closed-Loop RAG Learning System

A comprehensive educational guide explaining every concept, decision, and component in this specification.

---

## Table of Contents

1. [What is RAG and Why Does It Matter?](#1-what-is-rag-and-why-does-it-matter)
2. [The Problem with Static RAG](#2-the-problem-with-static-rag)
3. [What "Closed-Loop Learning" Means](#3-what-closed-loop-learning-means)
4. [How Vector Search Works](#4-how-vector-search-works)
5. [The Query Pipeline Explained](#5-the-query-pipeline-explained)
6. [Understanding Feedback Signals](#6-understanding-feedback-signals)
7. [How the System Learns](#7-how-the-system-learns)
8. [Model Training Deep Dive](#8-model-training-deep-dive)
9. [A/B Testing and Safe Deployment](#9-ab-testing-and-safe-deployment)
10. [The Data Model Explained](#10-the-data-model-explained)
11. [API Design Principles](#11-api-design-principles)
12. [Metrics That Matter](#12-metrics-that-matter)
13. [Failure Modes and Recovery](#13-failure-modes-and-recovery)
14. [Scaling Considerations](#14-scaling-considerations)
15. [Privacy and Security](#15-privacy-and-security)

---

## 1. What is RAG and Why Does It Matter?

### The Problem RAG Solves

Large Language Models (LLMs) like GPT-4 or Claude have a fundamental limitation: their knowledge is frozen at training time. They can't know about:
- Recent events
- Your company's internal documents
- Private data they weren't trained on

RAG (Retrieval-Augmented Generation) solves this by **adding a retrieval step** before generation:

```
Traditional LLM:
  Question → LLM → Answer (from training data only)

RAG:
  Question → Retrieve Documents → LLM + Documents → Answer (grounded in your data)
```

### Why This Matters

1. **Accuracy**: Answers cite specific sources, reducing hallucination
2. **Freshness**: Update documents without retraining the LLM
3. **Control**: You decide what information the system can access
4. **Auditability**: You can see exactly which documents informed an answer

### Real-World Example

**Without RAG:**
> Q: "What's our refund policy?"
> A: "Typically, refund policies allow returns within 30 days..." (generic guess)

**With RAG:**
> Q: "What's our refund policy?"
> A: "According to your customer service handbook (Section 4.2), refunds are available within 14 days for unopened items..." (specific, cited)

---

## 2. The Problem with Static RAG

### What "Static" Means

Most RAG systems work like this:

1. Embed your documents once
2. When a query comes in, find similar documents
3. Generate response
4. Done

The embedding model and retrieval logic **never change**. They don't learn from:
- Which results users actually clicked
- Which answers got thumbs up vs. thumbs down
- What questions users had to rephrase

### The Consequences

**Poor results stay poor.** If the embedding model doesn't understand that "cancellation" and "refund" are related in your domain, it will keep missing relevant documents forever.

**No domain adaptation.** Generic embedding models trained on Wikipedia don't understand your company's jargon, acronyms, or concepts.

**Invisible failures.** You don't know which queries are failing because you're not tracking outcomes.

### The Closed-Loop Solution

This system tracks what happens after retrieval:
- Did the user click the sources?
- Did they read the full response?
- Did they give positive/negative feedback?
- Did they rephrase and try again?

Then it uses that signal to **improve the retrieval model** so future similar queries get better results.

---

## 3. What "Closed-Loop Learning" Means

### Open Loop vs. Closed Loop

**Open Loop (Static):**
```
Input → Process → Output
         ↓
    (no feedback)
```

**Closed Loop (Learning):**
```
Input → Process → Output
         ↑           ↓
         └── Learn ←─┘
```

In a closed loop, the output influences future processing. The system observes outcomes and adjusts its behavior.

### The Feedback Loop in This System

```
1. USER QUERY
   "How do I reset my password?"
        ↓
2. RETRIEVE DOCUMENTS
   System finds 10 potentially relevant chunks
        ↓
3. GENERATE RESPONSE
   LLM synthesizes answer from chunks
        ↓
4. USER INTERACTS
   - Clicks on source #2 (implicit: relevant)
   - Ignores source #5 (implicit: less relevant)
   - Gives thumbs up (explicit: good answer)
        ↓
5. SYSTEM LEARNS
   - Creates training pair: (query, clicked_doc, ignored_doc)
   - Updates model to rank clicked docs higher
        ↓
6. NEXT SIMILAR QUERY
   System retrieves better documents
```

### Why "Closed Loop" is Powerful

Each interaction makes the system smarter. With 10,000 users asking questions, you get 10,000 training signals. The system continuously adapts to:
- Your specific domain vocabulary
- User intent patterns
- Document quality signals

---

## 4. How Vector Search Works

### The Core Idea

Text is hard for computers to compare. "Dog" and "canine" look completely different as strings, but mean the same thing.

**Embeddings** solve this by converting text into numerical vectors where similar meanings are close together:

```
"dog"    → [0.2, 0.8, 0.1, 0.5, ...]
"canine" → [0.2, 0.7, 0.1, 0.6, ...]  ← very close!
"table"  → [0.9, 0.1, 0.3, 0.2, ...]  ← far away
```

### How Embeddings Are Created

Neural networks (like BERT, sentence-transformers) are trained on massive text datasets to produce embeddings where:
- Semantically similar texts have vectors pointing in similar directions
- Different meanings have vectors pointing in different directions

### Similarity Search

To find relevant documents:

1. **Embed the query** → get query vector
2. **Compare to all document vectors** → compute similarity scores
3. **Return top-k most similar** → ranked results

The similarity is typically **cosine similarity**:
```
similarity = (A · B) / (|A| × |B|)
```
This measures the angle between vectors (1.0 = identical direction, 0 = perpendicular, -1 = opposite).

### Why We Need Vector Databases

With millions of documents, comparing every vector is too slow. Vector databases like Qdrant use algorithms like **HNSW (Hierarchical Navigable Small Worlds)** to:
- Build an index structure over vectors
- Find approximate nearest neighbors in milliseconds
- Scale to billions of vectors

### The Limitation We're Fixing

Generic embedding models don't know your domain. By learning from user feedback, we fine-tune embeddings so:
- "PTO request" and "vacation days" become closer (if users treat them as synonyms)
- Irrelevant documents get pushed further away
- Domain-specific relationships are captured

---

## 5. The Query Pipeline Explained

### Step-by-Step Walkthrough

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                            │
│                                                                   │
│  "How do I request time off?"                                    │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │ PREPROCESS  │  • Normalize whitespace                         │
│  │             │  • Expand acronyms (PTO → Paid Time Off)        │
│  │             │  • Add synonyms from learned mappings            │
│  └──────┬──────┘                                                 │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │   EMBED     │  • Convert query to 384-dim vector              │
│  │             │  • Uses current embedding model version          │
│  └──────┬──────┘                                                 │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │  RETRIEVE   │  • Vector similarity search                      │
│  │             │  • Return top 100 candidates                     │
│  │             │  • Apply filters (date, source, etc.)           │
│  └──────┬──────┘                                                 │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │  RE-RANK    │  • Cross-encoder scores query-doc pairs         │
│  │             │  • Consider recency, doc quality, user context  │
│  │             │  • Return top 10 final results                  │
│  └──────┬──────┘                                                 │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │  GENERATE   │  • Build prompt with retrieved context          │
│  │             │  • LLM generates response with citations        │
│  │             │  • Add confidence score                         │
│  └──────┬──────┘                                                 │
│         │                                                         │
│         ▼                                                         │
│  Response + Sources + Metadata                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two-Stage Retrieval (Retrieve → Re-rank)?

**First stage (Vector Search):**
- Very fast (milliseconds)
- Can search millions of documents
- But: rough similarity, misses nuance

**Second stage (Re-ranking):**
- More accurate (cross-encoder sees query AND document together)
- But: slow (can only process ~100 docs)

By combining them:
1. Vector search narrows millions → 100 candidates (fast)
2. Re-ranker picks best 10 from 100 candidates (accurate)

### The Generation Step

The LLM receives a prompt like:

```
Based on the following documents, answer the user's question.
Cite your sources using [1], [2], etc.

Documents:
[1] HR Policy Manual, Section 5.2: "Employees may request PTO through
    the HR portal. Submit requests at least 2 weeks in advance..."
[2] Employee Handbook: "Vacation days accrue at 1.5 days per month..."

Question: How do I request time off?

Answer:
```

The LLM then generates a response grounded in these specific documents rather than its general training.

---

## 6. Understanding Feedback Signals

### Why Feedback Matters

Feedback tells us whether retrieval worked. Without it, we're blind:
- Did we find the right documents?
- Was the answer helpful?
- Should we rank things differently?

### Implicit vs. Explicit Feedback

**Implicit feedback** is inferred from behavior—users don't actively provide it:

| Signal | What It Means | Strength |
|--------|---------------|----------|
| Click on source | "This looks relevant" | Medium |
| Dwell time >30s | "I'm reading this carefully" | Medium |
| Copy response text | "This is useful enough to save" | High |
| Share/export | "This is valuable" | High |
| Abandon (<5s) | "This isn't what I wanted" | Negative |
| Query reformulation | "First results weren't good" | Negative |

**Explicit feedback** is directly provided by users:

| Signal | What It Means | Strength |
|--------|---------------|----------|
| Thumbs up | "Good answer" | High positive |
| Thumbs down | "Bad answer" | High negative |
| Star rating | Nuanced quality signal | Variable |
| "Wrong source" flag | Specific document is irrelevant | Targeted |
| "Missing info" flag | Gap in knowledge base | Diagnostic |

### Why Both Types?

**Implicit is abundant but noisy.** Every interaction generates signals, but they're ambiguous—did they click because it was relevant, or just curious?

**Explicit is clear but sparse.** Thumbs up/down is unambiguous, but most users won't bother.

By combining them, we get volume (implicit) with calibration (explicit).

### Weighting Signals

Not all signals are equal:

```python
SIGNAL_WEIGHTS = {
    'thumbs_up': 1.0,
    'thumbs_down': -1.0,
    'share': 0.8,
    'copy': 0.7,
    'click': 0.6,
    'dwell_30s': 0.5,
    'query_reformulation': -0.3,
    'abandon': -0.4,
}
```

These weights are tuned based on correlation with explicit feedback. If users who click also tend to give thumbs up, click is a good positive signal.

---

## 7. How the System Learns

### From Feedback to Training Data

Raw feedback events need to be converted into training examples. The key insight: **we can learn from comparisons**.

If a user:
1. Saw documents A, B, C, D, E
2. Clicked on document B
3. Ignored documents A, C, D, E

We infer:
- B is **more relevant** than A, C, D, E for this query
- We create training pairs: (query, B, A), (query, B, C), etc.

### Contrastive Learning

The training objective is **contrastive loss**:

```
For each (query, positive_doc, negative_doc):
    - Embed query, positive, negative
    - score_pos = similarity(query, positive)
    - score_neg = similarity(query, negative)
    - loss = max(0, margin - score_pos + score_neg)
```

This pushes the model to:
- Increase similarity between query and positive doc
- Decrease similarity between query and negative doc

Over thousands of examples, the embedding space reshapes to reflect **actual user preferences**.

### Hard Negative Mining

Not all negatives are equally useful. Easy negatives (completely unrelated documents) don't teach much.

**Hard negatives** are documents that:
- Were retrieved but not clicked
- Are superficially similar but actually irrelevant

Learning from hard negatives forces the model to make finer distinctions.

### Batch vs. Online Learning

**Batch learning** (what we use):
- Collect feedback over time (e.g., 24 hours)
- Train on accumulated data
- Deploy new model
- Pro: Stable, well-validated
- Con: Delayed learning

**Online learning** (alternative):
- Update model after each interaction
- Pro: Immediate adaptation
- Con: Unstable, hard to validate

We chose batch learning for stability—online learning can cause the model to oscillate or degrade unpredictably.

---

## 8. Model Training Deep Dive

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
│                                                                   │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                │
│  │  Extract  │───▶│   Train   │───▶│  Evaluate │                │
│  │  Features │    │   Model   │    │           │                │
│  └───────────┘    └───────────┘    └─────┬─────┘                │
│       ▲                                   │                       │
│       │                                   ▼                       │
│  Feedback DB              ┌───────────────────────────┐          │
│                           │  Metrics vs. Baseline     │          │
│                           │  • MRR improved?          │          │
│                           │  • NDCG improved?         │          │
│                           │  • No regression?         │          │
│                           └───────────┬───────────────┘          │
│                                       │                           │
│                          ┌────────────┴────────────┐             │
│                          ▼                         ▼              │
│                    [Better]                  [Worse]             │
│                    Deploy                    Discard             │
│                    (Canary)                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Extraction

Raw feedback becomes training features:

```python
# From a single feedback session
{
    "query_id": "abc123",
    "query_text": "password reset",
    "query_embedding": [0.1, 0.2, ...],

    "positive_chunks": ["chunk_456"],  # clicked
    "negative_chunks": ["chunk_789", "chunk_012"],  # shown but ignored

    "explicit_label": "thumbs_up",
    "confidence": 0.9
}
```

### Training Configuration

Key hyperparameters:

```yaml
training:
  objective: contrastive      # or triplet, listwise
  learning_rate: 0.00002      # small for fine-tuning
  batch_size: 32              # query-doc pairs per batch
  epochs: 3                   # passes through data
  warmup_steps: 100           # gradual LR increase
  hard_negative_ratio: 0.5    # 50% hard negatives
```

### Evaluation Metrics

Before deploying, we evaluate on a held-out test set:

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **MRR** (Mean Reciprocal Rank) | How high is the first relevant result? | 1/rank of first relevant |
| **NDCG@10** | Quality of top 10 results | Discounted cumulative gain |
| **Recall@10** | What fraction of relevant docs are in top 10? | relevant_in_10 / total_relevant |
| **MAP** (Mean Average Precision) | Overall ranking quality | Average precision across queries |

### Validation Gate

New models must pass:
```python
def should_deploy(new_metrics, baseline_metrics):
    return (
        new_metrics['mrr'] >= baseline_metrics['mrr'] * 0.95 and
        new_metrics['ndcg'] >= baseline_metrics['ndcg'] * 0.95 and
        new_metrics['recall_10'] >= baseline_metrics['recall_10']
    )
```

We allow up to 5% degradation on any single metric if others improve significantly.

---

## 9. A/B Testing and Safe Deployment

### Why A/B Testing?

Offline metrics (MRR, NDCG) don't always correlate with real-world performance. A model that scores well on test data might:
- Fail on new query patterns
- Have latency issues in production
- Degrade user experience in subtle ways

A/B testing validates improvements with **real users**.

### How A/B Tests Work

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  Control (90%)  │          │ Treatment (10%) │
    │  Model v1.2.3   │          │  Model v1.2.4   │
    └─────────────────┘          └─────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Collect Metrics │
                    │ Compare Results │
                    └─────────────────┘
```

### Statistical Significance

We don't just compare averages—we test if the difference is **statistically significant**:

```python
from scipy import stats

def is_significant(control_mrr, treatment_mrr, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(control_mrr, treatment_mrr)
    return p_value < alpha
```

With p < 0.05, we're 95% confident the improvement is real, not random chance.

### Sample Size Requirements

Detecting small improvements requires many samples:

| Effect Size | Required Samples |
|-------------|------------------|
| 10% improvement | ~400 per variant |
| 5% improvement | ~1,600 per variant |
| 2% improvement | ~10,000 per variant |

We require minimum 1,000 samples before drawing conclusions.

### Canary Deployment

Even after A/B validation, we deploy gradually:

1. **5% traffic** for 1 hour → check for errors
2. **25% traffic** for 6 hours → monitor metrics
3. **50% traffic** for 12 hours → validate at scale
4. **100% traffic** → full deployment

At any stage, automated rollback if:
- Error rate increases >2x
- Latency p95 increases >50%
- User feedback ratio drops below threshold

---

## 10. The Data Model Explained

### Why This Schema Design?

The schema supports three main use cases:
1. **Operational queries** (real-time serving)
2. **Analytics** (understanding system behavior)
3. **ML training** (generating training data)

### Core Entity Relationships

```
┌───────────────────────────────────────────────────────────────┐
│                    ENTITY RELATIONSHIPS                        │
│                                                                 │
│  documents ──1:N──▶ document_chunks ──1:N──▶ chunk_embeddings  │
│                           │                                     │
│                           │ (cited in)                          │
│                           ▼                                     │
│  sessions ──1:N──▶ queries ──1:1──▶ responses ──1:N──▶ response_sources
│      │                                    │                     │
│      │                                    │ (feedback on)       │
│      │                                    ▼                     │
│      └─────────────────────────────▶ feedback_events           │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### Key Tables Explained

**documents / document_chunks / chunk_embeddings**

Documents are split into chunks because:
- LLMs have context limits
- Smaller chunks are more precise
- Different chunks may be relevant to different queries

Embeddings are separate from chunks because:
- Multiple embedding versions may exist
- Re-embedding doesn't require re-chunking
- Easy to roll back embedding models

**sessions / queries / responses**

This hierarchy enables:
- Session context (what did user ask before?)
- Query-response pairing (for feedback attribution)
- Duplicate detection (same user, same question)

**feedback_events / feedback_aggregates**

Events are raw signals; aggregates are pre-computed summaries:
- Events: High-volume append-only log
- Aggregates: Low-latency read for API responses

**training_pairs**

Generated from feedback, consumed by training:
```sql
-- Example training pair
{
  query: "password reset",
  positive_chunk: "To reset your password, click...",  -- clicked
  negative_chunk: "Password requirements include...",  -- ignored
  confidence: 0.8
}
```

### Why PostgreSQL + Vector Store?

**PostgreSQL** for:
- Transactional data (users, sessions, feedback)
- Complex joins (reports, analytics)
- ACID guarantees

**Qdrant** for:
- Vector similarity search
- Optimized ANN (approximate nearest neighbor)
- Scales to billions of vectors

They complement each other—we query Qdrant for retrieval, PostgreSQL for everything else.

---

## 11. API Design Principles

### RESTful Design

Resources are nouns, actions are HTTP verbs:

| Operation | Endpoint | Method |
|-----------|----------|--------|
| Submit query | `/query` | POST |
| Get response | `/query/:id` | GET |
| Submit feedback | `/feedback` | POST |
| Add document | `/documents` | POST |
| Delete document | `/documents/:id` | DELETE |

### Request/Response Design

**Requests** are minimal—only required fields mandatory:
```json
{
  "query": "How do I reset my password?",  // required
  "top_k": 5,                               // optional, has default
  "filters": {}                             // optional
}
```

**Responses** are comprehensive—include everything client might need:
```json
{
  "response_id": "uuid",           // for feedback reference
  "answer": "...",                 // the actual answer
  "sources": [...],                // citations
  "confidence": 0.85,              // quality signal
  "model_info": {...},             // reproducibility
  "latency_ms": 1234               // observability
}
```

### Error Handling

Errors follow RFC 7807 (Problem Details):
```json
{
  "type": "https://api.example.com/errors/rate-limited",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "You have exceeded 100 queries per minute",
  "retry_after": 30
}
```

Standard codes:
- 400: Client error (bad input)
- 401: Not authenticated
- 403: Not authorized
- 404: Resource not found
- 429: Rate limited
- 500: Server error
- 503: Service unavailable

### Streaming Responses

For long generations, streaming provides better UX:

```
event: source
data: {"chunk_id": "uuid", "content": "..."}

event: answer_chunk
data: {"text": "Based on "}

event: answer_chunk
data: {"text": "the documents..."}
```

Benefits:
- User sees results immediately
- Perceived latency is lower
- Can abort mid-generation if user navigates away

### Idempotency

Feedback submissions include `client_timestamp` for deduplication:
```json
{
  "response_id": "abc",
  "type": "thumbs_up",
  "client_timestamp": "2024-01-15T10:30:00Z"
}
```

If the same feedback is submitted twice (network retry), we ignore the duplicate.

---

## 12. Metrics That Matter

### Retrieval Quality Metrics

**MRR (Mean Reciprocal Rank)**
- Question: "How quickly do we find the right document?"
- Calculation: Average of 1/rank for first relevant result
- Target: >0.65 (first relevant result usually in top 2)

**NDCG@10 (Normalized Discounted Cumulative Gain)**
- Question: "How good is the overall ranking in top 10?"
- Accounts for position (higher is better) and relevance grades
- Target: >0.70

**Recall@k**
- Question: "What fraction of relevant docs are in top k?"
- Critical for not missing important information
- Target: Recall@20 > 0.85

### User Satisfaction Metrics

**Thumbs Up Ratio**
```
thumbs_up_ratio = thumbs_up / (thumbs_up + thumbs_down)
```
Target: >0.75 (3 out of 4 ratings positive)

**Click-Through Rate**
```
ctr = queries_with_source_clicks / total_queries
```
Target: >0.40 (users engaging with sources)

**Query Abandonment Rate**
```
abandonment = queries_with_no_interaction / total_queries
```
Target: <0.20 (users finding value)

### System Health Metrics

**Latency Percentiles**
- p50: Typical experience (target: <500ms end-to-end)
- p95: Worst 5% (target: <2000ms)
- p99: Outliers (target: <5000ms)

**Error Rate**
```
error_rate = failed_requests / total_requests
```
Target: <0.1% (99.9% success rate)

**Learning Pipeline Lag**
- Time from feedback event to model training
- Target: <1 hour

### How Metrics Connect

```
User asks question
        │
        ▼
[Retrieval Quality]     ◄── MRR, NDCG, Recall
Good documents found?
        │
        ▼
[Generation Quality]
Good answer synthesized?
        │
        ▼
[User Satisfaction]     ◄── Thumbs ratio, CTR, Abandonment
User found it helpful?
        │
        ▼
[Learning Signal]
Feedback collected?
        │
        ▼
[Model Improvement]     ◄── MRR delta over time
Retrieval getting better?
        │
        └──────────────────────▶ (loop back to retrieval)
```

---

## 13. Failure Modes and Recovery

### What Can Go Wrong

| Component | Failure Mode | Impact | Detection |
|-----------|--------------|--------|-----------|
| Vector store | Unavailable | No retrieval | Health check, circuit breaker |
| Embedding service | Timeout | Slow/no queries | Latency spike |
| LLM | Rate limited | No generation | 429 responses |
| Kafka | Backlog | Learning delayed | Consumer lag metric |
| Training job | Crashes | No model updates | Job status monitoring |
| New model | Degraded quality | Worse results | Metric regression |

### Circuit Breakers

When a dependency fails repeatedly, stop calling it:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=30):
        self.failures = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func):
        if self.state == 'OPEN':
            raise CircuitOpenError()
        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

States:
- **CLOSED**: Normal operation
- **OPEN**: Failing, reject all calls
- **HALF_OPEN**: Testing if service recovered

### Graceful Degradation

When components fail, degrade gracefully:

| Failure | Degradation |
|---------|-------------|
| Re-ranker down | Return vector search results directly |
| LLM down | Return sources without synthesis |
| Feedback ingestion down | Queue locally, replay later |
| Learning pipeline down | Keep serving current model |

### Rollback Procedures

**Automated rollback** triggers when:
- Validation metrics < baseline * 0.95
- A/B test shows treatment is worse
- Error rate spikes after deployment

**Manual rollback** via API:
```bash
curl -X POST /admin/models/v1.2.4/rollback \
  -d '{"reason": "User complaints about relevance"}'
```

Rollback completes in <5 minutes by:
1. Routing traffic to previous model
2. Marking current model as "rolled back"
3. Alerting team for investigation

---

## 14. Scaling Considerations

### Read vs. Write Patterns

| Operation | Frequency | Latency Requirement |
|-----------|-----------|---------------------|
| Query (read) | 100/s | <200ms retrieval |
| Feedback (write) | 500/s | <50ms (async) |
| Ingest (write) | 10/s | <5s (can be async) |
| Training (batch) | 1/day | Hours acceptable |

### Scaling Each Component

**Embedding Service**
- Stateless → horizontal scaling
- GPU optional (faster) but CPU works
- Cache embeddings for repeated queries

**Vector Store (Qdrant)**
- Shard by collection for parallelism
- Replicate for read throughput
- Use on-disk storage for large collections

**PostgreSQL**
- Read replicas for analytics
- Connection pooling (PgBouncer)
- Partition large tables by time

**Feedback Pipeline (Kafka)**
- Partition by response_id for ordering
- Multiple consumers for throughput
- Exactly-once semantics with idempotent producers

### Cost Optimization

| Component | Cost Driver | Optimization |
|-----------|-------------|--------------|
| Vector store | Memory | Use on-disk for cold data |
| Embeddings | GPU time | Batch requests, cache results |
| LLM | Tokens | Shorter prompts, caching |
| Storage | Volume | Retention policies, compression |

### Capacity Planning

Rule of thumb for sizing:
```
Vector store memory = num_vectors × dimensions × 4 bytes × 1.5 (overhead)
Example: 10M vectors × 384 dims × 4 × 1.5 = ~23 GB
```

Query throughput:
```
QPS = num_replicas × single_node_qps
Example: 3 replicas × 100 QPS = 300 QPS
```

---

## 15. Privacy and Security

### Data Sensitivity Levels

| Data Type | Sensitivity | Handling |
|-----------|-------------|----------|
| Query text | Medium | Anonymize user ID, retain 90 days |
| Response text | Low | Generated content, retain with query |
| Feedback | Low | Anonymized signals |
| User ID | High | Hash before storage, never log raw |
| Document content | Variable | Depends on source (may be PII) |
| Embeddings | Low | No direct PII recovery possible |

### Anonymization

User identifiers are hashed:
```python
import hashlib

def anonymize_user(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
```

This allows:
- Session continuity (same hash per user)
- No reverse lookup (can't get user_id from hash)
- Compliance with privacy regulations

### Data Retention

```yaml
retention_policy:
  queries: 90 days      # then delete
  feedback: 1 year      # needed for training
  sessions: 90 days     # then delete
  documents: indefinite # business data
  model_artifacts: indefinite  # reproducibility
```

### Access Control

| Role | Can Access |
|------|------------|
| API user | Own queries, submit feedback |
| Admin | All metrics, model management |
| Data scientist | Training data (anonymized) |
| System | Everything (service accounts) |

### Audit Logging

All sensitive operations are logged:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "actor": "admin@example.com",
  "action": "model.deploy",
  "resource": "model/v1.2.4",
  "details": {"strategy": "canary", "traffic": 10}
}
```

Audit logs are:
- Immutable (append-only)
- Retained for 7 years (compliance)
- Encrypted at rest

### Security Best Practices

1. **API Authentication**: JWT with short expiry (1 hour)
2. **Rate Limiting**: Per-user and per-IP
3. **Input Validation**: Sanitize queries to prevent injection
4. **Encryption**: TLS 1.3 in transit, AES-256 at rest
5. **Network Isolation**: Internal services not internet-accessible
6. **Secret Management**: Vault for API keys, no secrets in code

---

## Summary: The Complete Picture

This system transforms a static RAG pipeline into a learning system:

1. **Query** → User asks a question
2. **Retrieve** → Vector search finds relevant documents
3. **Re-rank** → Neural model improves ranking
4. **Generate** → LLM synthesizes answer with citations
5. **Collect** → Track what user does with the answer
6. **Learn** → Train better retrieval from feedback
7. **Deploy** → Safely roll out improvements
8. **Repeat** → Each cycle makes the system smarter

The key insight: **user behavior is training data**. Every click, every thumbs up, every reformulated query teaches the system what "relevant" means in your specific domain.

Over time, this creates a flywheel:
- Better retrieval → Better answers → More usage → More feedback → Better retrieval

The specs in this repository provide the blueprint. Implementation requires careful attention to the feedback loop, validation gates, and safe deployment practices that make this work in production.
