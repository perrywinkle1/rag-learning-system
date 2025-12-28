# Closed-Loop RAG Learning System

A self-improving Retrieval-Augmented Generation system that gets smarter with every user interaction.

## Project Scope

### What This System Does

This system answers user questions by retrieving relevant documents from a knowledge base and generating responses. Unlike static RAG systems, it **continuously learns** from user behavior to improve retrieval quality over time.

### Core Value Proposition

Traditional RAG systems have a fixed retrieval model—they don't learn whether their results were actually helpful. This system closes that loop:

1. User asks a question
2. System retrieves documents and generates answer
3. System observes what user does (clicks, reads, copies, rates)
4. System uses that signal to improve future retrievals
5. Next similar query gets better results

### Boundaries

**In Scope:**
- Document ingestion and chunking
- Vector embedding and similarity search
- Re-ranking retrieved results
- LLM-based response generation with citations
- Implicit feedback collection (behavioral signals)
- Explicit feedback collection (ratings, thumbs)
- Contrastive learning on feedback pairs
- A/B testing for model validation
- Automated model deployment with rollback

**Out of Scope (v1):**
- User-to-user collaborative filtering
- Multi-modal retrieval (images, audio)
- Cross-lingual queries
- Deep personalization per user
- Federated learning
- Active learning with human labelers

## Spec Structure

```
specs/
├── CLAUDE.md                    # This file - project overview
├── rag-learning-system.md       # Feature spec (what + why)
├── technical/
│   └── architecture.md          # Technical spec (how)
├── data-model/
│   └── schema.md                # Database schema
├── api/
│   └── endpoints.md             # REST API contract
└── forlearningpurposes.md       # Educational deep-dive
```

## Key Concepts

| Concept | Definition |
|---------|------------|
| RAG | Retrieval-Augmented Generation - ground LLM responses in retrieved documents |
| Embedding | Dense vector representation of text for similarity search |
| Contrastive Learning | Training by comparing positive/negative examples |
| Re-ranking | Second-pass scoring to refine initial retrieval |
| Closed Loop | Feedback from outputs improves future inputs |
| Canary Deployment | Gradually rolling out changes to subset of traffic |

## Tech Stack (Recommended)

- **Vector Store**: Qdrant (or Pinecone, Weaviate)
- **Database**: PostgreSQL with pgvector
- **Message Queue**: Kafka (or Redis Streams)
- **Embedding Models**: sentence-transformers, OpenAI ada-002
- **LLM**: GPT-4, Claude, or open-source alternatives
- **ML Training**: PyTorch, Hugging Face Transformers
- **Orchestration**: Kubernetes, Temporal for workflows

## Implementation Priority

1. **Phase 1**: Basic RAG pipeline (query → retrieve → generate)
2. **Phase 2**: Feedback collection infrastructure
3. **Phase 3**: Learning pipeline (training + evaluation)
4. **Phase 4**: Closed loop (automated deployment + rollback)

## Working With These Specs

When implementing:
- Start with `rag-learning-system.md` for requirements
- Reference `technical/architecture.md` for component design
- Use `data-model/schema.md` for database setup
- Follow `api/endpoints.md` for API contracts

For newcomers:
- Read `forlearningpurposes.md` for detailed explanations of every concept

## Success Metrics

The system is working when:
- MRR@10 improves >15% after 10,000 interactions
- User satisfaction (thumbs up) exceeds 75%
- Query abandonment decreases >20%
- System maintains gains with <5% regression

## Claude Code Instructions

**Context Management:**
- Autocompact when context window hits 40% for main agents and subagents
- This ensures agents can complete complex tasks without running out of context
