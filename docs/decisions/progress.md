# Learning Progress

This file is a running log of what I figure out while learning AI tech — especially retrieval-augmented generation, embeddings, evaluation, and the systems around them.

The point is not a polished essay. Each entry should capture what clicked, why it matters, and what is still fuzzy so I can see the trail later.

Newest entries go at the top.

---

## 2026-09-18 — Started a dedicated RAG lab

**What I figured out**
A real RAG system is a pipeline, not a single model call. Documents get ingested, chunked, embedded, stored, retrieved, ranked, and then passed to a generator. This repo is the place to learn those pieces by building and evaluating them.

**Why it matters**
Without a lab and a written trail, it is easy to try tools without remembering *why* a choice worked or failed. Logging the “aha” moments keeps the architecture decisions honest.

**Still fuzzy**
- How chunk size and overlap actually affect retrieval quality
- When keyword search beats (or should mix with) vector search
- What a useful eval set looks like for this project

**Next**
Stand up local Postgres with pgvector and get a first ingest → embed → retrieve loop running.

---

## YYYY-MM-DD — Short title of the insight

**What I figured out**
1–3 sentences. What clicked, in my own words.

**Why it matters**
How this changes what I build, measure, or decide next.

**Still fuzzy**
- Open questions
- Things I almost understand but cannot yet explain

**Next**
The smallest next experiment or reading that would make this clearer.
