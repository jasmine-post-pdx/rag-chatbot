---
title: RAG Context Budget
language: python
---

Context budgeting reserves tokens for instructions, the question, tool schemas,
output reserve, and a safety margin before selecting ranked chunks. Select ranked
chunks without exceeding budget. Do not silently send oversized prompts. Context
window allocation for retrieval happens before generation in a RAG service.
Tool schemas counted in the token budget reduce available retrieval capacity.
