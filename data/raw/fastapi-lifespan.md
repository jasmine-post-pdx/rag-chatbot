---
title: FastAPI Lifespan
framework: fastapi
language: python
---

FastAPI lifespan handlers create shared clients on startup and close them on
shutdown. Close OpenAI clients during FastAPI shutdown. Startup and shutdown
hooks keep shared FastAPI resources available for path operations without
leaking connections.
