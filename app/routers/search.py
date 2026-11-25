"""Retrieval-only search routes."""

import asyncio

from fastapi import APIRouter, HTTPException, status

from app.dependencies.search import SearchServiceDep
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search import SearchTopKOutOfRangeError

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    body: SearchRequest,
    search_service: SearchServiceDep,
) -> SearchResponse:
    """Search indexed chunks without calling a generation model.

    :param body: Search query, optional filters, and optional top_k.
    :param search_service: Lifespan-scoped search service.
    :returns: Ranked hits with embedding/model provenance.
    :raises HTTPException: When ``top_k`` exceeds the configured maximum."""
    try:
        return await asyncio.to_thread(
            search_service.search,
            body.query,
            framework=body.framework,
            language=body.language,
            top_k=body.top_k,
        )
    except SearchTopKOutOfRangeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
