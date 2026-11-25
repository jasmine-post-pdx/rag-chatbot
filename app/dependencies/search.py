"""FastAPI dependencies for retrieval-only search."""

from typing import Annotated

from fastapi import Depends, Request

from app.services.search import SearchService


def get_search_service(request: Request) -> SearchService:
    """Return the search service created during application lifespan.

    :param request: Current request (provides ``app.state``).
    :returns: The shared ``SearchService``.
    :raises RuntimeError: If lifespan did not install a search service."""
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise RuntimeError("Search service is not configured")
    return service


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
