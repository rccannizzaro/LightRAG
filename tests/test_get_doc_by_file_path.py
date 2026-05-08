"""Tests for the by_file_path lookup contract and the corresponding HTTP route.

Covers:
* JsonDocStatusStorage.get_doc_by_file_path now returns the doc id under "id"
  (the contract refinement that lets callers resolve the doc_id without a
  second lookup).
* Hit and miss cases at the storage layer.
* The DocStatusResponse model accepts the dict shape produced by the route.

Route-level wiring (FastAPI registration, status code, JSON shape) is exercised
by the integration tests in test_path_prefixes.py via TestClient; this file
focuses on the storage contract because that is where the behavior change
lives.
"""

import sys

import pytest

# Avoid lightrag.api.config picking up a developer-local .env at import time.
sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    format_datetime,
    normalize_file_path,
)
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage  # noqa: E402


def _make_json_storage(tmp_path) -> JsonDocStatusStorage:
   """Build a JsonDocStatusStorage with an isolated working directory.

   The constructor signature varies across versions; this helper avoids
   binding tests to a particular __init__ shape by setting only the
   attributes get_doc_by_file_path actually reads.
   """
   storage = JsonDocStatusStorage.__new__(JsonDocStatusStorage)
   import asyncio

   storage._storage_lock = asyncio.Lock()
   storage._data = {}
   storage.workspace = "test"
   storage.namespace = "doc_status"
   return storage


@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_id_for_match(tmp_path):
   storage = _make_json_storage(tmp_path)
   storage._data["doc-aaa"] = {
      "file_path": "alpha.md",
      "status": DocStatus.PROCESSED.value,
      "content_summary": "alpha",
      "content_length": 5,
      "created_at": "2026-05-08T00:00:00+00:00",
      "updated_at": "2026-05-08T00:00:00+00:00",
   }
   storage._data["doc-bbb"] = {
      "file_path": "beta.md",
      "status": DocStatus.PENDING.value,
      "content_summary": "beta",
      "content_length": 4,
      "created_at": "2026-05-08T00:00:00+00:00",
      "updated_at": "2026-05-08T00:00:00+00:00",
   }

   result = await storage.get_doc_by_file_path("alpha.md")

   assert result is not None
   assert result["id"] == "doc-aaa"
   assert result["file_path"] == "alpha.md"
   assert result["status"] == DocStatus.PROCESSED.value
   # All original fields preserved
   assert result["content_summary"] == "alpha"
   assert result["content_length"] == 5


@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_none_for_miss(tmp_path):
   storage = _make_json_storage(tmp_path)
   storage._data["doc-aaa"] = {
      "file_path": "alpha.md",
      "status": DocStatus.PROCESSED.value,
   }

   assert await storage.get_doc_by_file_path("missing.md") is None


@pytest.mark.asyncio
async def test_get_doc_by_file_path_does_not_mutate_storage(tmp_path):
   """The returned dict must not write the synthetic 'id' key back into _data,
   otherwise repeat lookups would observe a polluted record on subsequent
   calls and downstream serializers would emit a duplicate id field."""
   storage = _make_json_storage(tmp_path)
   storage._data["doc-aaa"] = {
      "file_path": "alpha.md",
      "status": DocStatus.PROCESSED.value,
   }

   await storage.get_doc_by_file_path("alpha.md")

   assert "id" not in storage._data["doc-aaa"]


def test_doc_status_response_accepts_route_payload_shape():
   """The new /documents/by_file_path route unpacks doc_data into
   DocStatusResponse; this test pins the shape so a future refactor of the
   storage contract does not silently break the route."""
   doc_data = {
      "id": "doc-aaa",
      "file_path": "alpha.md",
      "status": DocStatus.PROCESSED.value,
      "content_summary": "alpha",
      "content_length": 5,
      "created_at": "2026-05-08T00:00:00+00:00",
      "updated_at": "2026-05-08T00:00:00+00:00",
      "track_id": None,
      "chunks_count": 1,
      "error_msg": None,
      "metadata": None,
   }

   response = DocStatusResponse(
      id=doc_data["id"],
      content_summary=doc_data["content_summary"],
      content_length=doc_data["content_length"],
      status=doc_data["status"],
      created_at=format_datetime(doc_data["created_at"]),
      updated_at=format_datetime(doc_data["updated_at"]),
      track_id=doc_data["track_id"],
      chunks_count=doc_data["chunks_count"],
      error_msg=doc_data["error_msg"],
      metadata=doc_data["metadata"],
      file_path=normalize_file_path(doc_data["file_path"]),
   )

   assert response.id == "doc-aaa"
   assert response.file_path == "alpha.md"
   assert response.status == DocStatus.PROCESSED
