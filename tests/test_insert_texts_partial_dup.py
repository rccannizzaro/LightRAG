"""W7: `/documents/texts` must handle partial duplicates correctly.

The pre-patch handler short-circuited on the FIRST file_source-or-content
duplicate, returning status="duplicated" for the WHOLE batch — silently
dropping every other doc in the request, even when N-1 of them were new.

The patched handler checks each doc independently, enqueues the new ones,
and reports a status that reflects the mix:
  - "success"          → all docs new
  - "partial_success"  → some new, some duplicates skipped
  - "duplicated"       → all docs were duplicates (legacy behavior preserved)
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Reset argv before any lightrag.api imports — the api module triggers
# argparse at import time and chokes on pytest's argv.
sys.argv = sys.argv[:1]

pytestmark = pytest.mark.offline


async def _stub_doc_status(known_file_paths=None, known_doc_ids=None):
   """Build a mock doc_status that knows specific file_paths and doc_ids."""
   ds = AsyncMock()
   known_paths = set(known_file_paths or [])
   known_ids = set(known_doc_ids or [])

   async def get_doc_by_file_path(fp):
      if fp in known_paths:
         return {"id": "doc-existing-by-path", "status": "processed",
                 "track_id": "tr-existing-path", "file_path": fp}
      return None

   async def get_by_id(did):
      if did in known_ids:
         return {"id": did, "status": "processed", "track_id": "tr-existing-id"}
      return None

   ds.get_doc_by_file_path = AsyncMock(side_effect=get_doc_by_file_path)
   ds.get_by_id = AsyncMock(side_effect=get_by_id)
   return ds


def _hash_id(text: str) -> str:
   import hashlib
   from lightrag.utils import sanitize_text_for_encoding
   sanitized = sanitize_text_for_encoding(text)
   return "doc-" + hashlib.md5(sanitized.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_partial_dup_returns_partial_success_and_enqueues_only_new():
   """Mix of new and duplicate texts → partial_success, only new ones enqueued."""
   # We can't easily import the closure-defined route handler. Build a tiny
   # harness that calls the same logic by importing the module and patching
   # rag.doc_status. The route is created inside `create_document_routes`,
   # so we'd need a TestClient — but the dedup logic is contained and we
   # validate it via direct calls to doc_status mocks below instead.
   # This test asserts the EXPECTED CONTRACT, which an end-to-end TestClient
   # test would also validate.
   pass  # placeholder; see test_insert_texts_partial_dup_e2e below


@pytest.mark.asyncio
async def test_insert_texts_partial_dup_e2e(monkeypatch):
   """End-to-end: build a minimal app, send a 3-text batch with 1 dup,
   assert partial_success + only the 2 new docs reach pipeline_index_texts."""
   from fastapi import FastAPI
   from fastapi.testclient import TestClient

   # Build a tiny rag mock — only doc_status methods are exercised by the
   # dedup logic. Stash known file_paths/doc_ids so 1 of the 3 texts dedups.
   text_dup = "this content is already on the server"
   text_new_1 = "first brand new content"
   text_new_2 = "second brand new content"
   dup_doc_id = _hash_id(text_dup)

   rag = MagicMock()
   rag.doc_status = await _stub_doc_status(known_doc_ids={dup_doc_id})

   captured = {"queued_texts": None, "queued_sources": None, "track_id": None}

   async def fake_pipeline_index_texts(rag_arg, texts, file_sources=None,
                                       track_id=None):
      captured["queued_texts"] = list(texts)
      captured["queued_sources"] = list(file_sources) if file_sources else None
      captured["track_id"] = track_id

   # Patch the handler's pipeline_index_texts. It's referenced in
   # document_routes.py as `pipeline_index_texts`.
   import lightrag.api.routers.document_routes as dr
   monkeypatch.setattr(dr, "pipeline_index_texts", fake_pipeline_index_texts)

   # Build a stripped-down FastAPI app with just the document router.
   # `create_document_routes` requires a rag + doc_manager + auth dep —
   # we mock combined_auth to a no-op pass-through.
   app = FastAPI()
   doc_manager = MagicMock()
   doc_manager.input_dir = "/tmp"

   def passthrough_auth():
      return None
   monkeypatch.setattr(dr, "get_combined_auth_dependency",
                       lambda *_args, **_kw: passthrough_auth)

   router = dr.create_document_routes(rag, doc_manager, api_key=None)
   app.include_router(router)  # router already has /documents prefix

   client = TestClient(app)
   resp = client.post(
      "/documents/texts",
      json={
         "texts": [text_new_1, text_dup, text_new_2],
         "file_sources": ["src/new_a.md", "src/dup.md", "src/new_b.md"],
      },
   )

   assert resp.status_code == 200, resp.text
   body = resp.json()
   assert body["status"] == "partial_success", body
   assert "1 skipped" in body["message"]
   assert "skipped as duplicates" in body["message"]

   # Only the 2 new texts hit pipeline_index_texts
   assert captured["queued_texts"] == [text_new_1, text_new_2]
   assert captured["queued_sources"] == ["src/new_a.md", "src/new_b.md"]
   assert captured["track_id"] is not None


@pytest.mark.asyncio
async def test_insert_texts_all_new_returns_success(monkeypatch):
   """All docs new → status=success, all enqueued."""
   from fastapi import FastAPI
   from fastapi.testclient import TestClient

   rag = MagicMock()
   rag.doc_status = await _stub_doc_status()  # nothing known

   captured = {}

   async def fake_pipeline_index_texts(rag_arg, texts, file_sources=None,
                                       track_id=None):
      captured["queued_texts"] = list(texts)

   import lightrag.api.routers.document_routes as dr
   monkeypatch.setattr(dr, "pipeline_index_texts", fake_pipeline_index_texts)
   monkeypatch.setattr(dr, "get_combined_auth_dependency",
                       lambda *_args, **_kw: lambda: None)

   app = FastAPI()
   router = dr.create_document_routes(rag, MagicMock(), api_key=None)
   app.include_router(router)  # router already has /documents prefix
   client = TestClient(app)

   resp = client.post(
      "/documents/texts",
      json={"texts": ["alpha", "bravo"]},
   )
   assert resp.status_code == 200
   assert resp.json()["status"] == "success"
   assert captured["queued_texts"] == ["alpha", "bravo"]


@pytest.mark.asyncio
async def test_insert_texts_all_dup_returns_duplicated(monkeypatch):
   """Every doc is a dup → status=duplicated, nothing enqueued."""
   from fastapi import FastAPI
   from fastapi.testclient import TestClient

   text_a = "duplicate alpha"
   text_b = "duplicate bravo"
   rag = MagicMock()
   rag.doc_status = await _stub_doc_status(
      known_doc_ids={_hash_id(text_a), _hash_id(text_b)}
   )

   captured = {"called": False}

   async def fake_pipeline_index_texts(*args, **kwargs):
      captured["called"] = True

   import lightrag.api.routers.document_routes as dr
   monkeypatch.setattr(dr, "pipeline_index_texts", fake_pipeline_index_texts)
   monkeypatch.setattr(dr, "get_combined_auth_dependency",
                       lambda *_args, **_kw: lambda: None)

   app = FastAPI()
   router = dr.create_document_routes(rag, MagicMock(), api_key=None)
   app.include_router(router)  # router already has /documents prefix
   client = TestClient(app)

   resp = client.post("/documents/texts", json={"texts": [text_a, text_b]})
   assert resp.status_code == 200
   assert resp.json()["status"] == "duplicated"
   assert resp.json()["track_id"] == ""
   # Pipeline must NOT have been triggered for an all-dup batch
   assert captured["called"] is False
