"""W6: when adelete_by_doc_id is called with skip_persist=True, the per-doc
`_insert_done()` flush is suppressed — the caller is expected to flush at
the end of the batch.

This composes with skip_rebuild=True (W2). Together they amortize the two
per-doc costs: KG rebuild (LLM-driven) and storage flush (file-system I/O).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.base import DocStatus

pytestmark = pytest.mark.offline


def _make_no_chunks_rag_stub():
   """Doc with empty chunks_list (FAILED ingest cleanup case)."""
   rag = MagicMock()
   rag.workspace = "/tmp/w6-test-workspace"
   rag.doc_status = AsyncMock()
   rag.doc_status.get_by_id = AsyncMock(
      return_value={
         "status": DocStatus.FAILED.value,
         "file_path": "/fake/path.txt",
         "chunks_list": [],
      }
   )
   rag.doc_status.delete = AsyncMock()
   rag.doc_status.index_done_callback = AsyncMock(return_value=True)
   rag.full_docs = AsyncMock()
   rag.full_docs.delete = AsyncMock()
   rag.full_docs.index_done_callback = AsyncMock(return_value=True)
   for name in (
      "chunk_entity_relation_graph", "entities_vdb", "relationships_vdb",
      "chunks_vdb", "text_chunks", "full_entities", "full_relations",
      "entity_chunks", "relation_chunks",
   ):
      m = AsyncMock()
      m.index_done_callback = AsyncMock(return_value=True)
      setattr(rag, name, m)
   rag.llm_response_cache = AsyncMock()
   rag.llm_response_cache.index_done_callback = AsyncMock(return_value=True)
   rag.llm_response_cache.delete = AsyncMock()
   rag._insert_done = AsyncMock()
   return rag


@pytest.mark.asyncio
async def test_skip_persist_suppresses_per_doc_flush():
   """skip_persist=True must skip both _insert_done() AND the W5
   single-storage flush, regardless of whether the doc had chunks."""
   from lightrag.lightrag import LightRAG

   rag = _make_no_chunks_rag_stub()

   pipeline_status = {"busy": False, "history_messages": []}
   lock = asyncio.Lock()

   with (
      patch(
         "lightrag.lightrag.get_namespace_data",
         new_callable=AsyncMock,
         return_value=pipeline_status,
      ),
      patch("lightrag.lightrag.get_namespace_lock", return_value=lock),
   ):
      result = await LightRAG.adelete_by_doc_id(
         rag, "doc-no-chunks", skip_persist=True
      )

   assert result.status == "success"

   # The deletion still happened (doc_status + full_docs were deleted) ...
   rag.doc_status.delete.assert_awaited()
   rag.full_docs.delete.assert_awaited()

   # ... but NO storage was flushed (skip_persist deferred everything).
   rag._insert_done.assert_not_awaited()
   rag.doc_status.index_done_callback.assert_not_awaited()
   rag.full_docs.index_done_callback.assert_not_awaited()
   rag.chunk_entity_relation_graph.index_done_callback.assert_not_awaited()
   rag.entities_vdb.index_done_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_persist_default_false_preserves_per_doc_flush():
   """Without skip_persist, the W5 / chunk-bearing path still flushes."""
   from lightrag.lightrag import LightRAG

   rag = _make_no_chunks_rag_stub()

   pipeline_status = {"busy": False, "history_messages": []}
   lock = asyncio.Lock()

   with (
      patch(
         "lightrag.lightrag.get_namespace_data",
         new_callable=AsyncMock,
         return_value=pipeline_status,
      ),
      patch("lightrag.lightrag.get_namespace_lock", return_value=lock),
   ):
      # Default: no skip_persist
      result = await LightRAG.adelete_by_doc_id(rag, "doc-no-chunks")

   assert result.status == "success"

   # W5 path: doc_status + full_docs were flushed (no chunks → graph/VDBs skipped)
   rag.doc_status.index_done_callback.assert_awaited()
   rag.full_docs.index_done_callback.assert_awaited()
   rag.chunk_entity_relation_graph.index_done_callback.assert_not_awaited()
