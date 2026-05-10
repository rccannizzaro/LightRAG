"""W5: when adelete_by_doc_id takes the no-chunks early-return path,
the finally-block must NOT persist the graph or vector DBs (they weren't
touched). Only doc_status, full_docs, and optionally llm_response_cache
should be flushed.

Triggering scenario: bulk-deleting FAILED ingest docs (the user's
operational pain point). Without this guard, every doc in a 200-doc
batch rewrites the full ~1.5 GB of JSON state for zero benefit.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.base import DocStatus

pytestmark = pytest.mark.offline


def _make_no_chunks_rag_stub():
   """Build a mock RAG whose doc has empty chunks_list (FAILED ingest).

   Every storage exposes an AsyncMock `index_done_callback` so we can
   assert exactly which ones get awaited.
   """
   rag = MagicMock()
   rag.workspace = "/tmp/w5-test-workspace"

   rag.doc_status = AsyncMock()
   rag.doc_status.get_by_id = AsyncMock(
      return_value={
         "status": DocStatus.FAILED.value,
         "file_path": "/fake/path.txt",
         "chunks_list": [],  # the key signal: no chunks
      }
   )
   rag.doc_status.delete = AsyncMock()
   rag.doc_status.index_done_callback = AsyncMock(return_value=True)

   rag.full_docs = AsyncMock()
   rag.full_docs.delete = AsyncMock()
   rag.full_docs.index_done_callback = AsyncMock(return_value=True)

   # Every other storage MUST NOT be persisted in the no-chunks path.
   for name in (
      "chunk_entity_relation_graph",
      "entities_vdb",
      "relationships_vdb",
      "chunks_vdb",
      "text_chunks",
      "full_entities",
      "full_relations",
      "entity_chunks",
      "relation_chunks",
   ):
      mock = AsyncMock()
      mock.index_done_callback = AsyncMock(return_value=True)
      setattr(rag, name, mock)

   rag.llm_response_cache = AsyncMock()
   rag.llm_response_cache.index_done_callback = AsyncMock(return_value=True)
   rag.llm_response_cache.delete = AsyncMock()

   return rag


@pytest.mark.asyncio
async def test_no_chunks_path_skips_graph_and_vdb_persist():
   """The graph + VDBs must not be persisted when nothing touched them."""
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
      result = await LightRAG.adelete_by_doc_id(rag, "doc-no-chunks")

   assert result.status == "success"

   # Touched storages — must be flushed
   rag.doc_status.index_done_callback.assert_awaited()
   rag.full_docs.index_done_callback.assert_awaited()

   # Untouched storages — must NOT be flushed (this is the W5 guarantee)
   rag.chunk_entity_relation_graph.index_done_callback.assert_not_awaited()
   rag.entities_vdb.index_done_callback.assert_not_awaited()
   rag.relationships_vdb.index_done_callback.assert_not_awaited()
   rag.chunks_vdb.index_done_callback.assert_not_awaited()
   rag.text_chunks.index_done_callback.assert_not_awaited()
   rag.full_entities.index_done_callback.assert_not_awaited()
   rag.full_relations.index_done_callback.assert_not_awaited()
   rag.entity_chunks.index_done_callback.assert_not_awaited()
   rag.relation_chunks.index_done_callback.assert_not_awaited()

   # llm_response_cache: not requested for delete in this run, no flush
   rag.llm_response_cache.index_done_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_chunks_path_persists_llm_cache_when_requested():
   """When delete_llm_cache=True and metadata cache IDs exist, cleanup
   must persist the LLM cache too — but the graph/VDBs still stay put."""
   from lightrag.lightrag import LightRAG

   rag = _make_no_chunks_rag_stub()
   rag.doc_status.get_by_id = AsyncMock(
      return_value={
         "status": DocStatus.FAILED.value,
         "file_path": "/fake/path.txt",
         "chunks_list": [],
         "metadata": {"deletion_llm_cache_ids": ["cache-1", "cache-2"]},
      }
   )
   rag._get_existing_llm_cache_ids = AsyncMock(return_value=[])

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
         rag, "doc-no-chunks-with-cache", delete_llm_cache=True
      )

   assert result.status == "success"

   rag.doc_status.index_done_callback.assert_awaited()
   rag.full_docs.index_done_callback.assert_awaited()
   rag.llm_response_cache.index_done_callback.assert_awaited()

   # Graph + VDBs still untouched
   rag.chunk_entity_relation_graph.index_done_callback.assert_not_awaited()
   rag.entities_vdb.index_done_callback.assert_not_awaited()
   rag.relationships_vdb.index_done_callback.assert_not_awaited()
