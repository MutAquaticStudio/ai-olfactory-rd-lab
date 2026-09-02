"""Small application service around the local academic RAG CLI functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .ingest import PipelineConfig, ingest_academic_literature
from .query import query_academic_knowledge


@dataclass(frozen=True)
class AcademicRAGService:
    """Context retrieval service; it has no training/model dependencies."""

    config: PipelineConfig = PipelineConfig()

    def ingest(self, *, rebuild: bool = False) -> Dict[str, Any]:
        return ingest_academic_literature(self.config, rebuild=rebuild)

    def query(self, query_string: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        return query_academic_knowledge(
            query_string,
            top_k=top_k,
            index_dir=self.config.index_dir,
            device=self.config.device,
        )


__all__ = ["AcademicRAGService"]
