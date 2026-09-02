"""Ingestion façade for the tested academic_rag_pipeline CLI implementation."""

from academic_rag_pipeline import (  # noqa: F401
    ChunkPayload,
    DownloadError,
    ExtractionError,
    IndexCompatibilityError,
    PaperContent,
    PipelineConfig,
    PipelineError,
    ingest_academic_literature,
    process_paper,
)

__all__ = [
    "PipelineConfig",
    "PipelineError",
    "DownloadError",
    "ExtractionError",
    "IndexCompatibilityError",
    "PaperContent",
    "ChunkPayload",
    "process_paper",
    "ingest_academic_literature",
]
