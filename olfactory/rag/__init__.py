"""Application-facing boundary for the local academic RAG service.

The CLI remains backwards compatible while imports used by API services can be
scoped to ``olfactory.rag`` instead of importing training modules.  The heavy
LangChain/FAISS dependencies are imported only by the concrete modules.
"""

from .ingest import ingest_academic_literature, process_paper
from .query import query_academic_knowledge
from .service import AcademicRAGService
from .evidence import (
    ACADEMIC_EVIDENCE_MANIFEST_VERSION,
    ACADEMIC_EVIDENCE_SCHEMA_VERSION,
    AcademicDocument,
    AcademicEvidence,
    AcademicEvidenceService,
    AcademicEvidenceStore,
    AcademicEvidenceSummary,
    EvidenceStatus,
    MatchLevel,
    NormalizedStructure,
    ReviewState,
    StructureMention,
    StructureMentionKind,
    evidence_records_from_document,
    annotate_chunk_provenance,
    extract_structure_mentions,
    normalize_structure,
    verify_academic_evidence,
)

__all__ = [
    "ingest_academic_literature",
    "process_paper",
    "query_academic_knowledge",
    "AcademicRAGService",
    "ACADEMIC_EVIDENCE_SCHEMA_VERSION",
    "ACADEMIC_EVIDENCE_MANIFEST_VERSION",
    "AcademicDocument",
    "AcademicEvidence",
    "AcademicEvidenceService",
    "AcademicEvidenceStore",
    "AcademicEvidenceSummary",
    "EvidenceStatus",
    "MatchLevel",
    "NormalizedStructure",
    "ReviewState",
    "StructureMention",
    "StructureMentionKind",
    "evidence_records_from_document",
    "annotate_chunk_provenance",
    "extract_structure_mentions",
    "normalize_structure",
    "verify_academic_evidence",
]
