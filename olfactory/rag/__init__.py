"""Application-facing boundary for the local academic RAG service.

The CLI remains backwards compatible while imports used by API services can be
scoped to ``olfactory.rag`` instead of importing training modules.  The heavy
LangChain/FAISS dependencies are imported only by the concrete modules.
"""

from .ingest import ingest_academic_literature, process_paper
from .query import query_academic_knowledge
from .service import AcademicRAGService

__all__ = ["ingest_academic_literature", "process_paper", "query_academic_knowledge", "AcademicRAGService"]
