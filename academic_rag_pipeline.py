#!/usr/bin/env python3
"""Build and query a resumable local FAISS index of academic literature.

The ingestion path prefers direct or OpenAlex-discovered open-access PDFs. A
catalog abstract is used when full text cannot be obtained or extracted. PDFs
are temporary artifacts and are deleted after every extraction attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple
from urllib.parse import quote, unquote, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from olfactory.academic_evidence import (
    ACADEMIC_EVIDENCE_SCHEMA_VERSION,
    AcademicDocument,
    AcademicEvidence,
    AcademicEvidenceStore,
    annotate_chunk_provenance,
    evidence_records_from_document,
    sha256_text,
)

try:  # Prefer the current module name; retain compatibility with older PyMuPDF.
    import pymupdf as fitz
except ImportError:  # pragma: no cover - exercised through dependency guards.
    try:
        import fitz
    except ImportError:  # pragma: no cover - exercised through dependency guards.
        fitz = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised through dependency guards.
    FAISS = None
    Document = None
    HuggingFaceEmbeddings = None
    RecursiveCharacterTextSplitter = None


LOGGER = logging.getLogger("academic_rag")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "academic_literature.csv"
DEFAULT_PDF_DIR = SCRIPT_DIR / "academic_pdfs"
DEFAULT_INDEX_DIR = SCRIPT_DIR / "faiss_academic_index"

SCHEMA_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = frozenset({"Source", "Title", "Abstract", "Link"})
PDF_HEADER_SEARCH_BYTES = 1024
MIN_EXTRACTED_TEXT_CHARS = 100


class PipelineError(RuntimeError):
    """Base exception for expected pipeline failures."""


class DownloadError(PipelineError):
    """Raised when a URL does not yield a safe, readable PDF payload."""


class ExtractionError(PipelineError):
    """Raised when PyMuPDF cannot obtain useful text from a PDF."""


class IndexCompatibilityError(PipelineError):
    """Raised when an existing index was built with incompatible inputs."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    pdf_dir: Path = DEFAULT_PDF_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    batch_size: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32
    request_delay: float = 2.0
    max_pdf_bytes: int = 100 * 1024 * 1024
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    contact_email: str | None = None
    device: str = "cpu"
    evidence_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "csv_path", Path(self.csv_path))
        object.__setattr__(self, "pdf_dir", Path(self.pdf_dir))
        object.__setattr__(self, "index_dir", Path(self.index_dir))
        if self.evidence_path is None:
            object.__setattr__(self, "evidence_path", self.index_dir / "academic_evidence.jsonl")
        else:
            object.__setattr__(self, "evidence_path", Path(self.evidence_path))
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if self.request_delay < 0:
            raise ValueError("request_delay cannot be negative")
        if self.max_pdf_bytes < PDF_HEADER_SEARCH_BYTES:
            raise ValueError("max_pdf_bytes is too small to validate a PDF")


@dataclass(frozen=True, slots=True)
class PDFResolution:
    url: str | None
    source: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class PaperContent:
    paper_id: str
    title: str
    link: str
    source: str
    doi: str
    published_date: str
    text: str
    content_type: str
    evidence_records: Tuple[AcademicEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    document_id: str
    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SimpleDocument:
    """Small Document-compatible value used by isolated tests."""

    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BatchInfo:
    path: Path
    batch_number: int
    paper_ids: tuple[str, ...]
    paper_count: int
    chunk_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    doi = unquote(clean_scalar(value))
    lowered = doi.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi.strip()


def paper_id_for(row: Mapping[str, Any]) -> str:
    doi = normalize_doi(row.get("DOI"))
    identity = f"doi:{doi.lower()}" if doi else f"link:{clean_scalar(row.get('Link'))}"
    return f"paper-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def read_catalog(csv_path: Path) -> tuple[pd.DataFrame, str]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Academic catalog not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    links = frame["Link"].map(clean_scalar)
    skipped = int((links == "").sum())
    frame = frame.loc[links != ""].copy()
    frame.loc[:, "Link"] = links.loc[links != ""]
    if skipped:
        LOGGER.warning("Skipped %d catalog rows with an empty Link", skipped)
    return frame.reset_index(drop=True), sha256_file(csv_path)


def build_http_session(contact_email: str | None = None) -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    user_agent = "ScentMoleculeStudio-AcademicRAG/1.0"
    if contact_email:
        user_agent += f" (mailto:{contact_email})"
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/pdf, application/json;q=0.9, */*;q=0.1",
        }
    )
    return session


def looks_like_direct_pdf(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    return path.endswith(".pdf") or (
        hostname in {"arxiv.org", "www.arxiv.org"} and path.startswith("/pdf/")
    )


def doi_from_row(row: Mapping[str, Any]) -> str:
    doi = normalize_doi(row.get("DOI"))
    if doi:
        return doi
    link = clean_scalar(row.get("Link"))
    parsed = urlparse(link)
    if (parsed.hostname or "").lower() in {"doi.org", "www.doi.org", "dx.doi.org"}:
        return normalize_doi(parsed.path.lstrip("/"))
    return ""


def _openalex_pdf_candidates(payload: Mapping[str, Any]) -> Iterable[str]:
    locations: list[Any] = [payload.get("best_oa_location"), payload.get("primary_location")]
    locations.extend(payload.get("locations") or [])
    seen: set[str] = set()
    for location in locations:
        if not isinstance(location, Mapping) or not location.get("is_oa"):
            continue
        pdf_url = clean_scalar(location.get("pdf_url"))
        if pdf_url and pdf_url not in seen:
            seen.add(pdf_url)
            yield pdf_url


def resolve_pdf_url(
    row: Mapping[str, Any], session: requests.Session, config: PipelineConfig
) -> PDFResolution:
    link = clean_scalar(row.get("Link"))
    if looks_like_direct_pdf(link):
        return PDFResolution(link, "direct", None)

    doi = doi_from_row(row)
    if not doi:
        return PDFResolution(None, "none", "No direct PDF or DOI is available")

    work_id = quote(f"https://doi.org/{doi}", safe="/:")
    openalex_url = f"https://api.openalex.org/works/{work_id}"
    params = {"mailto": config.contact_email} if config.contact_email else None
    try:
        with session.get(
            openalex_url,
            params=params,
            timeout=(config.connect_timeout, config.read_timeout),
        ) as response:
            if response.status_code == 404:
                return PDFResolution(None, "openalex", "DOI was not found in OpenAlex")
            response.raise_for_status()
            payload = response.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        return PDFResolution(None, "openalex", f"OpenAlex lookup failed: {exc}")

    if not isinstance(payload, Mapping):
        return PDFResolution(None, "openalex", "OpenAlex returned an invalid response")
    pdf_url = next(iter(_openalex_pdf_candidates(payload)), None)
    if not pdf_url:
        return PDFResolution(None, "openalex", "No open-access PDF URL was reported")
    return PDFResolution(pdf_url, "openalex", None)


def download_pdf(
    url: str,
    destination: Path,
    session: requests.Session,
    config: PipelineConfig,
) -> int:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(".part")
    for stale_path in (partial_path, destination):
        if stale_path.exists():
            stale_path.unlink()

    bytes_written = 0
    header_probe = bytearray()
    try:
        with session.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=(config.connect_timeout, config.read_timeout),
        ) as response:
            response.raise_for_status()
            content_length = clean_scalar(response.headers.get("Content-Length"))
            if content_length:
                try:
                    if int(content_length) > config.max_pdf_bytes:
                        raise DownloadError(
                            f"PDF exceeds the {config.max_pdf_bytes}-byte safety limit"
                        )
                except ValueError:
                    LOGGER.debug("Ignoring malformed Content-Length: %s", content_length)

            with partial_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > config.max_pdf_bytes:
                        raise DownloadError(
                            f"PDF exceeds the {config.max_pdf_bytes}-byte safety limit"
                        )
                    if len(header_probe) < PDF_HEADER_SEARCH_BYTES:
                        remaining = PDF_HEADER_SEARCH_BYTES - len(header_probe)
                        header_probe.extend(chunk[:remaining])
                    handle.write(chunk)

        if b"%PDF-" not in header_probe:
            content_type = clean_scalar(response.headers.get("Content-Type"))
            raise DownloadError(
                f"Response does not contain a PDF signature (Content-Type: {content_type or 'unknown'})"
            )
        if bytes_written == 0:
            raise DownloadError("The PDF response was empty")
        os.replace(partial_path, destination)
        return bytes_written
    except (requests.RequestException, OSError) as exc:
        raise DownloadError(f"PDF download failed: {exc}") from exc
    finally:
        if partial_path.exists():
            partial_path.unlink()


def extract_pdf_text(pdf_path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required. Install packages from requirements-rag.txt")
    try:
        with fitz.open(str(pdf_path)) as document:
            if document.needs_pass:
                raise ExtractionError("PDF is encrypted and requires a password")
            pages = [page.get_text("text", sort=True).strip() for page in document]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"PyMuPDF could not read the PDF: {exc}") from exc

    text = "\n\n".join(page for page in pages if page).strip()
    if len(text) < MIN_EXTRACTED_TEXT_CHARS:
        raise ExtractionError("PDF contains no usable text; it may be image-only")
    return text


def _paper_metadata(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "paper_id": paper_id_for(row),
        "title": clean_scalar(row.get("Title")) or "Untitled paper",
        "link": clean_scalar(row.get("Link")),
        "source": clean_scalar(row.get("Source")),
        "doi": doi_from_row(row),
        "published_date": clean_scalar(row.get("Published Date")),
    }


def _source_type(source: str) -> str:
    """Classify catalog provenance without claiming peer-review status."""
    normalized = source.strip().lower()
    if "arxiv" in normalized or "preprint" in normalized:
        return "PREPRINT"
    if "review" in normalized:
        return "REVIEW"
    if "sensor" in normalized or "machine olfaction" in normalized:
        return "SENSOR_ONLY"
    if "pubmed" in normalized or "journal" in normalized:
        return "PRIMARY_STUDY_OR_REVIEW_UNCLASSIFIED"
    return "UNKNOWN"


def _build_paper_content(
    metadata: Mapping[str, str],
    text: str,
    content_type: str,
    *,
    open_access: bool = False,
    license_status: str | None = None,
) -> PaperContent:
    """Attach auditable evidence candidates while preserving raw paper text."""
    resolved_license = (
        license_status
        or ("OA_CONFIRMED" if open_access else "OA_ROUTE_UNVERIFIED")
        if content_type == "full_text"
        else "CATALOG_UNVERIFIED"
    )
    document = AcademicDocument(
        paper_id=metadata["paper_id"],
        title=metadata["title"],
        link=metadata["link"],
        source=metadata["source"],
        doi=metadata["doi"],
        published_date=metadata["published_date"],
        content_type=content_type,
        text_sha256=sha256_text(text),
        source_type=_source_type(metadata["source"]),
        # A PDF route is not itself proof of an open license.  Keep that
        # distinction explicit so exact evidence remains fail-closed.
        license_status=resolved_license,
        open_access=bool(open_access and content_type == "full_text"),
    )
    evidence = evidence_records_from_document(document, text)
    return PaperContent(
        **metadata,
        text=text,
        content_type=content_type,
        evidence_records=evidence,
    )


def process_paper(
    row: Mapping[str, Any],
    config: PipelineConfig,
    session: requests.Session,
    *,
    resolver: Callable[[Mapping[str, Any], requests.Session, PipelineConfig], PDFResolution] = resolve_pdf_url,
    downloader: Callable[[str, Path, requests.Session, PipelineConfig], int] = download_pdf,
    extractor: Callable[[Path], str] = extract_pdf_text,
) -> PaperContent | None:
    metadata = _paper_metadata(row)
    abstract = clean_scalar(row.get("Abstract"))
    try:
        resolution = resolver(row, session, config)
    except Exception as exc:
        LOGGER.warning("PDF resolution failed for %s: %s", metadata["title"], exc)
        resolution = PDFResolution(None, "resolver", str(exc))

    if resolution.url:
        pdf_path = config.pdf_dir / f"{metadata['paper_id']}.pdf"
        partial_path = pdf_path.with_suffix(".part")
        try:
            size = downloader(resolution.url, pdf_path, session, config)
            LOGGER.debug("Downloaded %s bytes for %s", size, metadata["title"])
            text = extractor(pdf_path)
            parsed_host = (urlparse(resolution.url).hostname or "").lower()
            route_is_oa = resolution.source == "openalex" or "arxiv.org" in parsed_host
            return _build_paper_content(
                metadata,
                text,
                "full_text",
                open_access=route_is_oa,
            )
        except Exception as exc:
            LOGGER.warning("Full-text processing failed for %s: %s", metadata["title"], exc)
        finally:
            for temporary_path in (pdf_path, partial_path):
                try:
                    if temporary_path.exists():
                        temporary_path.unlink()
                        LOGGER.debug("Deleted temporary file %s", temporary_path)
                except OSError as exc:
                    LOGGER.error("Could not delete temporary PDF %s: %s", temporary_path, exc)
    elif resolution.reason:
        LOGGER.info("No open PDF for %s: %s", metadata["title"], resolution.reason)

    if abstract:
        return _build_paper_content(metadata, abstract, "abstract")
    LOGGER.error("Skipping %s: neither full text nor Abstract is available", metadata["title"])
    return None


def _require_chunking_dependencies() -> None:
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError(
            "LangChain text splitters are required. Install packages from requirements-rag.txt"
        )


def _require_vector_dependencies() -> None:
    missing = []
    if FAISS is None:
        missing.append("langchain-community/faiss-cpu")
    if Document is None:
        missing.append("langchain-core")
    if HuggingFaceEmbeddings is None:
        missing.append("langchain-huggingface/sentence-transformers")
    if missing:
        raise RuntimeError(
            "Missing RAG dependencies: " + ", ".join(missing) + ". Install requirements-rag.txt"
        )


def chunk_paper(
    paper: PaperContent,
    config: PipelineConfig,
    *,
    splitter: Any | None = None,
) -> list[ChunkPayload]:
    if splitter is None:
        _require_chunking_dependencies()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    chunks: list[ChunkPayload] = []
    for index, text in enumerate(splitter.split_text(paper.text)):
        cleaned = text.strip()
        if not cleaned:
            continue
        metadata = {
            "paper_id": paper.paper_id,
            "chunk_index": index,
            "title": paper.title,
            "link": paper.link,
            # Keep the CSV-facing names as well as normalized lowercase keys.
            "Title": paper.title,
            "Link": paper.link,
            "source": paper.source,
            "Source": paper.source,
            "doi": paper.doi,
            "published_date": paper.published_date,
            "content_type": paper.content_type,
        }
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"academic-rag:{paper.paper_id}:{index}"))
        chunks.append(ChunkPayload(document_id, cleaned, metadata))
    return chunks


def pipeline_config_payload(config: PipelineConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "normalize_embeddings": True,
    }


def pipeline_fingerprint(config: PipelineConfig) -> str:
    encoded = json.dumps(pipeline_config_payload(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def batch_manifest_base(config: PipelineConfig, csv_sha256: str) -> dict[str, Any]:
    return {
        **pipeline_config_payload(config),
        "pipeline_fingerprint": pipeline_fingerprint(config),
        "csv_sha256": csv_sha256,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_index(config: PipelineConfig, csv_sha256: str) -> None:
    config.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.index_dir / "pipeline_manifest.json"
    expected = {
        **batch_manifest_base(config, csv_sha256),
        "created_at": utc_now(),
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(current, config, csv_sha256, manifest_path)
        return
    unexpected = [path for path in config.index_dir.iterdir() if not path.name.startswith(".")]
    if unexpected:
        raise IndexCompatibilityError(
            f"Index directory has no pipeline manifest: {config.index_dir}. Use --rebuild."
        )
    _atomic_write_json(manifest_path, expected)


def _validate_manifest(
    manifest: Mapping[str, Any],
    config: PipelineConfig,
    csv_sha256: str,
    source_path: Path,
) -> None:
    if manifest.get("pipeline_fingerprint") != pipeline_fingerprint(config):
        raise IndexCompatibilityError(
            f"Index configuration differs from {source_path}. Use --rebuild."
        )
    if manifest.get("csv_sha256") != csv_sha256:
        raise IndexCompatibilityError(
            f"CSV content differs from {source_path}. Use --rebuild."
        )


def _assert_trusted_batch_path(batch_path: Path, index_dir: Path) -> None:
    resolved_root = index_dir.resolve()
    resolved_batch = batch_path.resolve()
    if not resolved_batch.is_relative_to(resolved_root):
        raise PipelineError(f"Refusing to load an index outside {resolved_root}")
    if batch_path.is_symlink():
        raise PipelineError(f"Refusing to load a symlinked FAISS batch: {batch_path}")
    for filename in ("index.faiss", "index.pkl", "manifest.json"):
        candidate = batch_path / filename
        if candidate.is_symlink():
            raise PipelineError(f"Refusing to load a symlinked index artifact: {candidate}")


def _validate_batch_checksums(batch_path: Path, manifest: Mapping[str, Any]) -> None:
    checksums = manifest.get("artifact_sha256")
    if not isinstance(checksums, Mapping):
        raise IndexCompatibilityError(f"Missing artifact checksums in {batch_path}")
    for filename in ("index.faiss", "index.pkl"):
        artifact = batch_path / filename
        if not artifact.is_file():
            raise IndexCompatibilityError(f"Missing FAISS artifact: {artifact}")
        if checksums.get(filename) != sha256_file(artifact):
            raise IndexCompatibilityError(f"FAISS artifact checksum mismatch: {artifact}")


def discover_batches(config: PipelineConfig, csv_sha256: str) -> list[BatchInfo]:
    batches: list[BatchInfo] = []
    all_paper_ids: set[str] = set()
    for batch_path in sorted(config.index_dir.glob("batch_[0-9][0-9][0-9][0-9][0-9][0-9]")):
        if not batch_path.is_dir():
            continue
        _assert_trusted_batch_path(batch_path, config.index_dir)
        manifest_path = batch_path / "manifest.json"
        if not manifest_path.is_file():
            raise IndexCompatibilityError(f"Batch manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(manifest, config, csv_sha256, manifest_path)
        _validate_batch_checksums(batch_path, manifest)
        papers = manifest.get("papers") or []
        paper_ids = tuple(clean_scalar(paper.get("paper_id")) for paper in papers)
        if not all(paper_ids):
            raise IndexCompatibilityError(f"Batch contains an invalid paper_id: {manifest_path}")
        if int(manifest.get("paper_count", -1)) != len(papers):
            raise IndexCompatibilityError(f"Batch paper count is inconsistent: {manifest_path}")
        if int(manifest.get("chunk_count", 0)) < 1:
            raise IndexCompatibilityError(f"Batch chunk count is invalid: {manifest_path}")
        duplicate_ids = all_paper_ids.intersection(paper_ids)
        if duplicate_ids:
            raise IndexCompatibilityError(
                f"Paper appears in multiple FAISS batches: {sorted(duplicate_ids)[0]}"
            )
        all_paper_ids.update(paper_ids)
        batches.append(
            BatchInfo(
                path=batch_path,
                batch_number=int(manifest["batch_number"]),
                paper_ids=paper_ids,
                paper_count=int(manifest["paper_count"]),
                chunk_count=int(manifest["chunk_count"]),
            )
        )
    return batches


def processed_paper_ids(batches: Sequence[BatchInfo]) -> set[str]:
    return {paper_id for batch in batches for paper_id in batch.paper_ids}


def _create_embeddings(model_name: str, device: str, batch_size: int = 32) -> Any:
    _require_vector_dependencies()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


def save_faiss_batch(
    config: PipelineConfig,
    *,
    batch_number: int,
    csv_sha256: str,
    chunks: Sequence[ChunkPayload],
    papers: Sequence[Mapping[str, Any]],
    embeddings: Any,
    faiss_cls: Any | None = None,
    document_cls: Any | None = None,
) -> BatchInfo:
    if not chunks or not papers:
        raise ValueError("Cannot save an empty FAISS batch")
    if faiss_cls is None or document_cls is None:
        _require_vector_dependencies()
        faiss_cls = faiss_cls or FAISS
        document_cls = document_cls or Document

    final_path = config.index_dir / f"batch_{batch_number:06d}"
    if final_path.exists():
        raise FileExistsError(f"FAISS batch already exists: {final_path}")
    temporary_path = config.index_dir / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
    documents = [
        document_cls(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks
    ]
    ids = [chunk.document_id for chunk in chunks]
    try:
        store = faiss_cls.from_documents(
            documents,
            embeddings,
            ids=ids,
            normalize_L2=True,
        )
        store.save_local(str(temporary_path))
        artifact_sha256 = {
            filename: sha256_file(temporary_path / filename)
            for filename in ("index.faiss", "index.pkl")
        }
        manifest = {
            **batch_manifest_base(config, csv_sha256),
            "batch_number": batch_number,
            "created_at": utc_now(),
            "paper_count": len(papers),
            "chunk_count": len(chunks),
            "papers": [dict(paper) for paper in papers],
            "artifact_sha256": artifact_sha256,
        }
        _atomic_write_json(temporary_path / "manifest.json", manifest)
        os.replace(temporary_path, final_path)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    LOGGER.info(
        "Saved FAISS batch %d with %d papers and %d chunks",
        batch_number,
        len(papers),
        len(chunks),
    )
    return BatchInfo(
        path=final_path,
        batch_number=batch_number,
        paper_ids=tuple(clean_scalar(paper.get("paper_id")) for paper in papers),
        paper_count=len(papers),
        chunk_count=len(chunks),
    )


def _safe_rebuild_index(index_dir: Path) -> None:
    if index_dir.is_symlink():
        raise ValueError(f"Refusing to recursively remove a symlinked index path: {index_dir}")
    resolved = index_dir.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), SCRIPT_DIR.resolve()}
    if resolved in protected or len(resolved.parts) < 4:
        raise ValueError(f"Refusing to recursively remove unsafe index path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def ingest_academic_literature(
    config: PipelineConfig,
    *,
    rebuild: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    frame, csv_sha256 = read_catalog(config.csv_path)
    if rebuild:
        _safe_rebuild_index(config.index_dir)
    initialize_index(config, csv_sha256)
    existing_batches = discover_batches(config, csv_sha256)
    completed_ids = processed_paper_ids(existing_batches)
    evidence_store = AcademicEvidenceStore(config.evidence_path)
    evidence_store.assert_compatible(csv_sha256)
    next_batch = max((batch.batch_number for batch in existing_batches), default=0) + 1

    unprocessed_count = sum(
        1 for _, row in frame.iterrows() if paper_id_for(row) not in completed_ids
    )
    if not unprocessed_count:
        existing_evidence_count = len(evidence_store.load())
        LOGGER.info("All %d catalog papers are already indexed", len(frame))
        if not existing_evidence_count:
            LOGGER.warning(
                "Academic evidence store is empty for the existing index; "
                "run ingest --rebuild to derive structure mentions."
            )
        return {
            "catalog_rows": len(frame),
            "processed": 0,
            "full_text": 0,
            "abstract": 0,
            "failed": 0,
            "skipped_existing": len(frame),
            "evidence_records": existing_evidence_count,
        }

    embeddings = _create_embeddings(
        config.embedding_model,
        config.device,
        config.embedding_batch_size,
    )
    _require_chunking_dependencies()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    pending_chunks: list[ChunkPayload] = []
    pending_papers: list[dict[str, Any]] = []
    pending_evidence: list[AcademicEvidence] = []
    seen_ids = set(completed_ids)
    counters = {
        "catalog_rows": len(frame),
        "processed": 0,
        "full_text": 0,
        "abstract": 0,
        "failed": 0,
        "skipped_existing": len(completed_ids),
        "evidence_records": 0,
    }

    def flush() -> None:
        nonlocal next_batch
        if not pending_papers:
            return
        save_faiss_batch(
            config,
            batch_number=next_batch,
            csv_sha256=csv_sha256,
            chunks=pending_chunks,
            papers=pending_papers,
            embeddings=embeddings,
        )
        evidence_store.upsert(pending_evidence, dataset_sha256=csv_sha256)
        next_batch += 1
        pending_chunks.clear()
        pending_papers.clear()
        pending_evidence.clear()

    with build_http_session(config.contact_email) as session:
        for catalog_index, row in frame.iterrows():
            paper_id = paper_id_for(row)
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            try:
                content = process_paper(row, config, session)
                if content is None:
                    counters["failed"] += 1
                    continue
                chunks = chunk_paper(content, config, splitter=splitter)
                if not chunks:
                    LOGGER.error("No chunks were produced for %s", content.title)
                    counters["failed"] += 1
                    continue
                evidence_records = annotate_chunk_provenance(
                    content.evidence_records,
                    chunks,
                )
                pending_chunks.extend(chunks)
                pending_papers.append(
                    {
                        "paper_id": content.paper_id,
                        "title": content.title,
                        "link": content.link,
                        "content_type": content.content_type,
                        "chunk_count": len(chunks),
                    }
                )
                pending_evidence.extend(evidence_records)
                counters["evidence_records"] += len(evidence_records)
                counters["processed"] += 1
                counters[content.content_type] += 1
                if content.content_type == "full_text":
                    LOGGER.info(
                        "Processed and deleted PDF %d/%d: %s",
                        catalog_index + 1,
                        len(frame),
                        content.title,
                    )
                else:
                    LOGGER.info(
                        "Indexed abstract fallback %d/%d: %s",
                        catalog_index + 1,
                        len(frame),
                        content.title,
                    )
            except Exception:
                counters["failed"] += 1
                LOGGER.exception("Unexpected processing error for row %d", catalog_index + 1)
            finally:
                if config.request_delay:
                    sleep_fn(config.request_delay)
            if len(pending_papers) >= config.batch_size:
                flush()
        flush()

    LOGGER.info(
        "Ingestion complete: processed=%d full_text=%d abstract=%d failed=%d existing=%d",
        counters["processed"],
        counters["full_text"],
        counters["abstract"],
        counters["failed"],
        counters["skipped_existing"],
    )
    return counters


def _load_pipeline_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "pipeline_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Run the ingest command first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise IndexCompatibilityError(
            f"Unsupported index schema in {manifest_path}: {payload.get('schema_version')}"
        )
    return payload


def _query_config(index_dir: Path, manifest: Mapping[str, Any], device: str) -> PipelineConfig:
    return PipelineConfig(
        index_dir=index_dir,
        embedding_model=clean_scalar(manifest.get("embedding_model"))
        or DEFAULT_EMBEDDING_MODEL,
        chunk_size=int(manifest.get("chunk_size", 1000)),
        chunk_overlap=int(manifest.get("chunk_overlap", 200)),
        device=device,
    )


def query_academic_knowledge(
    query_string: str,
    top_k: int = 5,
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Print and return the most relevant chunks from a trusted local index."""

    query_string = clean_scalar(query_string)
    if not query_string:
        raise ValueError("query_string cannot be empty")
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")

    index_dir = Path(index_dir)
    manifest = _load_pipeline_manifest(index_dir)
    config = _query_config(index_dir, manifest, device)
    csv_sha256 = clean_scalar(manifest.get("csv_sha256"))
    batches = discover_batches(config, csv_sha256)
    if not batches:
        raise PipelineError(f"No completed FAISS batches were found in {index_dir}")

    embeddings = _create_embeddings(config.embedding_model, config.device)
    merged_store = None
    for batch in batches:
        _assert_trusted_batch_path(batch.path, index_dir)
        store = FAISS.load_local(
            str(batch.path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        if merged_store is None:
            merged_store = store
        else:
            merged_store.merge_from(store)

    matches = merged_store.similarity_search_with_score(query_string, k=top_k)
    results: list[dict[str, Any]] = []
    for rank, (document, squared_l2_distance) in enumerate(matches, start=1):
        metadata = dict(document.metadata)
        similarity = max(-1.0, min(1.0, 1.0 - float(squared_l2_distance) / 2.0))
        result = {
            "rank": rank,
            "title": clean_scalar(metadata.get("title") or metadata.get("Title"))
            or "Untitled paper",
            "link": clean_scalar(metadata.get("link") or metadata.get("Link")),
            "content_type": clean_scalar(metadata.get("content_type")) or "unknown",
            "score": similarity,
            "raw_distance": float(squared_l2_distance),
            "text": document.page_content,
            "metadata": metadata,
        }
        results.append(result)
        print(f"\n[{rank}] {result['title']}")
        print(f"Link: {result['link']}")
        print(f"Content: {result['content_type']} | Similarity: {similarity:.4f}")
        print(result["text"])
        print("-" * 88)
    return results


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query a local FAISS index of academic literature."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Download, extract, and index papers")
    ingest_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    ingest_parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ingest_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ingest_parser.add_argument(
        "--evidence-path",
        type=Path,
        help="Path for the derived academic evidence JSONL store",
    )
    ingest_parser.add_argument("--batch-size", type=int, default=50)
    ingest_parser.add_argument("--request-delay", type=float, default=2.0)
    ingest_parser.add_argument("--contact-email")
    ingest_parser.add_argument("--device", default="cpu")
    ingest_parser.add_argument("--rebuild", action="store_true")

    query_parser = subparsers.add_parser("query", help="Search the local academic index")
    query_parser.add_argument("query_string")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    query_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.command == "ingest":
            config = PipelineConfig(
                csv_path=args.csv,
                pdf_dir=args.pdf_dir,
                index_dir=args.index_dir,
                evidence_path=args.evidence_path,
                batch_size=args.batch_size,
                request_delay=args.request_delay,
                contact_email=args.contact_email,
                device=args.device,
            )
            ingest_academic_literature(config, rebuild=args.rebuild)
        else:
            query_academic_knowledge(
                args.query_string,
                top_k=args.top_k,
                index_dir=args.index_dir,
                device=args.device,
            )
    except (PipelineError, FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:  # Keep CLI failures concise; --verbose still aids diagnosis.
        LOGGER.error("Unexpected pipeline failure: %s", exc)
        LOGGER.debug("Unexpected pipeline failure details", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
