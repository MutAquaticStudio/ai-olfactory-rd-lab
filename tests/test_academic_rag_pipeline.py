import json
from pathlib import Path

import pandas as pd
import pytest

import academic_rag_pipeline as rag


class FakeResponse:
    def __init__(self, chunks, *, headers=None, status_code=200, json_data=None):
        self._chunks = chunks
        self.headers = headers or {}
        self.status_code = status_code
        self._json_data = json_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise rag.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requested_urls = []

    def get(self, url, **kwargs):
        del kwargs
        self.requested_urls.append(url)
        return self.responses.pop(0)


def make_config(tmp_path, **overrides):
    values = {
        "csv_path": tmp_path / "academic_literature.csv",
        "pdf_dir": tmp_path / "academic_pdfs",
        "index_dir": tmp_path / "faiss_academic_index",
        "request_delay": 0,
    }
    values.update(overrides)
    return rag.PipelineConfig(**values)


def test_read_catalog_validates_required_columns_and_skips_empty_links(tmp_path):
    config = make_config(tmp_path)
    pd.DataFrame(
        [
            {"Source": "Journal", "Title": "Paper", "Abstract": "Text", "Link": "https://example.org/a"},
            {"Source": "Journal", "Title": "No link", "Abstract": "Text", "Link": None},
        ]
    ).to_csv(config.csv_path, index=False)

    frame, csv_sha = rag.read_catalog(config.csv_path)

    assert list(frame["Title"]) == ["Paper"]
    assert len(csv_sha) == 64

    pd.DataFrame([{"Title": "Missing columns"}]).to_csv(config.csv_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        rag.read_catalog(config.csv_path)


def test_download_pdf_rejects_html_and_removes_partial_file(tmp_path):
    config = make_config(tmp_path)
    destination = config.pdf_dir / "paper.pdf"
    session = FakeSession(
        [FakeResponse([b"<html>not a pdf</html>"], headers={"Content-Type": "text/html"})]
    )

    with pytest.raises(rag.DownloadError, match="PDF signature"):
        rag.download_pdf("https://example.org/paper.pdf", destination, session, config)

    assert not destination.exists()
    assert not destination.with_suffix(".part").exists()


def test_process_paper_deletes_pdf_after_success_and_after_extraction_error(tmp_path):
    config = make_config(tmp_path)
    row = {
        "Source": "Journal",
        "Title": "Paper",
        "Abstract": "Fallback abstract",
        "Link": "https://example.org/paper.pdf",
        "DOI": "10.1000/example",
    }

    def resolver(*args):
        del args
        return rag.PDFResolution("https://example.org/paper.pdf", "direct", None)

    def downloader(url, destination, session, config):
        del url, session, config
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-fixture")
        return destination.stat().st_size

    content = rag.process_paper(
        row,
        config,
        session=object(),
        resolver=resolver,
        downloader=downloader,
        extractor=lambda path: "Extracted full text from the paper.",
    )
    assert content.content_type == "full_text"
    assert not list(config.pdf_dir.glob("*"))

    fallback = rag.process_paper(
        row,
        config,
        session=object(),
        resolver=resolver,
        downloader=downloader,
        extractor=lambda path: (_ for _ in ()).throw(rag.ExtractionError("corrupt")),
    )
    assert fallback.content_type == "abstract"
    assert fallback.text == "Fallback abstract"
    assert not list(config.pdf_dir.glob("*"))


def test_openalex_resolution_uses_only_open_access_pdf_url(tmp_path):
    config = make_config(tmp_path)
    row = {
        "Link": "https://doi.org/10.1000/example",
        "DOI": "10.1000/example",
    }
    session = FakeSession(
        [
            FakeResponse(
                [],
                json_data={
                    "best_oa_location": {
                        "is_oa": True,
                        "pdf_url": "https://repository.example.org/article.pdf",
                    }
                },
            )
        ]
    )

    resolution = rag.resolve_pdf_url(row, session, config)

    assert resolution.url == "https://repository.example.org/article.pdf"
    assert resolution.source == "openalex"
    assert "api.openalex.org" in session.requested_urls[0]


def test_chunk_metadata_contains_source_identity_and_content_type(tmp_path):
    config = make_config(tmp_path, chunk_size=10, chunk_overlap=2)
    paper = rag.PaperContent(
        paper_id="paper-1",
        title="Odor paper",
        link="https://example.org/paper",
        source="Journal",
        doi="10.1000/example",
        published_date="2026-01-01",
        text="abcdefghij klmnopqrst",
        content_type="full_text",
    )

    class Splitter:
        def split_text(self, text):
            assert text == paper.text
            return ["abcdefghij", "ijklmnopqr"]

    chunks = rag.chunk_paper(paper, config, splitter=Splitter())

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk.metadata["title"] == "Odor paper" for chunk in chunks)
    assert all(chunk.metadata["link"] == paper.link for chunk in chunks)
    assert all(chunk.metadata["Title"] == "Odor paper" for chunk in chunks)
    assert all(chunk.metadata["Link"] == paper.link for chunk in chunks)
    assert all(chunk.metadata["content_type"] == "full_text" for chunk in chunks)
    assert chunks[0].document_id != chunks[1].document_id


def test_batch_manifest_is_discoverable_and_resume_safe(tmp_path):
    config = make_config(tmp_path)
    config.index_dir.mkdir(parents=True)
    rag.initialize_index(config, "a" * 64)
    chunks = [
        rag.ChunkPayload(
            document_id="doc-1",
            page_content="A chunk",
            metadata={"paper_id": "paper-1", "title": "Paper", "link": "https://example.org"},
        )
    ]

    class FakeStore:
        @classmethod
        def from_documents(cls, documents, embedding, ids, normalize_L2):
            assert documents[0].page_content == "A chunk"
            assert ids == ["doc-1"]
            assert normalize_L2 is True
            del embedding
            return cls()

        def save_local(self, folder_path):
            folder = Path(folder_path)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "index.faiss").write_bytes(b"faiss")
            (folder / "index.pkl").write_bytes(b"pickle")

    rag.save_faiss_batch(
        config,
        batch_number=1,
        csv_sha256="a" * 64,
        chunks=chunks,
        papers=[{"paper_id": "paper-1", "content_type": "full_text"}],
        embeddings=object(),
        faiss_cls=FakeStore,
        document_cls=rag.SimpleDocument,
    )

    batches = rag.discover_batches(config, "a" * 64)
    assert len(batches) == 1
    assert batches[0].paper_ids == ("paper-1",)
    assert rag.processed_paper_ids(batches) == {"paper-1"}

    manifest = json.loads((batches[0].path / "manifest.json").read_text())
    assert manifest["paper_count"] == 1
    assert manifest["chunk_count"] == 1


def test_query_prints_title_link_and_returns_ranked_results(tmp_path, capsys, monkeypatch):
    config = make_config(tmp_path)
    config.index_dir.mkdir(parents=True)
    rag.initialize_index(config, "b" * 64)
    batch_dir = config.index_dir / "batch_000001"
    batch_dir.mkdir()
    (batch_dir / "index.faiss").write_bytes(b"faiss")
    (batch_dir / "index.pkl").write_bytes(b"pickle")
    artifact_sha256 = {
        filename: rag.sha256_file(batch_dir / filename)
        for filename in ("index.faiss", "index.pkl")
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                **rag.batch_manifest_base(config, "b" * 64),
                "batch_number": 1,
                "paper_count": 1,
                "chunk_count": 1,
                "papers": [{"paper_id": "paper-1", "content_type": "abstract"}],
                "artifact_sha256": artifact_sha256,
            }
        )
    )

    class Document:
        page_content = "Relevant passage"
        metadata = {
            "title": "Relevant paper",
            "link": "https://example.org/relevant",
            "content_type": "abstract",
        }

    class FakeLoadedStore:
        def similarity_search_with_score(self, query, k):
            assert query == "molecular odor"
            assert k == 5
            return [(Document(), 0.2)]

        def merge_from(self, other):
            del other

    class FakeFAISS:
        @classmethod
        def load_local(cls, folder_path, embeddings, allow_dangerous_deserialization):
            del folder_path, embeddings
            assert allow_dangerous_deserialization is True
            return FakeLoadedStore()

    monkeypatch.setattr(rag, "_create_embeddings", lambda *args, **kwargs: object())
    monkeypatch.setattr(rag, "FAISS", FakeFAISS)

    results = rag.query_academic_knowledge(
        "molecular odor", top_k=5, index_dir=config.index_dir
    )

    output = capsys.readouterr().out
    assert "Relevant paper" in output
    assert "https://example.org/relevant" in output
    assert results[0]["rank"] == 1
    assert results[0]["content_type"] == "abstract"


def test_ingest_flushes_every_batch_and_skips_completed_papers_on_resume(tmp_path, monkeypatch):
    config = make_config(tmp_path, batch_size=2)
    rows = [
        {"Source": "Journal", "Title": f"Paper {i}", "Abstract": f"Abstract {i}", "Link": f"https://example.org/{i}"}
        for i in range(5)
    ]
    pd.DataFrame(rows).to_csv(config.csv_path, index=False)

    class Splitter:
        def __init__(self, **kwargs):
            del kwargs

        def split_text(self, text):
            return [text]

    class FakeStore:
        @classmethod
        def from_documents(cls, documents, embedding, ids, normalize_L2):
            del documents, embedding, ids, normalize_L2
            return cls()

        def save_local(self, folder_path):
            folder = Path(folder_path)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "index.faiss").write_bytes(b"faiss")
            (folder / "index.pkl").write_bytes(b"pickle")

    class SessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, traceback):
            return False

    processed = []

    def fake_process(row, config, session):
        del config, session
        paper_id = rag.paper_id_for(row)
        processed.append(paper_id)
        return rag.PaperContent(
            paper_id=paper_id,
            title=row["Title"],
            link=row["Link"],
            source=row["Source"],
            doi="",
            published_date="",
            text=row["Abstract"],
            content_type="abstract",
        )

    monkeypatch.setattr(rag, "_create_embeddings", lambda *args, **kwargs: object())
    monkeypatch.setattr(rag, "RecursiveCharacterTextSplitter", Splitter)
    monkeypatch.setattr(rag, "FAISS", FakeStore)
    monkeypatch.setattr(rag, "Document", rag.SimpleDocument)
    monkeypatch.setattr(rag, "HuggingFaceEmbeddings", object)
    monkeypatch.setattr(rag, "build_http_session", lambda email: SessionContext())
    monkeypatch.setattr(rag, "process_paper", fake_process)

    first = rag.ingest_academic_literature(config)
    assert first["processed"] == 5
    assert first["abstract"] == 5
    assert len(list(config.index_dir.glob("batch_*"))) == 3
    assert len(processed) == 5

    second = rag.ingest_academic_literature(config)
    assert second["processed"] == 0
    assert second["skipped_existing"] == 5
    assert len(processed) == 5
