# -*- coding: utf-8 -*-
"""
Unit tests for the Azure Document Intelligence compatibility layer.

The internal PaddleOCR API (``POST /ocr/tasks``, ``GET /ocr/tasks/{id}`` and
``GET /ocr/tasks/{id}/pages/{i}/image``) is stubbed with an in-process ASGI
app, so the tests exercise the full Azure wire contract (submit -> poll ->
download archive PDF) without needing the OCR models or a running server.

Run from the project root:

    uv run pytest tests/test_azure_api.py
"""

import base64
import importlib.util
import io

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from pypdf import PdfReader
from reportlab.pdfgen import canvas

import azure_api
from azure_api import router as azure_router

TASK_ID = "task-123"
API_VERSION = "2024-11-30"
# The exact path the azure-ai-documentintelligence SDK submits to
# (model_id="prebuilt-read" is what Paperless-ngx sends).
ANALYZE_PATH = (
    f"/documentintelligence/documentModels/prebuilt-read:analyze"
    f"?api-version={API_VERSION}"
)
LEGACY_ANALYZE_PATH = (
    f"/documentintelligence/documentModels/prebuilt-layout:analyze"
    f"?api-version={API_VERSION}"
)


def _make_pdf() -> bytes:
    """Build a minimal one-page PDF (612 x 792 pt) with a native text layer."""
    buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=(612, 792))
    pdf_canvas.drawString(72, 720, "Original document text")
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()


def _make_png(width: int = 800, height: int = 600) -> bytes:
    """Build a minimal PNG image of the given pixel size."""
    from PIL import Image

    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pdf_page(lines: list[dict]) -> dict:
    """Canned PaddleOCR task page for a PDF task (boxes in PDF points)."""
    return {
        "page_index": 0,
        "status": "done",
        "processing_method": "ocr",
        "width": 1700,
        "height": 2200,
        "image_variants": {"original": None, "corrected": None},
        "default_image_variant": "original",
        "ocr_image_variant": "original",
        "native_text": None,
        "ocr_result": None,
        "lines": lines,
        "rule_result": None,
        "ai_result": None,
        "ai_status": "not_started",
        "ai_model_name": None,
        "ai_processed_at": None,
        "ai_error": None,
        "error_msg": None,
        "pdf_width_pts": 612.0,
        "pdf_height_pts": 792.0,
        "render_scale": 2.0,
    }


def _image_page(lines: list[dict], width: int = 800, height: int = 600) -> dict:
    """Canned PaddleOCR task page for an image task (boxes in pixels)."""
    page = _pdf_page(lines)
    page.update(
        {
            "width": width,
            "height": height,
            "pdf_width_pts": None,
            "pdf_height_pts": None,
            "render_scale": 1.0,
        }
    )
    return page


def _task_payload(
    status: str, pages: list[dict], file_type: str, error_msg: str | None = None
) -> dict:
    """Canned ``GET /ocr/tasks/{task_id}`` response body."""
    return {
        "task_id": TASK_ID,
        "ip": "127.0.0.1",
        "created_at": "2026-08-15T10:00:00",
        "status": status,
        "original_filename": f"document.{'pdf' if file_type == 'pdf' else 'png'}",
        "file_type": file_type,
        "use_doc_preprocessor": False,
        "page_count": len(pages),
        "completed_pages": len(pages),
        "failed_pages": 0,
        "pages": pages,
        "image_variants": {"original": None, "corrected": None},
        "default_image_variant": "original",
        "ocr_image_variant": "original",
        "ocr_result": None,
        "error_msg": error_msg,
        "queue_position": None,
        "retry_available_at": None,
        "retry_after_seconds": 0,
        "ai_organizer_available": False,
        "ai_organizer_model": None,
        "ai_organizer_unavailable_reason": None,
        "ai_repeat_allowed": False,
    }


@pytest.fixture()
def paddle_stub():
    """In-process stub of the internal PaddleOCR API.

    Returns ``(state, app)``: ``state`` lets tests configure the task status
    sequence and page images and inspect what was submitted; ``app`` is the
    ASGI application serving the stubbed endpoints.
    """
    state = {
        "submissions": [],
        "statuses": [],
        "page_images": {},
    }

    app = FastAPI()

    @app.post("/ocr/tasks")
    async def create_task(request: Request):
        form = await request.form()
        upload = form["file"]
        state["submissions"].append(
            {
                "filename": upload.filename,
                "bytes": await upload.read(),
                "force_ocr": form.get("force_ocr"),
            }
        )
        return {"task_id": TASK_ID, "status": "queued"}

    @app.get("/ocr/tasks/{task_id}")
    async def get_task(task_id: str):
        if not state["statuses"]:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        # Replay the configured sequence; the last entry is sticky.
        if len(state["statuses"]) > 1:
            return state["statuses"].pop(0)
        return state["statuses"][0]

    @app.get("/ocr/tasks/{task_id}/pages/{page_index}/image")
    async def get_page_image(task_id: str, page_index: int, variant: str = "original"):
        content = state["page_images"].get((page_index, variant))
        if content is None:
            return JSONResponse(status_code=404, content={"detail": "image not found"})
        return Response(content=content, media_type="image/png")

    return state, app


@pytest.fixture()
def azure_client(paddle_stub, monkeypatch):
    """TestClient for the Azure layer with the PaddleOCR backend stubbed."""
    state, stub_app = paddle_stub

    app = FastAPI()
    app.include_router(azure_router)

    def fake_paddle_client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_app))

    monkeypatch.setattr(azure_api, "_paddle_client", fake_paddle_client)
    monkeypatch.setattr(azure_api, "TASK_STORAGE", {})

    return TestClient(app), state


class TestSubmit:
    def test_submit_pdf_json_base64(self, azure_client):
        """The SDK format: JSON body with base64Source for a PDF."""
        client, state = azure_client
        pdf_bytes = _make_pdf()

        response = client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(pdf_bytes).decode("ascii")},
        )

        assert response.status_code == 202
        assert response.headers["Operation-Location"] == (
            f"http://testserver/documentintelligence/operations/{TASK_ID}"
            f"?api-version={API_VERSION}"
        )
        submission = state["submissions"][0]
        assert submission["filename"] == "document.pdf"
        assert submission["bytes"] == pdf_bytes

    def test_submit_image_json_base64(self, azure_client):
        """The SDK format: JSON body with base64Source for an image."""
        client, state = azure_client
        png_bytes = _make_png()

        response = client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(png_bytes).decode("ascii")},
        )

        assert response.status_code == 202
        submission = state["submissions"][0]
        assert submission["filename"] == "document.png"
        assert submission["bytes"] == png_bytes

    def test_submit_multipart_legacy_model_id(self, azure_client):
        """Legacy multipart upload still works, including prebuilt-layout."""
        client, state = azure_client
        pdf_bytes = _make_pdf()

        response = client.post(
            LEGACY_ANALYZE_PATH,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 202
        submission = state["submissions"][0]
        assert submission["filename"] == "document.pdf"
        assert submission["bytes"] == pdf_bytes

    def test_submit_bad_base64_returns_400(self, azure_client):
        client, _ = azure_client
        response = client.post(ANALYZE_PATH, json={"base64Source": "!!not-base64!!"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_submit_missing_base64_source_returns_400(self, azure_client):
        client, _ = azure_client
        response = client.post(
            ANALYZE_PATH, json={"urlSource": "https://example.com/doc.pdf"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"

    def test_submit_unsupported_binary_returns_400(self, azure_client):
        client, _ = azure_client
        response = client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(b"plain text").decode("ascii")},
        )
        assert response.status_code == 400
        assert "Unsupported document type" in response.json()["error"]["message"]

    def test_submit_wrong_content_type_returns_400(self, azure_client):
        client, _ = azure_client
        response = client.post(
            ANALYZE_PATH, content=b"raw", headers={"content-type": "text/plain"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BadRequest"


class TestPoll:
    def test_poll_running_then_succeeded_pdf(self, azure_client):
        client, state = azure_client
        lines = [
            {"text": "Hello World", "box": [72.0, 72.0, 200.0, 90.0]},
            {"text": "Second Line", "box": [72.0, 120.0, 250.0, 140.0]},
        ]
        state["statuses"] = [
            _task_payload("processing", [_pdf_page(lines)], "pdf"),
            _task_payload("done", [_pdf_page(lines)], "pdf"),
        ]
        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(_make_pdf()).decode("ascii")},
        )

        assert client.get(f"/documentintelligence/operations/{TASK_ID}").json() == {
            "status": "running"
        }

        result = client.get(f"/documentintelligence/operations/{TASK_ID}").json()
        assert result["status"] == "succeeded"
        analyze = result["analyzeResult"]
        assert analyze["modelId"] == "prebuilt-read"
        assert analyze["stringIndexType"] == "textElements"
        assert analyze["content"] == "Hello World\nSecond Line"
        page = analyze["pages"][0]
        assert page["pageNumber"] == 1
        assert page["unit"] == "inch"
        assert page["width"] == pytest.approx(612.0 / 72.0)
        assert page["height"] == pytest.approx(792.0 / 72.0)
        assert page["lines"] == [
            {"content": "Hello World"},
            {"content": "Second Line"},
        ]

    def test_poll_failed_returns_error_object(self, azure_client):
        client, state = azure_client
        state["statuses"] = [
            _task_payload("failed", [], "pdf", error_msg="OCR exploded")
        ]
        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(_make_pdf()).decode("ascii")},
        )

        result = client.get(f"/documentintelligence/operations/{TASK_ID}").json()
        assert result["status"] == "failed"
        assert result["error"]["code"] == "DocumentAnalysisFailed"
        assert result["error"]["message"] == "OCR exploded"

    def test_poll_unknown_operation_returns_404(self, azure_client):
        client, _ = azure_client
        response = client.get("/documentintelligence/operations/missing")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "OperationNotFound"

    def test_image_flow_succeeded_with_pixel_units(self, azure_client):
        client, state = azure_client
        png_bytes = _make_png(800, 600)
        lines = [{"text": "Image Text", "box": [50.0, 40.0, 300.0, 80.0]}]
        state["statuses"] = [_task_payload("done", [_image_page(lines)], "image")]
        state["page_images"] = {(0, "original"): png_bytes}

        response = client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(png_bytes).decode("ascii")},
        )
        assert response.status_code == 202
        assert state["submissions"][0]["filename"] == "document.png"

        result = client.get(f"/documentintelligence/operations/{TASK_ID}").json()
        assert result["status"] == "succeeded"
        page = result["analyzeResult"]["pages"][0]
        assert page["unit"] == "pixel"
        assert page["width"] == 800
        assert page["height"] == 600
        assert page["lines"] == [{"content": "Image Text"}]

    def test_image_flow_uses_corrected_variant(self, azure_client):
        """When OCR ran on the corrected image, the corrected image is fetched."""
        client, state = azure_client
        png_bytes = _make_png(400, 300)
        lines = [{"text": "Corrected", "box": [10.0, 10.0, 100.0, 30.0]}]
        page = _image_page(lines, 400, 300)
        page["ocr_image_variant"] = "corrected"
        state["statuses"] = [_task_payload("done", [page], "image")]
        state["page_images"] = {(0, "corrected"): png_bytes}

        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(png_bytes).decode("ascii")},
        )

        result = client.get(f"/documentintelligence/operations/{TASK_ID}").json()
        assert result["status"] == "succeeded"


class TestForceOcr:
    def test_force_ocr_replaces_native_text_layer(self, azure_client, monkeypatch):
        """FORCE_OCR_FOR_AZURE=true removes the native text layer and replaces it."""
        monkeypatch.setattr(azure_api, "FORCE_OCR_FOR_AZURE", True)
        client, state = azure_client
        lines = [{"text": "Hello World", "box": [72.0, 72.0, 200.0, 90.0]}]
        state["statuses"] = [_task_payload("done", [_pdf_page(lines)], "pdf")]

        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(_make_pdf()).decode("ascii")},
        )
        # The force_ocr form field is forwarded to the internal task API.
        assert state["submissions"][0]["force_ocr"] == "true"

        client.get(f"/documentintelligence/operations/{TASK_ID}")
        response = client.get(
            f"/documentintelligence/documentModels/prebuilt-read"
            f"/analyzeResults/{TASK_ID}/pdf"
        )
        assert response.status_code == 200
        text = PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
        assert "Hello World" in text
        assert "Original document text" not in text


class TestArchivePdf:
    def test_pdf_archive_via_analyze_results_route(self, azure_client):
        """New Azure SDK path: .../analyzeResults/{result_id}/pdf."""
        client, state = azure_client
        lines = [{"text": "Hello World", "box": [72.0, 72.0, 200.0, 90.0]}]
        state["statuses"] = [_task_payload("done", [_pdf_page(lines)], "pdf")]
        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(_make_pdf()).decode("ascii")},
        )
        client.get(f"/documentintelligence/operations/{TASK_ID}")

        response = client.get(
            f"/documentintelligence/documentModels/prebuilt-read"
            f"/analyzeResults/{TASK_ID}/pdf"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        reader = PdfReader(io.BytesIO(response.content))
        assert len(reader.pages) == 1
        assert float(reader.pages[0].mediabox.width) == pytest.approx(612.0)
        assert float(reader.pages[0].mediabox.height) == pytest.approx(792.0)
        # The invisible text layer is embedded and extractable.
        assert "Hello World" in reader.pages[0].extract_text()

    def test_pdf_archive_via_legacy_route(self, azure_client):
        """Legacy path: /documentintelligence/operations/{task_id}/pdf."""
        client, state = azure_client
        lines = [{"text": "Hello World", "box": [72.0, 72.0, 200.0, 90.0]}]
        state["statuses"] = [_task_payload("done", [_pdf_page(lines)], "pdf")]
        client.post(
            LEGACY_ANALYZE_PATH,
            files={
                "file": ("document.pdf", _make_pdf(), "application/pdf")
            },
        )
        client.get(f"/documentintelligence/operations/{TASK_ID}")

        response = client.get(f"/documentintelligence/operations/{TASK_ID}/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        reader = PdfReader(io.BytesIO(response.content))
        assert "Hello World" in reader.pages[0].extract_text()

    def test_image_archive_pdf_dimensions(self, azure_client):
        """Image archive: page size in points equals the pixel dimensions."""
        client, state = azure_client
        png_bytes = _make_png(800, 600)
        lines = [{"text": "Image Text", "box": [50.0, 40.0, 300.0, 80.0]}]
        state["statuses"] = [_task_payload("done", [_image_page(lines)], "image")]
        state["page_images"] = {(0, "original"): png_bytes}
        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(png_bytes).decode("ascii")},
        )
        client.get(f"/documentintelligence/operations/{TASK_ID}")

        response = client.get(
            f"/documentintelligence/documentModels/prebuilt-read"
            f"/analyzeResults/{TASK_ID}/pdf"
        )
        assert response.status_code == 200
        reader = PdfReader(io.BytesIO(response.content))
        assert float(reader.pages[0].mediabox.width) == pytest.approx(800.0)
        assert float(reader.pages[0].mediabox.height) == pytest.approx(600.0)
        assert "Image Text" in reader.pages[0].extract_text()

    def test_archive_not_ready_returns_404(self, azure_client):
        client, state = azure_client
        state["statuses"] = [_task_payload("processing", [_pdf_page([])], "pdf")]
        client.post(
            ANALYZE_PATH,
            json={"base64Source": base64.b64encode(_make_pdf()).decode("ascii")},
        )

        response = client.get(
            f"/documentintelligence/documentModels/prebuilt-read"
            f"/analyzeResults/{TASK_ID}/pdf"
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "ArchiveNotReady"


def _serve_app(app):
    """Serve an ASGI app on an ephemeral localhost port.

    Returns:
        ``(uvicorn.Server, base_url)`` - stop the server by setting
        ``server.should_exit = True``.
    """
    import threading
    import time

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


_SDK_AVAILABLE = importlib.util.find_spec("azure.ai.documentintelligence") is not None


@pytest.mark.skipif(
    not _SDK_AVAILABLE,
    reason="azure-ai-documentintelligence not installed",
)
class TestRealSdk:
    """End-to-end test with the real azure-ai-documentintelligence SDK.

    Replays the exact call sequence of Paperless-ngx's azureai provider
    (``src/paperless/parsers/remote.py``, ``_azure_ai_vision_parse``) against
    the live app served over real HTTP: submit -> ``poller.wait()`` ->
    ``poller.details["operation_id"]`` -> ``poller.result()`` ->
    ``get_analyze_result_pdf``.
    """

    def test_paperless_ngx_flow(self, paddle_stub, monkeypatch):
        state, stub_app = paddle_stub
        lines = [{"text": "Hello World", "box": [72.0, 72.0, 200.0, 90.0]}]
        state["statuses"] = [_task_payload("done", [_pdf_page(lines)], "pdf")]

        app = FastAPI()
        app.include_router(azure_router)

        def fake_paddle_client(timeout: float) -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_app))

        monkeypatch.setattr(azure_api, "_paddle_client", fake_paddle_client)
        monkeypatch.setattr(azure_api, "TASK_STORAGE", {})

        server, endpoint = _serve_app(app)
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.ai.documentintelligence.models import (
                AnalyzeDocumentRequest,
                AnalyzeOutputOption,
                DocumentContentFormat,
            )
            from azure.core.credentials import AzureKeyCredential

            client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential("unused"),
            )
            try:
                poller = client.begin_analyze_document(
                    model_id="prebuilt-read",
                    body=AnalyzeDocumentRequest(bytes_source=_make_pdf()),
                    output_content_format=DocumentContentFormat.TEXT,
                    output=[AnalyzeOutputOption.PDF],
                    content_type="application/json",
                )
                poller.wait()

                # The SDK parses the last path segment of the
                # Operation-Location URL - this API's task id.
                result_id = poller.details["operation_id"]
                assert result_id == TASK_ID

                result = poller.result()
                assert result.content == "Hello World"

                archive = b"".join(
                    client.get_analyze_result_pdf(
                        model_id="prebuilt-read", result_id=result_id
                    )
                )
                assert archive.startswith(b"%PDF-")
                reader = PdfReader(io.BytesIO(archive))
                assert "Hello World" in reader.pages[0].extract_text()
            finally:
                client.close()
        finally:
            server.should_exit = True
