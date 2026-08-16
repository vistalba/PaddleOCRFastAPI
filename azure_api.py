# -*- coding: utf-8 -*-
"""
Azure Document Intelligence compatibility layer.

This module exposes the subset of the Azure Document Intelligence REST API
that the ``azure-ai-documentintelligence`` Python SDK (and therefore
Paperless-ngx's ``azureai`` remote OCR provider) uses:

1. ``POST /documentintelligence/documentModels/{model_id}:analyze``
   Submit a document (PDF or image) for analysis.
   * SDK format: JSON body ``{"base64Source": "<base64>"}``
   * Legacy format: multipart form with a ``file`` field
   Responds with ``202 Accepted`` and an ``Operation-Location`` header.

2. ``GET /documentintelligence/operations/{task_id}``
   Poll the operation. Responds with ``{"status": "running"}`` while the
   underlying PaddleOCR task is queued or processing, an Azure-style
   ``error`` object when it failed, and the full ``analyzeResult`` when it
   succeeded.

3. ``GET /documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}/pdf``
   Download the searchable archive PDF. ``result_id`` equals the task id and
   matches the last path segment of the ``Operation-Location`` URL, which is
   how the SDK derives it.

4. ``GET /documentintelligence/operations/{task_id}/pdf``
   Legacy archive download path, kept for backward compatibility.

The layer is a thin translator: it forwards the document to the internal
``POST /ocr/tasks`` endpoint, polls ``GET /ocr/tasks/{task_id}``, and maps
the PaddleOCR task response onto the Azure wire format.

Notes:
* The ``model_id`` path segment is accepted but not validated; Paperless-ngx
  sends ``prebuilt-read``. The result echoes the requested model id.
* Requests are not authenticated. The ``Authorization: Bearer`` header sent
  by the SDK is accepted but ignored, which is fine for LAN deployments.
* Operation state is kept in memory (``TASK_STORAGE``) and is lost on
  restart; polling a task from before a restart returns 404.
"""

import base64
import binascii
import io
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from config import FORCE_OCR_FOR_AZURE

logger = logging.getLogger(__name__)

router = APIRouter()

# The internal PaddleOCR API is normally served by this same process. The
# base URL can be overridden (e.g. in unit tests) via PADDLE_BASE_URL.
LOCAL_API_PORT = os.getenv("PORT", "8000")
PADDLE_BASE_URL = os.getenv("PADDLE_BASE_URL", f"http://127.0.0.1:{LOCAL_API_PORT}")

# In-memory operation store, keyed by PaddleOCR task id.
TASK_STORAGE: dict[str, dict[str, Any]] = {}

# Document types recognised by magic bytes, mapped to the file extension and
# MIME type used when forwarding to the internal /ocr/tasks endpoint.
# The Azure SDK sends documents as base64 without a file name, so the type
# must be derived from the content itself.
_DOCUMENT_TYPES: dict[str, dict[str, str]] = {
    "pdf": {"ext": "pdf", "mime": "application/pdf"},
    "png": {"ext": "png", "mime": "image/png"},
    "jpeg": {"ext": "jpg", "mime": "image/jpeg"},
    "tiff": {"ext": "tiff", "mime": "image/tiff"},
    "bmp": {"ext": "bmp", "mime": "image/bmp"},
    "gif": {"ext": "gif", "mime": "image/gif"},
    "webp": {"ext": "webp", "mime": "image/webp"},
}


def _detect_document_type(data: bytes) -> Optional[str]:
    """Detect the document type from magic bytes.

    Args:
        data: Raw document bytes.

    Returns:
        One of the keys of ``_DOCUMENT_TYPES`` (``"pdf"``, ``"png"``,
        ``"jpeg"``, ``"tiff"``, ``"bmp"``, ``"gif"``, ``"webp"``) or ``None``
        if the content is not a recognised PDF/image type.
    """
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data.startswith(b"BM"):
        return "bmp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _azure_error(status_code: int, code: str, message: str) -> JSONResponse:
    """Build an Azure-style error payload: ``{"error": {"code", "message"}}``."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _paddle_client(timeout: float) -> httpx.AsyncClient:
    """Create the HTTP client used for internal PaddleOCR API calls.

    Factored out (instead of calling ``httpx.AsyncClient`` inline) so unit
    tests can monkeypatch it with an in-process ASGI-transport client.
    """
    return httpx.AsyncClient(timeout=timeout)


class _SubmitBodyError(ValueError):
    """Raised when an analyze request body cannot be interpreted."""


async def _read_submit_body(request: Request) -> tuple[bytes, str]:
    """Extract the document bytes from an analyze request.

    Supports both request formats:

    * ``application/json`` with a ``base64Source`` field - the format the
      Azure SDK sends, and
    * ``multipart/form-data`` with a ``file`` field - the legacy/curl format.

    Args:
        request: Incoming FastAPI request.

    Returns:
        ``(document_bytes, document_type)`` where ``document_type`` is one of
        the keys of ``_DOCUMENT_TYPES``.

    Raises:
        _SubmitBodyError: If the body is missing/malformed, or the content is
            not a supported PDF/image type.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception as exc:
            raise _SubmitBodyError("Request body must be valid JSON") from exc
        if not isinstance(payload, dict) or "base64Source" not in payload:
            raise _SubmitBodyError("JSON body must contain a 'base64Source' field")
        try:
            document_bytes = base64.b64decode(payload["base64Source"], validate=True)
        except (binascii.Error, TypeError) as exc:
            raise _SubmitBodyError("'base64Source' is not valid base64") from exc
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise _SubmitBodyError("Missing 'file' form field")
        document_bytes = await upload.read()
    else:
        raise _SubmitBodyError(
            "Content-Type must be application/json or multipart/form-data"
        )

    document_type = _detect_document_type(document_bytes)
    if document_type is None:
        raise _SubmitBodyError(
            "Unsupported document type: expected a PDF or an image "
            "(PNG, JPEG, TIFF, BMP, GIF or WebP)"
        )
    return document_bytes, document_type


def _extract_page_lines(page: dict) -> list[str]:
    """Return the non-empty text lines of a PaddleOCR task page."""
    texts: list[str] = []
    for line in page.get("lines", []):
        if isinstance(line, str):
            text = line
        else:
            text = line.get("text") or line.get("content") or ""
        if text:
            texts.append(text)
    return texts


def _build_pdf_from_images(page_images: list[bytes]) -> bytes:
    """Build a base PDF from page images at 72 DPI (1 pixel == 1 PDF point).

    PaddleOCR reports bounding boxes in image pixels for image tasks. By
    sizing the PDF pages in points equal to the pixel dimensions, the boxes
    map 1:1 onto the PDF coordinate space expected by
    ``create_searchable_pdf_layer``.

    Args:
        page_images: Raw image bytes (one entry per page, in page order).

    Returns:
        PDF bytes with one page per image.
    """
    from PIL import Image

    images = [Image.open(io.BytesIO(raw)) for raw in page_images]
    packet = io.BytesIO()
    images[0].save(
        packet,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=72.0,
    )
    packet.seek(0)
    return packet.getvalue()


async def _fetch_page_image(task_id: str, page_index: int, variant: str) -> bytes:
    """Fetch a page image from the internal PaddleOCR API.

    Falls back to the ``original`` variant when the requested variant
    (e.g. ``corrected``) is not available.

    Args:
        task_id: PaddleOCR task id.
        page_index: Zero-based page index.
        variant: ``"original"`` or ``"corrected"``.

    Returns:
        Raw image bytes.

    Raises:
        RuntimeError: If the image cannot be fetched.
    """
    url = f"{PADDLE_BASE_URL}/ocr/tasks/{task_id}/pages/{page_index}/image"
    async with _paddle_client(30.0) as client:
        response = await client.get(url, params={"variant": variant})
        if response.status_code == 404 and variant != "original":
            response = await client.get(url, params={"variant": "original"})
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch page image {page_index} ({variant}): "
                f"HTTP {response.status_code}"
            )
    return response.content


def _has_text_layer(orig_page) -> bool:
    """
    Check if a PDF page has any text content.

    Args:
        orig_page: pypdf PageObject to check

    Returns:
        True if page contains text operators, False otherwise
    """
    contents = orig_page.get_contents()
    if contents is None:
        return False

    # Check parsed operations instead of raw bytes
    operations = contents.operations
    if not operations:
        return False

    # Look for text operators in the operations list
    text_operators = {b'BT', b'ET', b'Tj', b'TJ', b'T*', b'Td', b'TD', b'Tm', b'Tf'}
    for _, operator in operations:
        if operator in text_operators:
            return True

    return False


def _remove_text_layer_from_page(orig_page) -> None:
    """
    Remove all text content from a PDF page while preserving images and graphics.

    This function modifies the page in-place by filtering out text operators
    from the content stream. The page retains all resources (images, fonts, etc.)
    but will have no text content.

    Uses pypdf's proper content stream parsing instead of regex to avoid
    corrupting the PDF structure.

    Args:
        orig_page: pypdf PageObject to modify

    Returns:
        None (modifies page in-place)
    """
    from pypdf.generic import ContentStream, NameObject, ArrayObject

    contents = orig_page.get_contents()
    if contents is None:
        return

    # Get parsed operations (properly parsed PDF content stream)
    operations = contents.operations
    if not operations:
        return

    # Filter out text operators while preserving graphics operators
    text_operators = {b'BT', b'ET', b'Tj', b'TJ', b'T*', b'Td', b'TD', b'Tm', b'Tf'}
    filtered_operations = [
        (operand, operator)
        for operand, operator in operations
        if operator not in text_operators
    ]

    # Create new content stream with filtered operations
    if filtered_operations:
        new_content = ContentStream(None, None)
        new_content.operations = filtered_operations
        orig_page[NameObject("/Contents")] = new_content
    else:
        # If all operations were text, clear the content stream
        orig_page[NameObject("/Contents")] = ArrayObject([])


def create_searchable_pdf_layer(
    paddle_page_data: dict,
    original_pdf_bytes: bytes,
    page_index: int,
    force_ocr: bool = False
) -> bytes:
    """
    Translates PaddleOCR text line bounding boxes to PDF coordinate system
    and creates invisible text layer overlay.

    NOTE: Boxes are in the coordinate space of ``original_pdf_bytes``:
    PDF points for PDF tasks, pixels for image tasks (the image archive base
    PDF is built at 72 DPI so 1 pixel == 1 point).
    """
    logger.info(f"=== PAGE {page_index} DEBUG ===")
    logger.info(f"Page {page_index}: Starting create_searchable_pdf_layer")

    try:
        input_pdf = PdfReader(io.BytesIO(original_pdf_bytes))
        logger.info(f"Page {page_index}: Loaded original PDF with {len(input_pdf.pages)} pages")

        if page_index >= len(input_pdf.pages):
            logger.warning(f"Page {page_index}: Index out of range, returning original PDF")
            return original_pdf_bytes

        orig_page = input_pdf.pages[page_index]
        pdf_width = float(orig_page.mediabox.width)
        pdf_height = float(orig_page.mediabox.height)
        logger.info(f"Page {page_index}: PDF dimensions: {pdf_width:.2f} x {pdf_height:.2f} points")

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(pdf_width, pdf_height))
        can.setFillColor(Color(0, 0, 0, alpha=0))
        logger.info(f"Page {page_index}: Created canvas for text layer")

        skipped_count = 0
        drawn_count = 0

        cells = paddle_page_data.get("cells", [])
        if not cells and "lines" in paddle_page_data:
            cells = paddle_page_data.get("lines", [])

        logger.info(f"Page {page_index}: Data source: cells={len(cells)}, has_lines={'lines' in paddle_page_data}, has_ocr_result={'ocr_result' in paddle_page_data}")

        # Also try to extract from ocr_result if no cells/lines available
        if not cells:
            ocr_result = paddle_page_data.get("ocr_result", [])
            render_scale = paddle_page_data.get("render_scale", 1.0)
            logger.info(f"Page {page_index}: Using ocr_result: render_scale={render_scale}, ocr_result_len={len(ocr_result) if isinstance(ocr_result, list) else 0}")
            if isinstance(ocr_result, list) and render_scale > 1.0:
                # Transform boxes to PDF coordinates
                for item in ocr_result:
                    if isinstance(item, dict):
                        texts = item.get("rec_texts", [])
                        boxes = item.get("rec_boxes", [])
                        if isinstance(texts, list) and isinstance(boxes, list):
                            for i, text in enumerate(texts):
                                if text and i < len(boxes):
                                    # Transform from image pixels to PDF points
                                    transformed_box = [coord / render_scale for coord in boxes[i]]
                                    cells.append({
                                        "text": text if isinstance(text, str) else str(text),
                                        "box": transformed_box
                                    })
                logger.info(f"Page {page_index}: Created {len(cells)} cells from ocr_result")

        logger.info(f"Page {page_index}: Processing {len(cells)} text cells")
        drawn_count = 0
        skipped_count = 0

        for cell in cells:
            if isinstance(cell, str):
                skipped_count += 1
                continue

            box = cell.get("box", [])
            text = cell.get("text") or cell.get("content") or ""

            if text and len(box) >= 4:
                # Box format: [left, top, right, bottom]
                # These coordinates are in PDF points, but y-axis is still image-based
                # (origin at top-left, y increases downward)
                try:
                    left = float(box[0])
                    top = float(box[1])      # Top edge (smaller y in image coords)
                    right = float(box[2])
                    bottom = float(box[3])   # Bottom edge (larger y in image coords)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Page {page_index}: Invalid box format for text '{text[:30]}': {e}")
                    skipped_count += 1
                    continue

                # Validate coordinates
                if right < left or bottom < top:
                    logger.warning(f"Page {page_index}: Invalid coordinates for text '{text[:30]}': left={left}, top={top}, right={right}, bottom={bottom}")
                    skipped_count += 1
                    continue

                # Convert to PDF coordinates (origin at bottom-left, y increases upward)
                # PDF y = pdf_height - image_y
                pdf_x = left
                pdf_y = pdf_height - bottom
                font_size = max(8, bottom - top)

                # Debug: Log first 3 lines of each page
                cell_index = cells.index(cell)
                if cell_index < 3:
                    logger.info(f"Page {page_index} line {cell_index+1}: text='{text[:40]}', box=[{left:.2f}, {top:.2f}, {right:.2f}, {bottom:.2f}], pdf_pos=({pdf_x:.2f}, {pdf_y:.2f}), font_size={font_size:.1f}")

                try:
                    can.setFont("Helvetica", font_size)
                    can.drawString(pdf_x, pdf_y, text)
                    drawn_count += 1
                    if cell_index < 3:
                        logger.info(f"Page {page_index} line {cell_index+1}: Successfully drew text at ({pdf_x:.2f}, {pdf_y:.2f})")
                except Exception as e:
                    logger.error(f"Page {page_index}: drawString failed for '{text[:50]}': {e}")
                    skipped_count += 1
            else:
                skipped_count += 1

        logger.info(f"Page {page_index}: Processing complete - drew {drawn_count} cells, skipped {skipped_count}")

        can.save()
        packet.seek(0)
        logger.info(f"Page {page_index}: Saved canvas to packet")

        mask_pdf = PdfReader(packet)
        logger.info(f"Page {page_index}: Loaded mask PDF with {len(mask_pdf.pages)} pages")

        # Add the page to the writer BEFORE mutating it: pypdf only allows
        # safe content-stream modifications on pages attached to a writer
        # (mutating reader pages is deprecated and unreliable). add_page()
        # returns the writer-attached copy of the page.
        output_writer = PdfWriter()
        writer_page = output_writer.add_page(orig_page)
        logger.info(f"Page {page_index}: Added page to output writer")

        # Remove existing text layer if FORCE_OCR is enabled
        if force_ocr:
            logger.info(f"Page {page_index}: FORCE_OCR enabled, checking for existing text layer")
            if _has_text_layer(writer_page):
                logger.info(f"Page {page_index}: Existing text layer detected, removing it")
                _remove_text_layer_from_page(writer_page)
                logger.info(f"Page {page_index}: Existing text layer removed")
            else:
                logger.info(f"Page {page_index}: No existing text layer found")

        writer_page.merge_page(mask_pdf.pages[0])
        logger.info(f"Page {page_index}: Merged text layer with original page")

        output_stream = io.BytesIO()
        output_writer.write(output_stream)
        logger.info(f"Page {page_index}: Successfully created searchable PDF layer")
        return output_stream.getvalue()

    except Exception as e:
        logger.error(f"Page {page_index}: Exception in create_searchable_pdf_layer: {e}", exc_info=True)
        return original_pdf_bytes


async def _build_azure_result(task_id: str, entry: dict, paddle_data: dict) -> tuple[bytes, dict]:
    """Build the Azure-style result JSON and the searchable archive PDF.

    For PDF tasks the archive is the original PDF with an invisible OCR text
    layer merged in. For image tasks the archive is rebuilt from the page
    images (1 pixel == 1 point) with the text layer overlaid on top.

    Args:
        task_id: PaddleOCR task id (used to fetch page images for image
            tasks).
        entry: ``TASK_STORAGE`` entry for the task.
        paddle_data: Parsed ``GET /ocr/tasks/{task_id}`` response.

    Returns:
        ``(archive_pdf_bytes, azure_json)``
    """
    document_type = entry["document_type"]
    force_ocr = bool(entry.get("force_ocr", False))
    pages = paddle_data.get("pages", [])

    full_text_blocks: list[str] = []
    azure_pages: list[dict] = []
    archive_writer = PdfWriter()

    if document_type == "pdf":
        original_bytes = entry["document_bytes"]
        for index, page in enumerate(pages):
            page_lines = _extract_page_lines(page)
            full_text_blocks.append("\n".join(page_lines))

            merged_bytes = create_searchable_pdf_layer(
                page, original_bytes, index, force_ocr=force_ocr
            )
            merged_page = PdfReader(io.BytesIO(merged_bytes)).pages[0]
            archive_writer.add_page(merged_page)

            azure_pages.append(
                {
                    "pageNumber": index + 1,
                    "angle": 0,
                    # Azure reports PDF page dimensions in inches
                    "width": float(merged_page.mediabox.width) / 72.0,
                    "height": float(merged_page.mediabox.height) / 72.0,
                    "unit": "inch",
                    "lines": [{"content": line} for line in page_lines],
                }
            )
    else:
        # Image task: fetch the page images PaddleOCR actually processed and
        # rebuild a base PDF from them (1px == 1pt) so the OCR boxes, which
        # are in pixels, map directly onto the PDF coordinate space.
        page_images: list[bytes] = []
        for index, page in enumerate(pages):
            variant = page.get("ocr_image_variant") or "original"
            page_images.append(await _fetch_page_image(task_id, index, variant))
        base_pdf = _build_pdf_from_images(page_images)

        for index, page in enumerate(pages):
            page_lines = _extract_page_lines(page)
            full_text_blocks.append("\n".join(page_lines))

            merged_bytes = create_searchable_pdf_layer(
                page, base_pdf, index, force_ocr=force_ocr
            )
            merged_page = PdfReader(io.BytesIO(merged_bytes)).pages[0]
            archive_writer.add_page(merged_page)

            width = page.get("width") or float(merged_page.mediabox.width)
            height = page.get("height") or float(merged_page.mediabox.height)
            azure_pages.append(
                {
                    "pageNumber": index + 1,
                    "angle": 0,
                    # Azure reports image page dimensions in pixels
                    "width": float(width),
                    "height": float(height),
                    "unit": "pixel",
                    "lines": [{"content": line} for line in page_lines],
                }
            )

    archive_stream = io.BytesIO()
    archive_writer.write(archive_stream)

    azure_json = {
        "status": "succeeded",
        "createdDateTime": entry["created_at"].isoformat(),
        "lastUpdatedDateTime": datetime.now(timezone.utc).isoformat(),
        "analyzeResult": {
            "apiVersion": entry.get("api_version", "2024-11-30"),
            "modelId": entry.get("model_id", "prebuilt-read"),
            "stringIndexType": "textElements",
            "content": "\n\n".join(full_text_blocks),
            "pages": azure_pages,
        },
    }
    return archive_stream.getvalue(), azure_json


@router.post("/documentintelligence/documentModels/{model_id}:analyze")
async def azure_submit_document(
    request: Request, model_id: str, api_version: str = "2024-11-30"
):
    """
    Submit a PDF or image for OCR analysis (Azure Document Intelligence format).

    Accepts the JSON ``base64Source`` body sent by the
    ``azure-ai-documentintelligence`` SDK as well as the legacy multipart
    ``file`` upload. Responds with ``202 Accepted`` and an
    ``Operation-Location`` header pointing at the polling endpoint.
    """
    try:
        document_bytes, document_type = await _read_submit_body(request)
    except _SubmitBodyError as exc:
        return _azure_error(400, "BadRequest", str(exc))

    spec = _DOCUMENT_TYPES[document_type]
    filename = f"document.{spec['ext']}"

    try:
        async with _paddle_client(30.0) as client:
            response = await client.post(
                f"{PADDLE_BASE_URL}/ocr/tasks",
                files={"file": (filename, document_bytes, spec["mime"])},
                data={"force_ocr": "true"} if FORCE_OCR_FOR_AZURE else {},
            )
    except httpx.HTTPError as exc:
        logger.error("PaddleOCR task submission failed: %s", exc)
        return _azure_error(
            502, "DocumentAnalysisFailed", "Failed to reach the internal OCR service"
        )

    if response.status_code not in (200, 202):
        # Surface upstream errors (413 too large, 415 unsupported type, 429
        # rate limited, ...) with an Azure-style error body instead of 500.
        logger.warning(
            "PaddleOCR task submission rejected: HTTP %s %s",
            response.status_code,
            response.text[:200],
        )
        return _azure_error(
            response.status_code,
            "DocumentAnalysisFailed",
            f"OCR task scheduling failed (HTTP {response.status_code})",
        )

    try:
        task_data = response.json()
    except ValueError as exc:
        logger.error("PaddleOCR task submission returned non-JSON body: %s", exc)
        return _azure_error(
            502, "DocumentAnalysisFailed", "OCR service returned an invalid response"
        )

    task_id = task_data.get("task_id")
    if not task_id:
        return _azure_error(
            502, "DocumentAnalysisFailed", "OCR service did not return a task id"
        )

    TASK_STORAGE[task_id] = {
        "document_bytes": document_bytes,
        "document_type": document_type,
        "model_id": model_id,
        "api_version": api_version,
        "azure_json": None,
        "pdf_archive": None,
        "created_at": datetime.now(timezone.utc),
        "force_ocr": FORCE_OCR_FOR_AZURE,
    }

    host_url = str(request.base_url).rstrip("/")
    operation_location = (
        f"{host_url}/documentintelligence/operations/{task_id}"
        f"?api-version={api_version}"
    )

    return Response(
        status_code=202, headers={"Operation-Location": operation_location}
    )


@router.get("/documentintelligence/operations/{task_id}")
async def azure_get_status(task_id: str, api_version: str = "2024-11-30"):
    """
    Poll an Azure-style operation.

    Returns ``{"status": "running"}`` while the PaddleOCR task is queued or
    processing, an Azure-style ``error`` object when it failed, and the full
    ``analyzeResult`` when it succeeded.
    """
    entry = TASK_STORAGE.get(task_id)
    if entry is None:
        return _azure_error(404, "OperationNotFound", f"Operation {task_id} not found")

    try:
        async with _paddle_client(10.0) as client:
            response = await client.get(f"{PADDLE_BASE_URL}/ocr/tasks/{task_id}")
    except httpx.HTTPError as exc:
        logger.error("Failed to poll PaddleOCR task %s: %s", task_id, exc)
        return _azure_error(
            502, "DocumentAnalysisFailed", "Failed to reach the internal OCR service"
        )

    if response.status_code != 200:
        return _azure_error(404, "OperationNotFound", "Task not found")

    try:
        paddle_data = response.json()
    except ValueError as exc:
        logger.error("PaddleOCR task %s returned non-JSON body: %s", task_id, exc)
        return _azure_error(
            502, "DocumentAnalysisFailed", "OCR service returned an invalid response"
        )

    status = str(paddle_data.get("status", "")).lower()

    if status in ("queued", "processing", "retry_available"):
        return {"status": "running"}

    if status == "failed":
        return {
            "status": "failed",
            "error": {
                "code": "DocumentAnalysisFailed",
                "message": paddle_data.get("error_msg") or "OCR task failed",
            },
        }

    # Status "done": build (and cache) the Azure result on first poll.
    if entry["azure_json"] is None:
        try:
            archive_bytes, azure_json = await _build_azure_result(
                task_id, entry, paddle_data
            )
        except Exception as exc:
            logger.error("Failed to build Azure result for task %s: %s", task_id, exc, exc_info=True)
            return _azure_error(
                500, "DocumentAnalysisFailed", f"Failed to build analysis result: {exc}"
            )
        entry["pdf_archive"] = archive_bytes
        entry["azure_json"] = azure_json

    return entry["azure_json"]


def _serve_pdf_archive(task_id: str):
    """Serve the cached searchable archive PDF for a task (or 404)."""
    entry = TASK_STORAGE.get(task_id)
    if entry and entry.get("pdf_archive"):
        return Response(
            content=entry["pdf_archive"],
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=archive_{task_id}.pdf"
            },
        )
    return _azure_error(404, "ArchiveNotReady", "PDF archive not ready")


@router.get(
    "/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}/pdf"
)
async def azure_download_pdf_analyze_results(
    model_id: str, result_id: str, api_version: str = "2024-11-30"
):
    """
    Download the searchable archive PDF (Azure SDK path).

    ``result_id`` equals the task id; the SDK derives it from the last path
    segment of the ``Operation-Location`` URL.
    """
    return _serve_pdf_archive(result_id)


@router.get("/documentintelligence/operations/{task_id}/pdf")
async def azure_download_pdf(task_id: str):
    """
    Download the searchable archive PDF with embedded text layer.

    Legacy path, kept for backward compatibility with earlier clients.
    """
    return _serve_pdf_archive(task_id)
