# -*- coding: utf-8 -*-

import os
import io
import httpx
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from pypdf import PdfReader, PdfWriter

router = APIRouter()

LOCAL_API_PORT = os.getenv("PORT", "8000")
PADDLE_BASE_URL = f"http://127.0.0.1:{LOCAL_API_PORT}"

TASK_STORAGE: dict[str, dict[str, Any]] = {}


def create_searchable_pdf_layer(
    paddle_page_data: dict, original_pdf_bytes: bytes, page_index: int
) -> bytes:
    """
    Translates PaddleOCR text line bounding boxes to PDF coordinate system
    and creates invisible text layer overlay.

    Args:
        paddle_page_data: PaddleOCR response for single page
        original_pdf_bytes: Original PDF file bytes
        page_index: Zero-based page index

    Returns:
        Modified PDF bytes with searchable text layer
    """
    try:
        input_pdf = PdfReader(io.BytesIO(original_pdf_bytes))
        if page_index >= len(input_pdf.pages):
            return original_pdf_bytes

        orig_page = input_pdf.pages[page_index]
        width = float(orig_page.mediabox.width)
        height = float(orig_page.mediabox.height)

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(width, height))
        can.setFillColor(Color(0, 0, 0, alpha=0))

        cells = paddle_page_data.get("cells", [])
        if not cells and "lines" in paddle_page_data:
            cells = paddle_page_data.get("lines", [])

        for cell in cells:
            if isinstance(cell, str):
                continue

            box = cell.get("box", [])
            text = cell.get("text") or cell.get("content") or ""

            if text and len(box) >= 4:
                x_coords = [pt for pt in box if isinstance(pt, (int, float))]

                if len(x_coords) < 4:
                    continue

                x_min = min(x_coords[:2])
                y_max = max(x_coords[1:3]) if len(x_coords) > 2 else max(x_coords)
                y_min = min(x_coords[1:3]) if len(x_coords) > 2 else min(x_coords)

                pdf_y = height - y_max
                font_size = max(8, y_max - y_min)

                can.setFont("Helvetica", font_size)
                can.drawString(x_min, pdf_y, text)

        can.save()
        packet.seek(0)

        mask_pdf = PdfReader(packet)
        orig_page.merge_page(mask_pdf.pages[0])

        output_writer = PdfWriter()
        output_writer.add_page(orig_page)

        output_stream = io.BytesIO()
        output_writer.write(output_stream)
        return output_stream.getvalue()

    except Exception:
        return original_pdf_bytes


@router.post("/documentintelligence/documentModels/prebuilt-layout:analyze")
async def azure_submit_document(request: Request, api_version: str = "2024-11-30"):
    """
    Azure Document Intelligence compatible endpoint for PDF analysis.
    Accepts PDF via multipart form, schedules OCR task, returns 202 with Operation-Location.
    """
    try:
        form = await request.form()
        file = form.get("file")

        if not file or not hasattr(file, "read"):
            return JSONResponse(
                status_code=400, content={"error": "Missing PDF file"}
            )

        file_bytes = await file.read()

        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": ("document.pdf", file_bytes, "application/pdf")}
            response = await client.post(
                f"{PADDLE_BASE_URL}/ocr/tasks", files=files
            )

            if response.status_code != 200:
                return JSONResponse(
                    status_code=500,
                    content={"error": "PaddleOCR task scheduling failed"},
                )

            task_data = response.json()
            task_id = task_data.get("task_id")

        TASK_STORAGE[task_id] = {
            "raw_pdf": file_bytes,
            "azure_json": None,
            "pdf_archive": None,
            "created_at": datetime.now(timezone.utc),
        }

        host_url = str(request.base_url).rstrip("/")
        operation_location = f"{host_url}/documentintelligence/operations/{task_id}?api-version={api_version}"

        return Response(
            status_code=202, headers={"Operation-Location": operation_location}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Internal server error: {str(e)}"}
        )


@router.get("/documentintelligence/operations/{task_id}")
async def azure_get_status(task_id: str, api_version: str = "2024-11-30"):
    """
    Polling endpoint for Azure Document Intelligence operation status.
    Returns running/succeeded/failed status with full result when complete.
    """
    if task_id not in TASK_STORAGE:
        return JSONResponse(status_code=404, content={"error": "Operation not found"})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{PADDLE_BASE_URL}/ocr/tasks/{task_id}"
            )
            if response.status_code != 200:
                return JSONResponse(
                    status_code=404, content={"error": "Task not found"}
                )

            paddle_data = response.json()

        current_status = paddle_data.get("status", "").lower()

        if current_status in ["queued", "processing"]:
            return {"status": "running"}

        if current_status in ["failed"]:
            return {"status": "failed"}

        if not TASK_STORAGE[task_id]["azure_json"]:
            full_text_blocks = []
            azure_pages = []

            original_pdf = TASK_STORAGE[task_id]["raw_pdf"]
            final_pdf_writer = PdfWriter()

            pages = paddle_data.get("pages", [])
            for idx, page in enumerate(pages):
                page_num = idx + 1

                lines = page.get("lines", [])
                page_text_lines = [
                    l if isinstance(l, str) else l.get("text") or l.get("content") or ""
                    for l in lines
                ]
                page_text_lines = [text for text in page_text_lines if text]

                page_full_text = "\n".join(page_text_lines)
                full_text_blocks.append(page_full_text)

                page_merged_bytes = create_searchable_pdf_layer(
                    page, original_pdf, idx
                )
                page_reader = PdfReader(io.BytesIO(page_merged_bytes))
                final_pdf_writer.add_page(page_reader.pages[0])

                azure_pages.append(
                    {
                        "pageNumber": page_num,
                        "angle": 0,
                        "width": page.get("width", 8.5),
                        "height": page.get("height", 11),
                        "unit": "inch",
                        "lines": [{"content": l} for l in page_text_lines],
                    }
                )

            pdf_output_stream = io.BytesIO()
            final_pdf_writer.write(pdf_output_stream)

            TASK_STORAGE[task_id]["pdf_archive"] = pdf_output_stream.getvalue()
            TASK_STORAGE[task_id]["azure_json"] = {
                "status": "succeeded",
                "createdDateTime": TASK_STORAGE[task_id][
                    "created_at"
                ].isoformat(),
                "lastUpdatedDateTime": datetime.now(timezone.utc).isoformat(),
                "analyzeResult": {
                    "apiVersion": api_version,
                    "modelId": "prebuilt-layout",
                    "content": "\n\n".join(full_text_blocks),
                    "pages": azure_pages,
                },
            }

        return TASK_STORAGE[task_id]["azure_json"]

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Internal server error: {str(e)}"}
        )


@router.get("/documentintelligence/operations/{task_id}/pdf")
async def azure_download_pdf(task_id: str):
    """
    Download searchable PDF with embedded text layer.
    """
    if task_id in TASK_STORAGE and TASK_STORAGE[task_id]["pdf_archive"]:
        return Response(
            content=TASK_STORAGE[task_id]["pdf_archive"],
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=archive_{task_id}.pdf"
            },
        )

    return JSONResponse(status_code=404, content={"error": "PDF not ready"})