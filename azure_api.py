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
from config import FORCE_OCR_FOR_AZURE

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
    
    NOTE: Boxes are now in PDF point coordinates (72 DPI), not image pixels.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        input_pdf = PdfReader(io.BytesIO(original_pdf_bytes))
        if page_index >= len(input_pdf.pages):
            return original_pdf_bytes

        orig_page = input_pdf.pages[page_index]
        pdf_width = float(orig_page.mediabox.width)
        pdf_height = float(orig_page.mediabox.height)
        
        logger.info(f"=== PAGE {page_index} DEBUG ===")
        logger.info(f"PDF dimensions: {pdf_width:.2f} x {pdf_height:.2f} points")

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(pdf_width, pdf_height))
        can.setFillColor(Color(0, 0, 0, alpha=0))

        cells = paddle_page_data.get("cells", [])
        if not cells and "lines" in paddle_page_data:
            cells = paddle_page_data.get("lines", [])
        
        logger.info(f"Data source: cells={len(cells)}, has_lines={'lines' in paddle_page_data}, has_ocr_result={'ocr_result' in paddle_page_data}")
        
        # Also try to extract from ocr_result if no cells/lines available
        if not cells:
            ocr_result = paddle_page_data.get("ocr_result", [])
            render_scale = paddle_page_data.get("render_scale", 1.0)
            logger.info(f"Using ocr_result: render_scale={render_scale}, ocr_result_len={len(ocr_result) if isinstance(ocr_result, list) else 0}")
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
                logger.info(f"Created {len(cells)} cells from ocr_result")

        for cell in cells:
            if isinstance(cell, str):
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
                except (ValueError, TypeError):
                    logger.warning(f"Page {page_index}: Invalid box format for text '{text[:30]}'")
                    continue

                # Validate coordinates
                if right < left or bottom < top:
                    logger.warning(f"Page {page_index}: Invalid coordinates for text '{text[:30]}': left={left}, top={top}, right={right}, bottom={bottom}")
                    continue

                # Convert to PDF coordinates (origin at bottom-left, y increases upward)
                # PDF y = pdf_height - image_y
                pdf_x = left
                pdf_y = pdf_height - bottom
                font_size = max(8, bottom - top)

                # Debug: Log first 3 lines of each page
                if cells.index(cell) < 3:
                    logger.info(f"Page {page_index} line {cells.index(cell)+1}: text='{text[:40]}', box=[{left:.2f}, {top:.2f}, {right:.2f}, {bottom:.2f}], pdf_pos=({pdf_x:.2f}, {pdf_y:.2f}), font_size={font_size:.1f}")

                can.setFont("Helvetica", font_size)
                can.drawString(pdf_x, pdf_y, text)

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
            data = {"force_ocr": "true"} if FORCE_OCR_FOR_AZURE else {}
            response = await client.post(
                f"{PADDLE_BASE_URL}/ocr/tasks", files=files, data=data
            )

            if response.status_code not in [200, 202]:
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
                        # Read actual PDF page dimensions from the original PDF
                        "width": page_reader.pages[0].mediabox.width / 72.0,
                        "height": page_reader.pages[0].mediabox.height / 72.0,
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