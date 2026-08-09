# -*- coding: utf-8 -*-
"""Prepare a single uploaded image or PDF as one or more displayable pages."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium

from config import (
    MAX_MULTI_IMAGE_PAGES,
    MAX_PDF_PAGES,
    PDF_MAX_RENDER_PIXELS,
    PDF_NATIVE_TEXT_MIN_CHARS,
    OCR_DPI,
)


class DocumentPreparationError(RuntimeError):
    """Raised when an uploaded document cannot be prepared safely."""


def normalize_pdf_text(value: str) -> str:
    """Normalize PDF text while preserving useful paragraph boundaries."""
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    value = "".join(
        char
        for char in value
        if char in {"\n", "\t"} or ord(char) >= 32
    )

    lines = []
    blank_pending = False
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line:
            if blank_pending and lines:
                lines.append("")
            lines.append(line)
            blank_pending = False
        elif lines:
            blank_pending = True
    return "\n".join(lines).strip()


def meaningful_text_char_count(value: str) -> int:
    return sum(1 for char in value if char.isalnum())


def should_use_native_pdf_text(
    value: str,
    minimum_chars: int = PDF_NATIVE_TEXT_MIN_CHARS,
) -> bool:
    return meaningful_text_char_count(value) >= max(1, minimum_chars)


def _calculate_scale_from_dpi(pdf_width_pts: float, pdf_height_pts: float) -> float:
    """Calculate render scale based on OCR_DPI and PDF dimensions.
    
    PDF native DPI is 72 points per inch.
    Scale = OCR_DPI / 72 to achieve target DPI.
    """
    scale = OCR_DPI / 72.0
    
    # Check if this would exceed max pixels
    estimated_pixels = pdf_width_pts * pdf_height_pts * scale * scale
    if estimated_pixels <= PDF_MAX_RENDER_PIXELS:
        return scale
    
    # Fallback: reduce scale to fit
    max_scale = math.sqrt(PDF_MAX_RENDER_PIXELS / (pdf_width_pts * pdf_height_pts))
    if max_scale < 0.25:
        raise DocumentPreparationError("PDF page too large to render at minimum quality")
    return max_scale


def _render_pdf_page(
    page: Any, 
    output_path: Path,
    pdf_width_pts: float,
    pdf_height_pts: float
) -> tuple[int, int, float]:
    """Render PDF page to image at DPI-based scale.
    
    Returns: (image_width, image_height, actual_scale)
    """
    scale = _calculate_scale_from_dpi(pdf_width_pts, pdf_height_pts)
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil().convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return image.size + (scale,)
    finally:
        bitmap.close()


def _prepare_pdf(source_path: Path, task_dir: Path, force_ocr: bool = False) -> list[dict[str, Any]]:
    try:
        document = pdfium.PdfDocument(str(source_path))
    except pdfium.PdfiumError as exc:
        if exc.err_code in {
            pdfium.raw.FPDF_ERR_PASSWORD,
            pdfium.raw.FPDF_ERR_SECURITY,
        }:
            raise DocumentPreparationError("Encrypted PDFs are not yet supported") from exc
        raise DocumentPreparationError(
            "Unable to open PDF; file may be corrupted or format unsupported"
        ) from exc
    except Exception as exc:
        raise DocumentPreparationError(
            "Unable to open PDF; file may be corrupted or format unsupported"
        ) from exc

    try:
        page_count = len(document)
        if page_count < 1:
            raise DocumentPreparationError("PDF contains no processable pages")
        if page_count > MAX_PDF_PAGES:
            raise DocumentPreparationError(
                f"PDF has {page_count} pages, exceeding maximum limit of {MAX_PDF_PAGES} pages"
            )

        pages = []
        for page_index in range(page_count):
            try:
                page = document[page_index]
            except Exception:
                pages.append(
                    {
                        "page_index": page_index,
                        "processing_method": "ocr",
                        "original_image_path": None,
                        "native_text": None,
                        "width": None,
                        "height": None,
                        "preparation_error": f"PDF page {page_index + 1} cannot be read",
                    }
                )
                continue

            try:
                native_text = ""
                try:
                    text_page = page.get_textpage()
                    try:
                        native_text = normalize_pdf_text(
                            text_page.get_text_range()
                        )
                    finally:
                        text_page.close()
                except Exception:
                    # Page can still be rendered and fall back to OCR when text layer is unreadable。
                    native_text = ""

                # Get PDF page dimensions in points (PDF native coordinate system)
                page_width_pts, page_height_pts = page.get_size()

                page_dir = task_dir / "pages" / f"page_{page_index + 1:04d}"
                original_path = page_dir / "original.png"
                image_width, image_height, render_scale = _render_pdf_page(
                    page, original_path, page_width_pts, page_height_pts
                )
                processing_method = (
                    "ocr"
                    if force_ocr or not should_use_native_pdf_text(native_text)
                    else "native_text"
                )
                pages.append(
                    {
                        "page_index": page_index,
                        "processing_method": processing_method,
                        "original_image_path": str(original_path),
                        "native_text": (
                            native_text if processing_method == "native_text" else None
                        ),
                        "width": image_width,
                        "height": image_height,
                        "pdf_width_pts": page_width_pts,
                        "pdf_height_pts": page_height_pts,
                        "render_scale": render_scale,
                        "preparation_error": None,
                    }
                )
            except Exception as exc:
                pages.append(
                    {
                        "page_index": page_index,
                        "processing_method": "ocr",
                        "original_image_path": None,
                        "native_text": None,
                        "width": None,
                        "height": None,
                        "preparation_error": (
                            f"PDF page {page_index + 1} failed to render: {exc}"
                        ),
                    }
                )
            finally:
                page.close()
        return pages
    finally:
        document.close()


def _prepare_image(source_path: Path) -> list[dict[str, Any]]:
    try:
        encoded = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise DocumentPreparationError("Unable to read uploaded image") from exc

    if image is None or image.size == 0:
        raise DocumentPreparationError("Unable to read uploaded image")

    height, width = image.shape[:2]
    return [
        {
            "page_index": 0,
            "processing_method": "ocr",
            "original_image_path": str(source_path),
            "native_text": None,
            "width": int(width),
            "height": int(height),
            "preparation_error": None,
        }
    ]


def prepare_document_file(source_path: str, force_ocr: bool = False) -> dict[str, Any]:
    """Prepare an uploaded source file inside a process-pool worker."""
    path = Path(source_path)
    if not path.is_file():
        raise DocumentPreparationError("Uploaded file does not exist")

    file_type = "pdf" if path.suffix.lower() == ".pdf" else "image"
    task_dir = path.parent
    pages = (
        _prepare_pdf(path, task_dir, force_ocr)
        if file_type == "pdf"
        else _prepare_image(path)
    )
    return {"file_type": file_type, "pages": pages}


def prepare_document_files(source_paths: list[str]) -> dict[str, Any]:
    """Prepare an ordered group of images as one multi-page task."""
    if not source_paths:
        raise DocumentPreparationError("Multi-page task contains no images")
    if len(source_paths) > MAX_MULTI_IMAGE_PAGES:
        raise DocumentPreparationError(
            f"Multi-page task has {len(source_paths)} images, "
            f"exceeding maximum limit of {MAX_MULTI_IMAGE_PAGES}"
        )
    if len(source_paths) == 1:
        return prepare_document_file(source_paths[0])

    pages = []
    for page_index, source_path in enumerate(source_paths):
        path = Path(source_path)
        if path.suffix.lower() == ".pdf":
            raise DocumentPreparationError("Multi-page image mode does not yet support PDF")
        try:
            page = _prepare_image(path)[0]
            page["page_index"] = page_index
            pages.append(page)
        except DocumentPreparationError as exc:
            pages.append(
                {
                    "page_index": page_index,
                    "processing_method": "ocr",
                    "original_image_path": None,
                    "native_text": None,
                    "width": None,
                    "height": None,
                    "preparation_error": (
                        f"Image {page_index + 1} failed to process: {exc}"
                    ),
                }
            )
    return {"file_type": "image", "pages": pages}
