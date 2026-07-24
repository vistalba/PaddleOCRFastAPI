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
    PDF_RENDER_SCALE,
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


def _safe_pdf_scale(width: float, height: float) -> float:
    if width <= 0 or height <= 0:
        raise DocumentPreparationError("PDF 页面尺寸无效")

    requested = max(0.1, PDF_RENDER_SCALE)
    estimated_pixels = width * height * requested * requested
    if estimated_pixels <= PDF_MAX_RENDER_PIXELS:
        return requested

    scale = math.sqrt(PDF_MAX_RENDER_PIXELS / (width * height))
    if scale < 0.25:
        raise DocumentPreparationError("PDF 页面尺寸过大，无法在安全范围内渲染")
    return scale


def _render_pdf_page(page: Any, output_path: Path) -> tuple[int, int]:
    width, height = page.get_size()
    scale = _safe_pdf_scale(float(width), float(height))
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil().convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return image.size
    finally:
        bitmap.close()


def _prepare_pdf(source_path: Path, task_dir: Path) -> list[dict[str, Any]]:
    try:
        document = pdfium.PdfDocument(str(source_path))
    except pdfium.PdfiumError as exc:
        if exc.err_code in {
            pdfium.raw.FPDF_ERR_PASSWORD,
            pdfium.raw.FPDF_ERR_SECURITY,
        }:
            raise DocumentPreparationError("暂不支持加密 PDF") from exc
        raise DocumentPreparationError(
            "无法打开 PDF，文件可能损坏或格式不受支持"
        ) from exc
    except Exception as exc:
        raise DocumentPreparationError(
            "无法打开 PDF，文件可能损坏或格式不受支持"
        ) from exc

    try:
        page_count = len(document)
        if page_count < 1:
            raise DocumentPreparationError("PDF 不包含可处理页面")
        if page_count > MAX_PDF_PAGES:
            raise DocumentPreparationError(
                f"PDF 共 {page_count} 页，超过最大限制 {MAX_PDF_PAGES} 页"
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
                        "preparation_error": f"PDF 第 {page_index + 1} 页无法读取",
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
                    # 文本层不可读时仍可渲染页面并回退到 OCR。
                    native_text = ""

                page_dir = task_dir / "pages" / f"page_{page_index + 1:04d}"
                original_path = page_dir / "original.png"
                width, height = _render_pdf_page(page, original_path)
                processing_method = (
                    "native_text"
                    if should_use_native_pdf_text(native_text)
                    else "ocr"
                )
                pages.append(
                    {
                        "page_index": page_index,
                        "processing_method": processing_method,
                        "original_image_path": str(original_path),
                        "native_text": (
                            native_text if processing_method == "native_text" else None
                        ),
                        "width": width,
                        "height": height,
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
                            f"PDF 第 {page_index + 1} 页渲染失败：{exc}"
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
        raise DocumentPreparationError("无法读取上传的图片") from exc

    if image is None or image.size == 0:
        raise DocumentPreparationError("无法读取上传的图片")

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


def prepare_document_file(source_path: str) -> dict[str, Any]:
    """Prepare an uploaded source file inside a process-pool worker."""
    path = Path(source_path)
    if not path.is_file():
        raise DocumentPreparationError("上传文件不存在")

    file_type = "pdf" if path.suffix.lower() == ".pdf" else "image"
    task_dir = path.parent
    pages = (
        _prepare_pdf(path, task_dir)
        if file_type == "pdf"
        else _prepare_image(path)
    )
    return {"file_type": file_type, "pages": pages}


def prepare_document_files(source_paths: list[str]) -> dict[str, Any]:
    """Prepare an ordered group of images as one multi-page task."""
    if not source_paths:
        raise DocumentPreparationError("多页任务不包含图片")
    if len(source_paths) > MAX_MULTI_IMAGE_PAGES:
        raise DocumentPreparationError(
            f"多页任务共 {len(source_paths)} 张图片，"
            f"超过最大限制 {MAX_MULTI_IMAGE_PAGES} 张"
        )
    if len(source_paths) == 1:
        return prepare_document_file(source_paths[0])

    pages = []
    for page_index, source_path in enumerate(source_paths):
        path = Path(source_path)
        if path.suffix.lower() == ".pdf":
            raise DocumentPreparationError("多页图片模式暂不支持 PDF")
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
                        f"第 {page_index + 1} 张图片处理失败：{exc}"
                    ),
                }
            )
    return {"file_type": "image", "pages": pages}
