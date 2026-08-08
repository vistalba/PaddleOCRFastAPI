# -*- coding: utf-8 -*-
"""
OCR worker functions for ProcessPoolExecutor.
Each worker process lazily loads its own OCR/DocPreprocessor instance on first call,
reusing it for subsequent tasks to avoid initialization overhead.
"""

from pathlib import Path
from typing import Any

import cv2

from utils.ocr_runtime import build_doc_preprocessor, build_ocr
from utils.document_processor import prepare_document_file, prepare_document_files

_ocr_instances = {}
_doc_preprocessor = None


def _get_ocr():
    global _ocr_instances
    if "default" in _ocr_instances:
        return _ocr_instances["default"]

    ocr = build_ocr()

    _ocr_instances["default"] = ocr
    return ocr


def _get_doc_preprocessor():
    global _doc_preprocessor
    if _doc_preprocessor is not None:
        return _doc_preprocessor

    _doc_preprocessor = build_doc_preprocessor()
    return _doc_preprocessor


def _save_png(image: Any, output_path: Path) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to save corrected image")
    output_path.write_bytes(encoded.tobytes())


def _run_doc_preprocessor(image_path: Path) -> tuple[Path, dict[str, Any]]:
    result = _get_doc_preprocessor().predict(
        str(image_path),
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    if not result:
        raise RuntimeError("Document preprocessor returned no results")

    doc_result = result[0]
    output_img = doc_result.get("output_img")
    if output_img is None:
        raise RuntimeError("Document preprocessor result missing output_img")

    corrected_path = image_path.parent / "corrected.png"
    _save_png(output_img, corrected_path)

    doc_meta = {
        "input_path": None,
        "page_index": doc_result.get("page_index"),
        "model_settings": doc_result.get("model_settings", {}),
        "angle": int(doc_result.get("angle", -1)),
    }
    return corrected_path, doc_meta


def run_ocr_file(image_path: str, use_doc_preprocessor: bool = False) -> dict[str, Any]:
    """Execute OCR recognition in worker process, returning serializable dict.
    This function must be a top-level function to support cross-process pickle."""
    source_path = Path(image_path)
    ocr_input = str(source_path)
    ocr_image_variant = "original"
    doc_preprocessor_meta = None
    corrected_image_path = None

    if use_doc_preprocessor:
        corrected_path, doc_preprocessor_meta = _run_doc_preprocessor(source_path)
        ocr_input = str(corrected_path)
        ocr_image_variant = "corrected"
        corrected_image_path = str(corrected_path)

    # After explicit preprocessing, OCR always recognizes the current input image, avoiding re-entry into internal document correction pipeline.
    ocr = _get_ocr()
    results = ocr.predict(ocr_input)
    serialized = [result.json.get("res", result.json) for result in results]

    for item in serialized:
        item["input_path"] = None
        model_settings = item.setdefault("model_settings", {})
        model_settings["use_doc_preprocessor"] = use_doc_preprocessor
        item["ocr_image_variant"] = ocr_image_variant
        if doc_preprocessor_meta is not None:
            item["doc_preprocessor_res"] = doc_preprocessor_meta

    return {
        "ocr_result": serialized,
        "ocr_image_variant": ocr_image_variant,
        "corrected_image_path": corrected_image_path,
    }
