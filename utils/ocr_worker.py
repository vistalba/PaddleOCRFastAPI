# -*- coding: utf-8 -*-
"""
OCR worker 函数，专供 ProcessPoolExecutor 使用。
每个 worker 进程在第一次调用时懒加载自己的 OCR / DocPreprocessor 实例，
后续任务直接复用，避免重复初始化开销。
"""

from pathlib import Path
from typing import Any

import cv2

from utils.ocr_runtime import build_doc_preprocessor, build_ocr

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
        raise RuntimeError("保存矫正后图片失败")
    output_path.write_bytes(encoded.tobytes())


def _run_doc_preprocessor(image_path: Path) -> tuple[Path, dict[str, Any]]:
    result = _get_doc_preprocessor().predict(
        str(image_path),
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    if not result:
        raise RuntimeError("文档矫正未返回任何结果")

    doc_result = result[0]
    output_img = doc_result.get("output_img")
    if output_img is None:
        raise RuntimeError("文档矫正结果缺少 output_img")

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
    """在 worker 进程中执行 OCR 识别，结果为可序列化的 dict。
    此函数必须是顶层函数以支持跨进程 pickle。"""
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

    # 显式预处理后，OCR 始终对当前输入图做识别，避免再次进入内部文档矫正链路。
    ocr = _get_ocr()
    results = ocr.predict(ocr_input)
    serialized = [result.json.get("res", result.json) for result in results]

    for item in serialized:
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
