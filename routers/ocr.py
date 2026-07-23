# -*- coding: utf-8 -*-

from typing import Any

import requests
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from models.OCRModel import *
from models.RestfulModel import *
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
from utils.ocr_runtime import build_doc_preprocessor, build_ocr

router = APIRouter(prefix="/ocr", tags=["OCR"])

_ocr_instance: Any | None = None
_doc_preprocessor: Any | None = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = build_ocr()
    return _ocr_instance


def _get_doc_preprocessor():
    global _doc_preprocessor
    if _doc_preprocessor is None:
        _doc_preprocessor = build_doc_preprocessor()
    return _doc_preprocessor


def _preprocess_image(image: Any) -> tuple[Any, dict[str, Any]]:
    result = _get_doc_preprocessor().predict(
        image,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    if not result:
        raise RuntimeError("文档矫正未返回任何结果")

    doc_result = result[0]
    output_img = doc_result.get("output_img")
    if output_img is None:
        raise RuntimeError("文档矫正结果缺少 output_img")

    doc_meta = {
        "input_path": None,
        "page_index": doc_result.get("page_index"),
        "model_settings": doc_result.get("model_settings", {}),
        "angle": int(doc_result.get("angle", -1)),
    }
    return output_img, doc_meta


def _run_ocr(image: Any, use_doc_preprocessor: bool = False):
    ocr_input = image
    ocr_image_variant = "original"
    doc_preprocessor_meta = None

    if use_doc_preprocessor:
        ocr_input, doc_preprocessor_meta = _preprocess_image(image)
        ocr_image_variant = "corrected"

    results = _get_ocr().predict(ocr_input)
    serialized = [result.json.get("res", result.json) for result in results]

    for item in serialized:
        model_settings = item.setdefault("model_settings", {})
        model_settings["use_doc_preprocessor"] = use_doc_preprocessor
        item["ocr_image_variant"] = ocr_image_variant
        if doc_preprocessor_meta is not None:
            item["doc_preprocessor_res"] = doc_preprocessor_meta

    return serialized


@router.get('/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(image_path: str, use_doc_preprocessor: bool = False):
    result = _run_ocr(image_path, use_doc_preprocessor=use_doc_preprocessor)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(base64model: Base64PostModel, use_doc_preprocessor: bool = False):
    img = base64_to_ndarray(base64model.base64_str)
    result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(file: UploadFile, use_doc_preprocessor: bool = Form(False)):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_bytes = file.file.read()
        img = bytes_to_ndarray(file_bytes)
        result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(imageUrl: str, use_doc_preprocessor: bool = False):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
        restfulModel.data = result
        restfulModel.message = "Success"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel
