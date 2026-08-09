# -*- coding: utf-8 -*-

import asyncio
import ipaddress
import json
import logging
import mimetypes
import re
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import (
    MAX_CONCURRENT_OCR,
    MAX_MULTI_IMAGE_PAGES,
    MAX_UPLOAD_SIZE_MB,
    RATE_LIMIT,
    RETRY_COOLDOWN_SECONDS,
    UPLOAD_DIR,
)
from database import SessionLocal, get_db
from limiter import get_client_ip, limiter
from models.TaskModel import Task, TaskPage
from utils.ocr_worker import (
    prepare_document_file,
    prepare_document_files,
    run_ocr_file,
)
from utils.text_organizer import (
    ORGANIZATION_RESULT_VERSION,
    TextOrganizerError,
    ai_organizer_status,
    build_ai_result,
    build_rule_result,
)

logger = logging.getLogger(__name__)

# OCR models and PDF rendering run in the process pool to avoid blocking the FastAPI event loop.
_ocr_pool = ProcessPoolExecutor(max_workers=MAX_CONCURRENT_OCR)
_ocr_pool_lock = asyncio.Lock()

# llama.cpp manages CPU threads itself; a separate single-worker thread pool ensures the model is loaded only once.
_ai_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="text-ai")
_ai_page_locks: dict[tuple[str, int], asyncio.Lock] = {}

# One queue task corresponds to an ordered set of inputs: single image/PDF = 1 path, multiple images = N paths.
_task_queue: asyncio.Queue[tuple[str, tuple[Path, ...]]] = asyncio.Queue()
_worker_tasks: list[asyncio.Task] = []

router = APIRouter(prefix="/ocr", tags=["Tasks"])

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | {".pdf"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
RETRY_STATE_FILENAME = "retry_state.json"


# ── Response schemas ──────────────────────────────────────────────────────────

class TaskCreateResponse(BaseModel):
    task_id: str
    status: str


class TaskImageVariants(BaseModel):
    original: Optional[str] = None
    corrected: Optional[str] = None


class TaskPageResponse(BaseModel):
    page_index: int
    status: str
    processing_method: str
    width: Optional[int] = None
    height: Optional[int] = None
    image_variants: TaskImageVariants
    default_image_variant: str = "original"
    ocr_image_variant: str = "original"
    native_text: Optional[str] = None
    ocr_result: Optional[Any] = None
    lines: List[dict] = Field(default_factory=list)
    rule_result: Optional[Any] = None
    ai_result: Optional[Any] = None
    ai_status: str = "not_started"
    ai_model_name: Optional[str] = None
    ai_processed_at: Optional[datetime] = None
    ai_error: Optional[str] = None
    error_msg: Optional[str] = None


class TaskDetailResponse(BaseModel):
    task_id: str
    ip: str
    created_at: datetime
    status: str
    original_filename: Optional[str]
    file_type: str = "image"
    use_doc_preprocessor: bool = False
    page_count: int = 0
    completed_pages: int = 0
    failed_pages: int = 0
    pages: List[TaskPageResponse] = Field(default_factory=list)

    # First page compatibility field for old frontends or old API callers.
    image_variants: TaskImageVariants
    default_image_variant: str = "original"
    ocr_image_variant: str = "original"
    ocr_result: Optional[Any]
    error_msg: Optional[str]
    queue_position: Optional[int] = None
    retry_available_at: Optional[datetime] = None
    retry_after_seconds: int = 0
    ai_organizer_available: bool = False
    ai_organizer_model: Optional[str] = None
    ai_organizer_unavailable_reason: Optional[str] = None
    ai_repeat_allowed: bool = False


class TaskListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[TaskDetailResponse]


class AIOrganizePageResponse(BaseModel):
    task_id: str
    page_index: int
    ai_status: str
    ai_result: Optional[Any] = None
    ai_model_name: Optional[str] = None
    ai_processed_at: Optional[datetime] = None
    ai_error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Return naive UTC to match the existing SQLite datetime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sanitize_ip(ip: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", ip)


def _task_dir(ip: str, task_id: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    folder = UPLOAD_DIR / f"{_sanitize_ip(ip)}_{task_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _task_source_files(task_dir: Path) -> list[Path]:
    multi_image_sources = sorted(
        task_dir.glob("sources/page_*/original.*")
    )
    if multi_image_sources:
        return multi_image_sources
    candidates = sorted(task_dir.glob("original.*"))
    return candidates[:1]


def _task_source_file(task_dir: Path) -> Optional[Path]:
    sources = _task_source_files(task_dir)
    return sources[0] if sources else None


def _task_file_type(task: Task) -> str:
    filename = task.original_filename or ""
    return "pdf" if Path(filename).suffix.lower() == ".pdf" else "image"


def _parse_json(value: Optional[str]) -> Optional[Any]:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _is_loopback_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _build_rule_result_for_page(page: TaskPage) -> dict[str, Any]:
    return build_rule_result(
        processing_method=page.processing_method,
        native_text=page.native_text,
        ocr_result=_parse_json(page.ocr_result),
        page_width=page.width,
    )


def _ensure_rule_result(page: TaskPage) -> bool:
    if page.status != "done":
        return False
    existing_result = _parse_json(page.rule_result)
    if (
        isinstance(existing_result, dict)
        and existing_result.get("version")
        == ORGANIZATION_RESULT_VERSION
    ):
        return False
    page.rule_result = _serialize_json(_build_rule_result_for_page(page))
    return True


def _retry_state_path(task: Task) -> Optional[Path]:
    if not task.file_dir:
        return None
    return Path(task.file_dir) / RETRY_STATE_FILENAME


def _write_retry_state(
    task: Task,
    available_at: Optional[datetime] = None,
) -> Optional[datetime]:
    state_path = _retry_state_path(task)
    if state_path is None:
        return None
    retry_available_at = available_at or (
        _utcnow()
        + timedelta(seconds=max(0, RETRY_COOLDOWN_SECONDS))
    )
    state_path.write_text(
        json.dumps(
            {"retry_available_at": retry_available_at.isoformat()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return retry_available_at


def _read_retry_available_at(task: Task) -> Optional[datetime]:
    state_path = _retry_state_path(task)
    if state_path is not None and state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            value = payload.get("retry_available_at")
            if isinstance(value, str):
                return datetime.fromisoformat(value)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning(
                "Unable to read task retry status, task_id=%s",
                task.task_id,
            )

    # Compatibility for tasks failed before upgrade: infer cooldown from manifest mtime.
    if task.file_dir:
        manifest_path = Path(task.file_dir) / "manifest.json"
        if manifest_path.is_file():
            return (
                datetime.fromtimestamp(
                    manifest_path.stat().st_mtime,
                    timezone.utc,
                ).replace(tzinfo=None)
                + timedelta(seconds=max(0, RETRY_COOLDOWN_SECONDS))
            )
    return (task.created_at or _utcnow()) + timedelta(
        seconds=max(0, RETRY_COOLDOWN_SECONDS)
    )


def _retry_after_seconds(task: Task) -> tuple[Optional[datetime], int]:
    if task.status != "failed":
        return None, 0
    available_at = _read_retry_available_at(task)
    if available_at is None:
        return None, 0
    remaining = max(
        0,
        ceil((available_at - _utcnow()).total_seconds()),
    )
    return available_at, remaining


def _clear_retry_state(task: Task) -> None:
    state_path = _retry_state_path(task)
    if state_path is not None:
        state_path.unlink(missing_ok=True)


def _remove_task_files(file_dir: Optional[str]) -> None:
    if not file_dir:
        return

    upload_root = UPLOAD_DIR.resolve()
    task_dir = Path(file_dir).resolve()
    try:
        task_dir.relative_to(upload_root)
    except ValueError:
        logger.error("Refusing to delete task files outside upload directory: %s", task_dir)
        return

    if task_dir == upload_root:
        logger.error("Refusing to delete upload root directory: %s", task_dir)
        return

    if task_dir.is_dir():
        shutil.rmtree(task_dir)


def _legacy_task_image_files(task_dir: Path) -> dict[str, Optional[Path]]:
    original_path = _task_source_file(task_dir)
    corrected_path = task_dir / "corrected.png"
    return {
        "original": original_path,
        "corrected": corrected_path if corrected_path.exists() else None,
    }


def _page_image_files(page: TaskPage) -> dict[str, Optional[Path]]:
    original_path = (
        Path(page.original_image_path) if page.original_image_path else None
    )
    corrected_path = (
        Path(page.corrected_image_path) if page.corrected_image_path else None
    )
    return {
        "original": (
            original_path
            if original_path is not None and original_path.exists()
            else None
        ),
        "corrected": (
            corrected_path
            if corrected_path is not None and corrected_path.exists()
            else None
        ),
    }


def _resolve_result_image_variant(
    ocr_result: Optional[Any],
    corrected_exists: bool,
    use_doc_preprocessor: bool,
) -> str:
    if isinstance(ocr_result, list) and ocr_result:
        first_item = ocr_result[0]
        if isinstance(first_item, dict):
            variant = first_item.get("ocr_image_variant")
            if variant in {"original", "corrected"}:
                return variant
    if corrected_exists and use_doc_preprocessor:
        return "corrected"
    return "original"


def _extract_lines_from_ocr_result(ocr_result: Optional[Any]) -> list[dict]:
    """Extract text lines with bounding boxes from PaddleOCR ocr_result."""
    lines = []
    if not isinstance(ocr_result, list):
        return lines
    
    for item in ocr_result:
        if not isinstance(item, dict):
            continue
        
        texts = item.get("rec_texts", [])
        boxes = item.get("rec_boxes", [])
        
        if isinstance(texts, list) and isinstance(boxes, list):
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else []
                if text and isinstance(text, str):
                    lines.append({
                        "text": text.strip(),
                        "box": box if isinstance(box, list) else []
                    })
    
    return lines


def _build_page_image_variants(
    task_id: str,
    page_index: int,
    image_files: dict[str, Optional[Path]],
) -> TaskImageVariants:
    base_url = f"/ocr/tasks/{task_id}/pages/{page_index}/image"
    return TaskImageVariants(
        original=(
            f"{base_url}?variant=original"
            if image_files["original"] is not None
            else None
        ),
        corrected=(
            f"{base_url}?variant=corrected"
            if image_files["corrected"] is not None
            else None
        ),
    )


def _build_task_image_variants(
    task_id: str,
    image_files: dict[str, Optional[Path]],
) -> TaskImageVariants:
    return TaskImageVariants(
        original=(
            f"/ocr/tasks/{task_id}/image?variant=original"
            if image_files["original"] is not None
            else None
        ),
        corrected=(
            f"/ocr/tasks/{task_id}/image?variant=corrected"
            if image_files["corrected"] is not None
            else None
        ),
    )


def _build_page_response(
    task: Task,
    page: TaskPage,
    include_result: bool,
) -> TaskPageResponse:
    ocr_result = _parse_json(page.ocr_result) if include_result else None
    image_files = _page_image_files(page)
    image_variants = _build_page_image_variants(
        task.task_id, page.page_index, image_files
    )
    ocr_image_variant = _resolve_result_image_variant(
        ocr_result,
        corrected_exists=image_files["corrected"] is not None,
        use_doc_preprocessor=bool(task.use_doc_preprocessor),
    )
    default_image_variant = (
        "corrected" if image_files["corrected"] is not None else "original"
    )
    lines = _extract_lines_from_ocr_result(ocr_result) if include_result else []
    return TaskPageResponse(
        page_index=page.page_index,
        status=page.status,
        processing_method=page.processing_method,
        width=page.width,
        height=page.height,
        image_variants=image_variants,
        default_image_variant=default_image_variant,
        ocr_image_variant=ocr_image_variant,
        native_text=page.native_text if include_result else None,
        ocr_result=ocr_result,
        lines=lines,
        rule_result=(
            _parse_json(page.rule_result) if include_result else None
        ),
        ai_result=(
            _parse_json(page.ai_result) if include_result else None
        ),
        ai_status=page.ai_status or "not_started",
        ai_model_name=page.ai_model_name,
        ai_processed_at=page.ai_processed_at,
        ai_error=page.ai_error,
        error_msg=page.error_msg,
    )


def _build_legacy_page_response(
    task: Task,
    include_result: bool,
) -> Optional[TaskPageResponse]:
    if not task.file_dir or _task_file_type(task) == "pdf":
        return None

    task_dir = Path(task.file_dir)
    image_files = _legacy_task_image_files(task_dir)
    if image_files["original"] is None:
        return None

    ocr_result = _parse_json(task.ocr_result) if include_result else None
    rule_result = (
        build_rule_result(
            processing_method="ocr",
            native_text=None,
            ocr_result=ocr_result,
        )
        if include_result and ocr_result
        else None
    )
    ocr_image_variant = _resolve_result_image_variant(
        ocr_result,
        corrected_exists=image_files["corrected"] is not None,
        use_doc_preprocessor=bool(task.use_doc_preprocessor),
    )
    return TaskPageResponse(
        page_index=0,
        status=task.status,
        processing_method="ocr",
        image_variants=_build_task_image_variants(task.task_id, image_files),
        default_image_variant=(
            "corrected" if image_files["corrected"] is not None else "original"
        ),
        ocr_image_variant=ocr_image_variant,
        ocr_result=ocr_result,
        rule_result=rule_result,
        error_msg=task.error_msg,
    )


def _build_task_response(
    db: Session,
    task: Task,
    queue_position: Optional[int] = None,
    include_page_results: bool = True,
    ai_repeat_allowed: bool = False,
) -> TaskDetailResponse:
    page_models = (
        db.query(TaskPage)
        .filter(TaskPage.task_id == task.task_id)
        .order_by(TaskPage.page_index)
        .all()
    )
    if include_page_results:
        rule_result_pages: list[TaskPage] = []
        for page in page_models:
            if _ensure_rule_result(page):
                rule_result_pages.append(page)
        if rule_result_pages:
            db.commit()
            for page in rule_result_pages:
                _write_page_result(page)
    pages = [
        _build_page_response(task, page, include_page_results)
        for page in page_models
    ]

    if not pages:
        legacy_page = _build_legacy_page_response(task, include_page_results)
        if legacy_page is not None:
            pages = [legacy_page]

    first_page = pages[0] if pages else None
    image_variants = (
        first_page.image_variants if first_page else TaskImageVariants()
    )
    completed_pages = sum(page.status == "done" for page in pages)
    failed_pages = sum(page.status == "failed" for page in pages)
    retry_available_at, retry_after_seconds = _retry_after_seconds(task)
    ai_status = ai_organizer_status()
    return TaskDetailResponse(
        task_id=task.task_id,
        ip=task.ip,
        created_at=task.created_at,
        status=task.status,
        original_filename=task.original_filename,
        file_type=_task_file_type(task),
        use_doc_preprocessor=bool(task.use_doc_preprocessor),
        page_count=len(pages),
        completed_pages=completed_pages,
        failed_pages=failed_pages,
        pages=pages,
        image_variants=image_variants,
        default_image_variant=(
            first_page.default_image_variant if first_page else "original"
        ),
        ocr_image_variant=(
            first_page.ocr_image_variant if first_page else "original"
        ),
        ocr_result=(
            _parse_json(task.ocr_result) if include_page_results else None
        ),
        error_msg=task.error_msg,
        queue_position=queue_position,
        retry_available_at=retry_available_at,
        retry_after_seconds=retry_after_seconds,
        ai_organizer_available=bool(ai_status["available"]),
        ai_organizer_model=ai_status["model_name"],
        ai_organizer_unavailable_reason=ai_status["reason"],
        ai_repeat_allowed=ai_repeat_allowed,
    )


async def _save_upload_file(
    file: UploadFile,
    output_path: Path,
    max_bytes: Optional[int] = None,
    limit_label: str = "File size",
) -> int:
    byte_limit = (
        max_bytes
        if max_bytes is not None
        else MAX_UPLOAD_SIZE_MB * 1024 * 1024
    )
    total = 0
    with output_path.open("wb") as output:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > byte_limit:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"{limit_label} exceeds {MAX_UPLOAD_SIZE_MB} MB limit"
                    ),
                )
            output.write(chunk)

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    return total


def _validate_uploaded_file(path: Path, suffix: str) -> None:
    if suffix == ".pdf":
        with path.open("rb") as source:
            header = source.read(1024)
        if b"%PDF-" not in header:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content is not a valid PDF",
            )
        return

    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content is not a valid image",
        ) from exc


def _write_page_result(page: TaskPage) -> None:
    if not page.original_image_path:
        return
    result_path = Path(page.original_image_path).parent / "result.json"
    payload = {
        "page_index": page.page_index,
        "processing_method": page.processing_method,
        "native_text": page.native_text,
        "ocr_result": _parse_json(page.ocr_result),
        "rule_result": _parse_json(page.rule_result),
        "ai_result": _parse_json(page.ai_result),
        "ai_status": page.ai_status or "not_started",
        "ai_model_name": page.ai_model_name,
        "ai_processed_at": (
            page.ai_processed_at.isoformat()
            if page.ai_processed_at
            else None
        ),
        "ai_error": page.ai_error,
        "error_msg": page.error_msg,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_manifest(task: Task, pages: list[TaskPage]) -> None:
    if not task.file_dir:
        return
    payload = {
        "task_id": task.task_id,
        "file_type": _task_file_type(task),
        "page_count": len(pages),
        "status": task.status,
        "pages": [
            {
                "page_index": page.page_index,
                "status": page.status,
                "processing_method": page.processing_method,
                "width": page.width,
                "height": page.height,
                "error_msg": page.error_msg,
            }
            for page in pages
        ],
    }
    (Path(task.file_dir) / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


async def _run_in_ocr_pool(function, *args):
    """Run work in the OCR pool and replace a pool whose child has died."""
    global _ocr_pool

    active_pool = _ocr_pool
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            active_pool,
            function,
            *args,
        )
    except BrokenProcessPool:
        async with _ocr_pool_lock:
            if _ocr_pool is active_pool:
                try:
                    active_pool.shutdown(
                        wait=False,
                        cancel_futures=True,
                    )
                except Exception:
                    logger.exception("Failed to release corrupted OCR process pool")
                _ocr_pool = ProcessPoolExecutor(
                    max_workers=MAX_CONCURRENT_OCR
                )
                logger.error("OCR subprocess exited abnormally; process pool has been rebuilt")
        raise


# ── Queue workers ─────────────────────────────────────────────────────────────

async def _ocr_queue_worker() -> None:
    while True:
        task_id, source_paths = await _task_queue.get()
        try:
            await _process_ocr_async(task_id, source_paths)
        except Exception:
            logger.exception("OCR worker encountered uncaught exception, task_id=%s", task_id)
        finally:
            _task_queue.task_done()


async def start_workers() -> None:
    for _ in range(MAX_CONCURRENT_OCR):
        worker_task = asyncio.create_task(_ocr_queue_worker())
        _worker_tasks.append(worker_task)

    db = SessionLocal()
    try:
        stale_ai_pages = (
            db.query(TaskPage)
            .filter(TaskPage.ai_status == "processing")
            .all()
        )
        for page in stale_ai_pages:
            page.ai_status = "failed"
            page.ai_error = "Service restart interrupted AI organization; please re-organize"
        if stale_ai_pages:
            db.commit()

        stuck = (
            db.query(Task)
            .filter(Task.status.in_(["queued", "processing"]))
            .order_by(Task.created_at)
            .all()
        )
        for task in stuck:
            if not task.file_dir:
                continue
            source_paths = _task_source_files(Path(task.file_dir))
            if not source_paths:
                continue
            task.status = "queued"
            db.commit()
            await _task_queue.put((task.task_id, tuple(source_paths)))
            logger.info("Crash recovery: re-queuing task_id=%s", task.task_id)
    finally:
        db.close()


async def stop_workers() -> None:
    for worker_task in _worker_tasks:
        worker_task.cancel()
    await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks.clear()


def _upsert_task_pages(
    db: Session,
    task: Task,
    prepared_pages: list[dict[str, Any]],
) -> list[TaskPage]:
    existing_pages = {
        page.page_index: page
        for page in (
            db.query(TaskPage)
            .filter(TaskPage.task_id == task.task_id)
            .all()
        )
    }
    valid_indexes = {item["page_index"] for item in prepared_pages}
    for page_index, page in existing_pages.items():
        if page_index not in valid_indexes:
            db.delete(page)

    pages = []
    for item in prepared_pages:
        page = existing_pages.get(item["page_index"])
        if page is None:
            page = TaskPage(
                task_id=task.task_id,
                page_index=item["page_index"],
            )
            db.add(page)

        preparation_error = item.get("preparation_error")
        page.status = "failed" if preparation_error else "queued"
        page.processing_method = item["processing_method"]
        page.width = item.get("width")
        page.height = item.get("height")
        page.original_image_path = item.get("original_image_path")
        page.corrected_image_path = None
        page.native_text = item.get("native_text")
        page.ocr_result = None
        page.rule_result = None
        page.ai_result = None
        page.ai_status = "not_started"
        page.ai_model_name = None
        page.ai_processed_at = None
        page.ai_error = None
        page.error_msg = preparation_error
        pages.append(page)

    db.commit()
    return (
        db.query(TaskPage)
        .filter(TaskPage.task_id == task.task_id)
        .order_by(TaskPage.page_index)
        .all()
    )


async def _process_ocr_async(
    task_id: str,
    source_paths: Path | tuple[Path, ...] | list[Path],
) -> None:
    db = SessionLocal()
    try:
        task: Task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return

        task.status = "processing"
        task.error_msg = None
        db.commit()

        normalized_sources = (
            [source_paths]
            if isinstance(source_paths, Path)
            else list(source_paths)
        )
        if len(normalized_sources) == 1:
            prepared = await _run_in_ocr_pool(
                prepare_document_file,
                str(normalized_sources[0]),
                bool(task.force_ocr),
            )
        else:
            prepared = await _run_in_ocr_pool(
                prepare_document_files,
                [str(path) for path in normalized_sources],
            )
        pages = _upsert_task_pages(db, task, prepared["pages"])

        successful_pages = 0
        failed_pages = 0
        first_page_ocr_result: Optional[list[Any]] = None
        use_doc_preprocessor = bool(task.use_doc_preprocessor)

        for page in pages:
            if page.error_msg and not page.original_image_path:
                failed_pages += 1
                continue

            page.status = "processing"
            db.commit()
            try:
                if page.processing_method == "native_text":
                    page.status = "done"
                    page.corrected_image_path = None
                    page.ocr_result = None
                else:
                    worker_payload = await _run_in_ocr_pool(
                        run_ocr_file,
                        str(page.original_image_path),
                        use_doc_preprocessor,
                    )
                    page_result = worker_payload["ocr_result"]
                    page.ocr_result = json.dumps(page_result, ensure_ascii=False)
                    page.corrected_image_path = worker_payload[
                        "corrected_image_path"
                    ]
                    page.status = "done"
                    if page.page_index == 0:
                        first_page_ocr_result = page_result

                page.rule_result = _serialize_json(
                    _build_rule_result_for_page(page)
                )
                page.ai_result = None
                page.ai_status = "not_started"
                page.ai_model_name = None
                page.ai_processed_at = None
                page.ai_error = None
                page.error_msg = None
                successful_pages += 1
            except Exception as exc:
                page.status = "failed"
                page.error_msg = str(exc)
                failed_pages += 1
                logger.exception(
                    "Page processing failed, task_id=%s page_index=%s",
                    task_id,
                    page.page_index,
                )

            db.commit()
            _write_page_result(page)

        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return
        task.ocr_result = (
            json.dumps(first_page_ocr_result, ensure_ascii=False)
            if first_page_ocr_result
            else None
        )
        if successful_pages > 0:
            task.status = "done"
            task.error_msg = (
                f"{failed_pages} pages failed to process"
                if failed_pages > 0
                else None
            )
            _clear_retry_state(task)
        else:
            task.status = "failed"
            task.error_msg = "All pages failed to process"
            _write_retry_state(task)
        db.commit()

        refreshed_pages = (
            db.query(TaskPage)
            .filter(TaskPage.task_id == task_id)
            .order_by(TaskPage.page_index)
            .all()
        )
        _write_manifest(task, refreshed_pages)

    except Exception as exc:
        db.rollback()
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if task:
            task.status = "failed"
            task.error_msg = str(exc)
            _write_retry_state(task)
            db.commit()
            pages = (
                db.query(TaskPage)
                .filter(TaskPage.task_id == task_id)
                .order_by(TaskPage.page_index)
                .all()
            )
            _write_manifest(task, pages)
        logger.exception("Task processing failed，task_id=%s", task_id)
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/tasks",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create OCR task (single file, can include multiple pages)",
)
@limiter.limit(RATE_LIMIT)
async def create_task(
    request: Request,
    file: UploadFile,
    use_doc_preprocessor: bool = Form(False),
    force_ocr: bool = Form(False),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format, please upload: {supported}",
        )

    ip = get_client_ip(request)
    task_id = str(uuid.uuid4())
    task_dir = _task_dir(ip, task_id)
    source_path = task_dir / f"original{suffix}"

    try:
        await _save_upload_file(file, source_path)
        _validate_uploaded_file(source_path, suffix)
    except Exception:
        if task_dir.exists():
            shutil.rmtree(task_dir)
        raise
    finally:
        await file.close()

    task = Task(
        task_id=task_id,
        ip=ip,
        created_at=_utcnow(),
        status="queued",
        original_filename=file.filename,
        file_dir=str(task_dir),
        use_doc_preprocessor=use_doc_preprocessor,
        force_ocr=force_ocr,
    )
    db.add(task)
    db.commit()

    await _task_queue.put((task_id, (source_path,)))
    return TaskCreateResponse(task_id=task_id, status="queued")


@router.post(
    "/tasks/multi-image",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create multi-image OCR task (upload order is page order)",
)
@limiter.limit(RATE_LIMIT)
async def create_multi_image_task(
    request: Request,
    files: List[UploadFile] = File(...),
    use_doc_preprocessor: bool = Form(False),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload at least one image",
        )
    if len(files) > MAX_MULTI_IMAGE_PAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Upload at most {MAX_MULTI_IMAGE_PAGES} images at once"
            ),
        )

    task_dir: Optional[Path] = None
    source_paths: list[Path] = []
    original_names: list[str] = []
    try:
        suffixes = []
        for file in files:
            filename = file.filename or ""
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                supported = ", ".join(
                    sorted(ALLOWED_IMAGE_EXTENSIONS)
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
detail=(
                        "Multi-page mode only supports images, not PDF; "
                        f"please upload: {supported}"
                    ),
                )
            suffixes.append(suffix)
            original_names.append(filename)

        ip = get_client_ip(request)
        task_id = str(uuid.uuid4())
        task_dir = _task_dir(ip, task_id)
        remaining_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

        for page_index, (file, suffix) in enumerate(
            zip(files, suffixes)
        ):
            page_dir = (
                task_dir
                / "sources"
                / f"page_{page_index + 1:04d}"
            )
            page_dir.mkdir(parents=True, exist_ok=True)
            source_path = page_dir / f"original{suffix}"
            written = await _save_upload_file(
                file,
                source_path,
                max_bytes=remaining_bytes,
                limit_label="Multi-page task total file size",
            )
            remaining_bytes -= written
            _validate_uploaded_file(source_path, suffix)
            source_paths.append(source_path)
    except Exception:
        if task_dir is not None and task_dir.exists():
            shutil.rmtree(task_dir)
        raise
    finally:
        await asyncio.gather(
            *(file.close() for file in files),
            return_exceptions=True,
        )

    display_name = (
        original_names[0]
        if len(original_names) == 1
        else f"{original_names[0]} and {len(original_names) - 1} other image(s)"
    )
    task = Task(
        task_id=task_id,
        ip=ip,
        created_at=_utcnow(),
        status="queued",
        original_filename=display_name,
        file_dir=str(task_dir),
        use_doc_preprocessor=use_doc_preprocessor,
    )
    try:
        db.add(task)
        db.commit()
    except Exception:
        db.rollback()
        if task_dir.exists():
            shutil.rmtree(task_dir)
        raise

    await _task_queue.put((task_id, tuple(source_paths)))
    return TaskCreateResponse(task_id=task_id, status="queued")


def _reset_failed_task_for_retry(
    db: Session,
    task: Task,
) -> tuple[Path, ...]:
    if task.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed tasks can be retried",
        )

    _, retry_after_seconds = _retry_after_seconds(task)
    if retry_after_seconds > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Please try again in {retry_after_seconds} seconds",
            headers={
                "Retry-After": str(retry_after_seconds),
            },
        )

    if not task.file_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task original file does not exist, cannot retry",
        )
    source_paths = tuple(_task_source_files(Path(task.file_dir)))
    if not source_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task original file does not exist, cannot retry",
        )

    task.status = "queued"
    task.error_msg = None
    task.ocr_result = None
    pages = (
        db.query(TaskPage)
        .filter(TaskPage.task_id == task.task_id)
        .order_by(TaskPage.page_index)
        .all()
    )
    for page in pages:
        page.status = "queued"
        page.error_msg = None
        page.ocr_result = None
        page.rule_result = None
        page.ai_result = None
        page.ai_status = "not_started"
        page.ai_model_name = None
        page.ai_processed_at = None
        page.ai_error = None
        page.corrected_image_path = None

    db.commit()
    _clear_retry_state(task)
    _write_manifest(task, pages)
    return source_paths


@router.post(
    "/tasks/{task_id}/retry",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry failed task after cooldown period",
)
@limiter.limit(RATE_LIMIT)
async def retry_task(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db),
):
    task: Task = (
        db.query(Task)
        .filter(Task.task_id == task_id)
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task does not exist",
        )

    source_paths = _reset_failed_task_for_retry(db, task)
    await _task_queue.put((task_id, source_paths))
    return TaskCreateResponse(task_id=task_id, status="queued")


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    summary="Query OCR task status and page-by-page results",
)
def get_task(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db),
):
    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task does not exist",
        )

    queue_position: Optional[int] = None
    if task.status == "queued":
        ahead = (
            db.query(Task)
            .filter(Task.status == "queued", Task.created_at < task.created_at)
            .count()
        )
        queue_position = ahead + 1

    return _build_task_response(
        db,
        task,
        queue_position,
        ai_repeat_allowed=_is_loopback_ip(get_client_ip(request)),
    )


def _materialize_legacy_page(
    db: Session,
    task: Task,
    page_index: int,
) -> Optional[TaskPage]:
    if (
        page_index != 0
        or _task_file_type(task) != "image"
        or not task.file_dir
    ):
        return None
    image_files = _legacy_task_image_files(Path(task.file_dir))
    if image_files["original"] is None:
        return None
    page = TaskPage(
        task_id=task.task_id,
        page_index=0,
        status=task.status,
        processing_method="ocr",
        original_image_path=str(image_files["original"]),
        corrected_image_path=(
            str(image_files["corrected"])
            if image_files["corrected"] is not None
            else None
        ),
        ocr_result=task.ocr_result,
    )
    db.add(page)
    db.flush()
    _ensure_rule_result(page)
    db.commit()
    return page


async def _run_in_ai_pool(page: TaskPage) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _ai_pool,
        build_ai_result,
        page.processing_method,
        page.native_text,
        _parse_json(page.ocr_result),
        page.width,
    )


@router.post(
    "/tasks/{task_id}/pages/{page_index}/organize",
    response_model=AIOrganizePageResponse,
    summary="Use optional local model to organize current page text",
)
@limiter.limit(RATE_LIMIT)
async def organize_task_page(
    request: Request,
    task_id: str,
    page_index: int,
    db: Session = Depends(get_db),
):
    key = (task_id, page_index)
    page_lock = _ai_page_locks.setdefault(key, asyncio.Lock())
    async with page_lock:
        db.expire_all()
        task: Task = (
            db.query(Task)
            .filter(Task.task_id == task_id)
            .first()
        )
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task does not exist",
            )
        page = (
            db.query(TaskPage)
            .filter(
                TaskPage.task_id == task_id,
                TaskPage.page_index == page_index,
            )
            .first()
        )
        if page is None:
            page = _materialize_legacy_page(
                db, task, page_index
            )
        if page is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Page does not exist",
            )
        if page.status != "done":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Page must be recognized before AI organization",
            )

        repeat_allowed = _is_loopback_ip(get_client_ip(request))
        if page.ai_result and not repeat_allowed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Current page has already been organized by AI",
            )
        if page.ai_status == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Current page is being organized by AI",
            )

        organizer = ai_organizer_status()
        if not organizer["available"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=organizer["reason"],
            )
        _ensure_rule_result(page)
        rule_result = _parse_json(page.rule_result)
        if not isinstance(rule_result, dict) or not rule_result.get(
            "segments"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current page has no text to organize",
            )

        page.ai_status = "processing"
        page.ai_error = None
        db.commit()
        _write_page_result(page)

        try:
            ai_result = await _run_in_ai_pool(page)
        except TextOrganizerError as exc:
            page = (
                db.query(TaskPage)
                .filter(
                    TaskPage.task_id == task_id,
                    TaskPage.page_index == page_index,
                )
                .one()
            )
            page.ai_status = "failed"
            page.ai_error = str(exc)
            db.commit()
            _write_page_result(page)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI organization failed：{exc}",
            ) from exc
        except Exception as exc:
            logger.exception(
                "AI organization failed，task_id=%s page_index=%s",
                task_id,
                page_index,
            )
            page = (
                db.query(TaskPage)
                .filter(
                    TaskPage.task_id == task_id,
                    TaskPage.page_index == page_index,
                )
                .one()
            )
            page.ai_status = "failed"
            page.ai_error = "AI organization encountered an internal error"
            db.commit()
            _write_page_result(page)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI organization encountered an internal error",
            ) from exc

        page = (
            db.query(TaskPage)
            .filter(
                TaskPage.task_id == task_id,
                TaskPage.page_index == page_index,
            )
            .one()
        )
        processed_at = _utcnow()
        page.ai_result = _serialize_json(ai_result)
        page.ai_status = "done"
        page.ai_model_name = organizer["model_name"]
        page.ai_processed_at = processed_at
        page.ai_error = None
        db.commit()
        _write_page_result(page)
        return AIOrganizePageResponse(
            task_id=task_id,
            page_index=page_index,
            ai_status="done",
            ai_result=ai_result,
            ai_model_name=page.ai_model_name,
            ai_processed_at=processed_at,
        )


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete completed OCR task and its saved files",
)
def delete_task(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db),
):
    ip = get_client_ip(request)
    task: Task = (
        db.query(Task)
        .filter(Task.task_id == task_id, Task.ip == ip)
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task does not exist",
        )
    if task.status not in {"done", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task is processing, can only delete after completion",
        )

    file_dir = task.file_dir
    db.delete(task)
    db.commit()
    try:
        _remove_task_files(file_dir)
    except OSError:
        logger.exception("Failed to clean task files，task_id=%s", task_id)


def _serve_image_file(image_file: Path) -> FileResponse:
    media_type, _ = mimetypes.guess_type(image_file.name)
    return FileResponse(
        path=str(image_file),
        media_type=media_type or "image/png",
    )


@router.get(
    "/tasks/{task_id}/pages/{page_index}/image",
    summary="Get specified page image for task",
)
def get_task_page_image(
    task_id: str,
    page_index: int,
    variant: str = "original",
    db: Session = Depends(get_db),
):
    if variant not in {"original", "corrected"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image variant",
        )

    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task does not exist",
        )

    page = (
        db.query(TaskPage)
        .filter(
            TaskPage.task_id == task_id,
            TaskPage.page_index == page_index,
        )
        .first()
    )
    if page is not None:
        image_file = _page_image_files(page)[variant]
    elif page_index == 0 and task.file_dir and _task_file_type(task) == "image":
        image_file = _legacy_task_image_files(Path(task.file_dir))[variant]
    else:
        image_file = None

    if image_file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page image does not exist",
        )
    return _serve_image_file(image_file)


@router.get(
    "/tasks/{task_id}/image",
    summary="Get task first page image (compatibility interface)",
)
def get_task_image(
    task_id: str,
    variant: str = "original",
    db: Session = Depends(get_db),
):
    if variant not in {"original", "corrected"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image variant",
        )

    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task or not task.file_dir:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task does not exist",
        )

    first_page = (
        db.query(TaskPage)
        .filter(TaskPage.task_id == task_id)
        .order_by(TaskPage.page_index)
        .first()
    )
    image_file = (
        _page_image_files(first_page)[variant]
        if first_page is not None
        else _legacy_task_image_files(Path(task.file_dir))[variant]
    )
    if image_file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image does not exist",
        )
    return _serve_image_file(image_file)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="Query historical task list for current IP",
)
def list_tasks(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    ip = get_client_ip(request)
    query = db.query(Task).filter(Task.ip == ip).order_by(Task.created_at.desc())
    total = query.count()
    tasks = query.offset((page - 1) * page_size).limit(page_size).all()
    items = [
        _build_task_response(db, task, include_page_results=False)
        for task in tasks
    ]
    return TaskListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )
