import asyncio
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from database import Base
from models.TaskModel import Task, TaskPage
from routers import tasks


class TaskPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_stale_rule_result_version_is_rebuilt(self):
        page = TaskPage(
            task_id="stale-rule",
            page_index=0,
            status="done",
            processing_method="ocr",
            ocr_result="[]",
            rule_result=json.dumps({"version": 1, "segments": []}),
        )
        rebuilt = {
            "version": tasks.ORGANIZATION_RESULT_VERSION,
            "method": "coordinate_rules",
            "segments": [],
            "text": "",
        }
        with patch.object(
            tasks,
            "_build_rule_result_for_page",
            return_value=rebuilt,
        ) as builder:
            self.assertTrue(tasks._ensure_rule_result(page))
            self.assertFalse(tasks._ensure_rule_result(page))
        builder.assert_called_once_with(page)
        self.assertEqual(json.loads(page.rule_result), rebuilt)

    def test_old_image_task_is_exposed_as_one_page(self):
        task_dir = self.root / "legacy"
        task_dir.mkdir()
        Image.new("RGB", (120, 80), "white").save(task_dir / "original.png")
        ocr_result = [
            {
                "rec_texts": ["legacy"],
                "rec_scores": [0.99],
                "rec_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                "rec_boxes": [[0, 0, 10, 10]],
            }
        ]
        db = self.session_factory()
        try:
            task = Task(
                task_id="legacy-task",
                ip="127.0.0.1",
                status="done",
                original_filename="legacy.png",
                file_dir=str(task_dir),
                ocr_result=json.dumps(ocr_result),
            )
            db.add(task)
            db.commit()
            response = tasks._build_task_response(db, task)
            self.assertEqual(response.file_type, "image")
            self.assertEqual(response.page_count, 1)
            self.assertEqual(response.pages[0].ocr_result[0]["rec_texts"], ["legacy"])
        finally:
            db.close()

    def test_mixed_pages_complete_with_page_results(self):
        task_dir = self.root / "mixed"
        first_page_dir = task_dir / "pages" / "page_0001"
        second_page_dir = task_dir / "pages" / "page_0002"
        first_page_dir.mkdir(parents=True)
        second_page_dir.mkdir(parents=True)
        source_path = task_dir / "original.pdf"
        source_path.write_bytes(b"%PDF-1.4\n% test")
        first_image = first_page_dir / "original.png"
        second_image = second_page_dir / "original.png"
        Image.new("RGB", (120, 80), "white").save(first_image)
        Image.new("RGB", (120, 80), "white").save(second_image)

        db = self.session_factory()
        db.add(
            Task(
                task_id="mixed-task",
                ip="127.0.0.1",
                status="queued",
                original_filename="mixed.pdf",
                file_dir=str(task_dir),
            )
        )
        db.commit()
        db.close()

        prepared = {
            "file_type": "pdf",
            "pages": [
                {
                    "page_index": 0,
                    "processing_method": "native_text",
                    "original_image_path": str(first_image),
                    "native_text": "Native page text with enough characters.",
                    "width": 120,
                    "height": 80,
                },
                {
                    "page_index": 1,
                    "processing_method": "ocr",
                    "original_image_path": str(second_image),
                    "native_text": None,
                    "width": 120,
                    "height": 80,
                },
            ],
        }
        fake_ocr_result = [
            {
                "input_path": None,
                "rec_texts": ["scanned page"],
                "rec_scores": [0.95],
                "rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_boxes": [[0, 0, 20, 10]],
                "ocr_image_variant": "original",
            }
        ]

        def fake_prepare(_source_path, _force_ocr=False):
            return prepared

        def fake_ocr(_image_path, _use_doc_preprocessor):
            return {
                "ocr_result": fake_ocr_result,
                "ocr_image_variant": "original",
                "corrected_image_path": None,
            }

        test_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with (
                patch.object(tasks, "SessionLocal", self.session_factory),
                patch.object(tasks, "_ocr_pool", test_pool),
                patch.object(tasks, "prepare_document_file", fake_prepare),
                patch.object(tasks, "run_ocr_file", fake_ocr),
            ):
                asyncio.run(
                    tasks._process_ocr_async("mixed-task", source_path)
                )
        finally:
            test_pool.shutdown(wait=True)

        db = self.session_factory()
        try:
            task = db.query(Task).filter(Task.task_id == "mixed-task").one()
            pages = (
                db.query(TaskPage)
                .filter(TaskPage.task_id == "mixed-task")
                .order_by(TaskPage.page_index)
                .all()
            )
            self.assertEqual(task.status, "done")
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0].processing_method, "native_text")
            self.assertEqual(pages[0].status, "done")
            self.assertEqual(pages[1].processing_method, "ocr")
            self.assertEqual(
                json.loads(pages[1].ocr_result)[0]["rec_texts"],
                ["scanned page"],
            )
            response = tasks._build_task_response(db, task)
            self.assertEqual(response.page_count, 2)
            self.assertEqual(response.completed_pages, 2)
        finally:
            db.close()

    def test_preparation_failure_is_isolated_to_one_page(self):
        task_dir = self.root / "partial"
        first_page_dir = task_dir / "pages" / "page_0001"
        first_page_dir.mkdir(parents=True)
        source_path = task_dir / "original.pdf"
        source_path.write_bytes(b"%PDF-1.4\n% test")
        first_image = first_page_dir / "original.png"
        Image.new("RGB", (120, 80), "white").save(first_image)

        db = self.session_factory()
        db.add(
            Task(
                task_id="partial-task",
                ip="127.0.0.1",
                status="queued",
                original_filename="partial.pdf",
                file_dir=str(task_dir),
            )
        )
        db.commit()
        db.close()

        prepared = {
            "file_type": "pdf",
            "pages": [
                {
                    "page_index": 0,
                    "processing_method": "native_text",
                    "original_image_path": str(first_image),
                    "native_text": "The first page remains readable.",
                    "width": 120,
                    "height": 80,
                    "preparation_error": None,
                },
                {
                    "page_index": 1,
                    "processing_method": "ocr",
                    "original_image_path": None,
                    "native_text": None,
                    "width": None,
                    "height": None,
                    "preparation_error": "PDF 第 2 页渲染失败",
                },
            ],
        }

        test_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with (
                patch.object(tasks, "SessionLocal", self.session_factory),
                patch.object(tasks, "_ocr_pool", test_pool),
                patch.object(
                    tasks,
                    "prepare_document_file",
                    lambda _source_path, _force_ocr=False: prepared,
                ),
            ):
                asyncio.run(
                    tasks._process_ocr_async("partial-task", source_path)
                )
        finally:
            test_pool.shutdown(wait=True)

        db = self.session_factory()
        try:
            task = db.query(Task).filter(Task.task_id == "partial-task").one()
            response = tasks._build_task_response(db, task)
            self.assertEqual(task.status, "done")
            self.assertEqual(response.completed_pages, 1)
            self.assertEqual(response.failed_pages, 1)
            self.assertEqual(response.pages[1].status, "failed")
            self.assertIn("第 2 页", response.pages[1].error_msg)
        finally:
            db.close()

    def test_multi_image_sources_become_ordered_task_pages(self):
        task_dir = self.root / "multi-image"
        first_source_dir = task_dir / "sources" / "page_0001"
        second_source_dir = task_dir / "sources" / "page_0002"
        first_source_dir.mkdir(parents=True)
        second_source_dir.mkdir(parents=True)
        first_source = first_source_dir / "original.png"
        second_source = second_source_dir / "original.jpg"
        Image.new("RGB", (90, 140), "blue").save(first_source)
        Image.new("RGB", (120, 80), "red").save(second_source)

        db = self.session_factory()
        db.add(
            Task(
                task_id="multi-image-task",
                ip="127.0.0.1",
                status="queued",
                original_filename="first.png 等 2 张图片",
                file_dir=str(task_dir),
            )
        )
        db.commit()
        db.close()

        def fake_ocr(image_path, _use_doc_preprocessor):
            page_name = Path(image_path).parent.name
            return {
                "ocr_result": [
                    {
                        "input_path": None,
                        "rec_texts": [page_name],
                        "rec_scores": [0.99],
                        "rec_polys": [
                            [[0, 0], [20, 0], [20, 10], [0, 10]]
                        ],
                        "rec_boxes": [[0, 0, 20, 10]],
                        "ocr_image_variant": "original",
                    }
                ],
                "ocr_image_variant": "original",
                "corrected_image_path": None,
            }

        test_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with (
                patch.object(tasks, "SessionLocal", self.session_factory),
                patch.object(tasks, "_ocr_pool", test_pool),
                patch.object(tasks, "run_ocr_file", fake_ocr),
            ):
                source_paths = tasks._task_source_files(task_dir)
                self.assertEqual(source_paths, [first_source, second_source])
                asyncio.run(
                    tasks._process_ocr_async(
                        "multi-image-task",
                        tuple(source_paths),
                    )
                )
        finally:
            test_pool.shutdown(wait=True)

        db = self.session_factory()
        try:
            task = (
                db.query(Task)
                .filter(Task.task_id == "multi-image-task")
                .one()
            )
            response = tasks._build_task_response(db, task)
            self.assertEqual(task.status, "done")
            self.assertEqual(response.page_count, 2)
            self.assertEqual(response.completed_pages, 2)
            self.assertEqual(
                [(page.width, page.height) for page in response.pages],
                [(90, 140), (120, 80)],
            )
            self.assertEqual(
                response.pages[0].ocr_result[0]["rec_texts"],
                ["page_0001"],
            )
            self.assertEqual(
                response.pages[1].ocr_result[0]["rec_texts"],
                ["page_0002"],
            )
        finally:
            db.close()

    def test_upload_size_and_file_signature_validation(self):
        oversized = UploadFile(
            filename="oversized.png",
            file=io.BytesIO(b"x" * (1024 * 1024 + 1)),
        )
        output_path = self.root / "oversized.png"
        with patch.object(tasks, "MAX_UPLOAD_SIZE_MB", 1):
            with self.assertRaises(HTTPException) as oversized_error:
                asyncio.run(tasks._save_upload_file(oversized, output_path))
        self.assertEqual(oversized_error.exception.status_code, 413)

        invalid_pdf = self.root / "invalid.pdf"
        invalid_pdf.write_bytes(b"not a PDF")
        with self.assertRaises(HTTPException) as pdf_error:
            tasks._validate_uploaded_file(invalid_pdf, ".pdf")
        self.assertEqual(pdf_error.exception.status_code, 400)

        invalid_image = self.root / "invalid.png"
        invalid_image.write_bytes(b"not an image")
        with self.assertRaises(HTTPException) as image_error:
            tasks._validate_uploaded_file(invalid_image, ".png")
        self.assertEqual(image_error.exception.status_code, 400)

    def test_task_page_unique_constraint(self):
        db = self.session_factory()
        try:
            db.add(
                Task(
                    task_id="unique-task",
                    ip="127.0.0.1",
                    status="queued",
                )
            )
            db.commit()
            db.add_all(
                [
                    TaskPage(
                        task_id="unique-task",
                        page_index=0,
                        processing_method="ocr",
                    ),
                    TaskPage(
                        task_id="unique-task",
                        page_index=0,
                        processing_method="ocr",
                    ),
                ]
            )
            with self.assertRaises(IntegrityError):
                db.commit()
        finally:
            db.rollback()
            db.close()

    def test_failed_task_cooldown_and_retry(self):
        task_dir = self.root / "retry"
        task_dir.mkdir()
        source_path = task_dir / "original.png"
        Image.new("RGB", (120, 80), "white").save(source_path)

        db = self.session_factory()
        task = Task(
            task_id="retry-task",
            ip="127.0.0.1",
            status="failed",
            original_filename="retry.png",
            file_dir=str(task_dir),
            error_msg="temporary failure",
        )
        db.add(task)
        db.commit()

        with patch.object(tasks, "RETRY_COOLDOWN_SECONDS", 60):
            tasks._write_retry_state(
                task,
                datetime.now(timezone.utc).replace(tzinfo=None)
                + timedelta(seconds=60),
            )
            response = tasks._build_task_response(db, task)
            self.assertGreater(response.retry_after_seconds, 0)
            self.assertLessEqual(response.retry_after_seconds, 60)
            self.assertIsNotNone(response.retry_available_at)

            with self.assertRaises(HTTPException) as cooldown_error:
                tasks._reset_failed_task_for_retry(db, task)
            self.assertEqual(cooldown_error.exception.status_code, 429)
            self.assertIn(
                "Retry-After",
                cooldown_error.exception.headers,
            )

            tasks._write_retry_state(
                task,
                datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(seconds=1),
            )
            source_paths = tasks._reset_failed_task_for_retry(db, task)

        self.assertEqual(source_paths, (source_path,))
        self.assertEqual(task.status, "queued")
        self.assertIsNone(task.error_msg)
        self.assertFalse(
            (task_dir / tasks.RETRY_STATE_FILENAME).exists()
        )
        db.close()

        def fake_ocr(_image_path, _use_doc_preprocessor):
            return {
                "ocr_result": [
                    {
                        "input_path": None,
                        "rec_texts": ["retried"],
                        "rec_scores": [0.99],
                        "rec_polys": [
                            [[0, 0], [20, 0], [20, 10], [0, 10]]
                        ],
                        "rec_boxes": [[0, 0, 20, 10]],
                        "ocr_image_variant": "original",
                    }
                ],
                "ocr_image_variant": "original",
                "corrected_image_path": None,
            }

        test_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with (
                patch.object(tasks, "SessionLocal", self.session_factory),
                patch.object(tasks, "_ocr_pool", test_pool),
                patch.object(tasks, "run_ocr_file", fake_ocr),
            ):
                asyncio.run(
                    tasks._process_ocr_async(
                        "retry-task",
                        source_paths,
                    )
                )
        finally:
            test_pool.shutdown(wait=True)

        db = self.session_factory()
        try:
            retried_task = (
                db.query(Task)
                .filter(Task.task_id == "retry-task")
                .one()
            )
            retried_response = tasks._build_task_response(
                db,
                retried_task,
            )
            self.assertEqual(retried_task.status, "done")
            self.assertEqual(
                retried_response.pages[0].ocr_result[0]["rec_texts"],
                ["retried"],
            )
            self.assertEqual(retried_response.retry_after_seconds, 0)
        finally:
            db.close()

    def test_ai_organization_is_saved_and_only_loopback_can_repeat(self):
        ocr_result = [
            {
                "rec_texts": ["第一行", "第二行"],
                "rec_scores": [0.99, 0.98],
                "rec_polys": [
                    [[0, 0], [100, 0], [100, 10], [0, 10]],
                    [[0, 14], [100, 14], [100, 24], [0, 24]],
                ],
                "rec_boxes": [[0, 0, 100, 10], [0, 14, 100, 24]],
            }
        ]
        db = self.session_factory()
        db.add(
            Task(
                task_id="ai-task",
                ip="10.0.0.8",
                status="done",
                original_filename="ai.png",
                pages=[
                    TaskPage(
                        page_index=0,
                        status="done",
                        processing_method="ocr",
                        width=120,
                        height=80,
                        ocr_result=json.dumps(ocr_result),
                    )
                ],
            )
        )
        db.commit()

        first_page_result = {
            "version": 1,
            "method": "local_ai_page",
            "model_name": "test-model",
            "segments": [
                {
                    "text": "第一行第二行",
                    "source_line_indices": [0, 1],
                }
            ],
            "text": "第一行第二行",
        }
        first_boundary_result = {
            **first_page_result,
            "method": "local_ai_boundary",
            "segments": [
                {"text": "第一行", "source_line_indices": [0]},
                {"text": "第二行", "source_line_indices": [1]},
            ],
            "text": "第一行\n\n第二行",
        }

        def comparison_result(primary, page_result, boundary_result):
            return {
                **primary,
                "method": "local_ai_compare",
                "selected_variant": (
                    "page" if primary is page_result else "boundary"
                ),
                "variants": {
                    "page": {
                        "status": "done",
                        "duration_ms": 120,
                        "result": page_result,
                        "error": None,
                    },
                    "boundary": {
                        "status": "done",
                        "duration_ms": 180,
                        "result": boundary_result,
                        "error": None,
                    },
                },
            }

        first_result = comparison_result(
            first_page_result,
            first_page_result,
            first_boundary_result,
        )
        second_result = comparison_result(
            first_boundary_result,
            first_page_result,
            first_boundary_result,
        )
        remote_request = Request({
            "type": "http",
            "method": "POST",
            "path": "/ocr/tasks/ai-task/pages/0/organize",
            "headers": [],
            "query_string": b"",
            "client": ("10.0.0.8", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        })
        local_request = Request({
            **remote_request.scope,
            "client": ("127.0.0.1", 1234),
        })

        async def run_scenario():
            with (
                patch.object(
                    tasks,
                    "ai_organizer_status",
                    return_value={
                        "available": True,
                        "model_name": "test-model",
                        "reason": None,
                    },
                ),
                patch.object(
                    tasks,
                    "_run_in_ai_pool",
                    new=AsyncMock(
                        side_effect=[first_result, second_result]
                    ),
                ),
            ):
                response = await tasks.organize_task_page(
                    remote_request,
                    "ai-task",
                    0,
                    db,
                )
                self.assertEqual(response.ai_status, "done")
                with self.assertRaises(HTTPException) as repeated:
                    await tasks.organize_task_page(
                        remote_request,
                        "ai-task",
                        0,
                        db,
                    )
                self.assertEqual(
                    repeated.exception.status_code,
                    409,
                )
                repeated_response = await tasks.organize_task_page(
                    local_request,
                    "ai-task",
                    0,
                    db,
                )
                self.assertEqual(
                    len(repeated_response.ai_result["segments"]),
                    2,
                )

        asyncio.run(run_scenario())
        page = (
            db.query(TaskPage)
            .filter(TaskPage.task_id == "ai-task")
            .one()
        )
        self.assertEqual(page.ai_status, "done")
        saved_ai_result = json.loads(page.ai_result)
        self.assertEqual(len(saved_ai_result["segments"]), 2)
        self.assertEqual(
            set(saved_ai_result["variants"]),
            {"page", "boundary"},
        )
        self.assertIsNotNone(page.rule_result)
        self.assertIsNotNone(page.ai_processed_at)
        db.close()

    def test_delete_finished_task_removes_database_rows_and_files(self):
        task_dir = self.root / "deletable-task"
        task_dir.mkdir()
        (task_dir / "original.png").write_bytes(b"task data")

        db = self.session_factory()
        db.add(
            Task(
                task_id="deletable-task",
                ip="127.0.0.1",
                status="done",
                original_filename="delete.png",
                file_dir=str(task_dir),
                pages=[
                    TaskPage(
                        page_index=0,
                        status="done",
                        processing_method="ocr",
                    )
                ],
            )
        )
        db.commit()

        request = Request({
            "type": "http",
            "method": "DELETE",
            "path": "/ocr/tasks/deletable-task",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        })
        with patch.object(tasks, "UPLOAD_DIR", self.root):
            tasks.delete_task(request, "deletable-task", db)

        self.assertIsNone(
            db.query(Task)
            .filter(Task.task_id == "deletable-task")
            .first()
        )
        self.assertEqual(
            db.query(TaskPage)
            .filter(TaskPage.task_id == "deletable-task")
            .count(),
            0,
        )
        self.assertFalse(task_dir.exists())
        db.close()

    def test_delete_rejects_active_task(self):
        db = self.session_factory()
        db.add(
            Task(
                task_id="active-task",
                ip="127.0.0.1",
                status="processing",
            )
        )
        db.commit()

        request = Request({
            "type": "http",
            "method": "DELETE",
            "path": "/ocr/tasks/active-task",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        })
        with self.assertRaises(HTTPException) as active_error:
            tasks.delete_task(request, "active-task", db)

        self.assertEqual(active_error.exception.status_code, 409)
        self.assertIsNotNone(
            db.query(Task)
            .filter(Task.task_id == "active-task")
            .first()
        )
        db.close()


if __name__ == "__main__":
    unittest.main()
