import asyncio
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

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

        def fake_prepare(_source_path):
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
                    lambda _source_path: prepared,
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


if __name__ == "__main__":
    unittest.main()
