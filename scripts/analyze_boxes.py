#!/usr/bin/env python3
"""
Analyze bounding boxes and coordinate transformation from a PaddleOCR task response.

This script replays the exact call sequence the azure-ai-documentintelligence
SDK (and Paperless-ngx's azureai provider) performs against the Azure
compatibility layer:

1. POST /documentintelligence/documentModels/prebuilt-read:analyze
   with a JSON body {"base64Source": "<base64>"} (the SDK format)
2. GET  /documentintelligence/operations/{task_id} until succeeded
3. GET  /documentintelligence/documentModels/prebuilt-read/analyzeResults/{task_id}/pdf

It accepts PDFs and images (PNG, JPEG, TIFF, BMP, GIF, WebP), downloads the
searchable archive PDF, and prints the per-page box/coordinate analysis.

Usage:
    python scripts/analyze_boxes.py ./test.pdf
    python scripts/analyze_boxes.py ./scan.png
"""

import base64
import json
import os
import sys
import time

import requests

# Configuration
BASE_URL = "http://localhost:8000"
DOCUMENT_FILE = "./test.pdf"
API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-read"  # the model id Paperless-ngx sends


def submit_document(document_path):
    """Submit a PDF or image to the Azure-compatible endpoint (SDK format)."""
    print(f"Submitting {document_path}...")

    with open(document_path, "rb") as f:
        document_bytes = f.read()

    response = requests.post(
        f"{BASE_URL}/documentintelligence/documentModels/{MODEL_ID}:analyze"
        f"?api-version={API_VERSION}",
        json={"base64Source": base64.b64encode(document_bytes).decode("ascii")},
    )

    if response.status_code != 202:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    # Extract task ID from the Operation-Location header
    operation_location = response.headers.get("Operation-Location")
    task_id = operation_location.split("/")[-1].split("?")[0]
    print(f"Task ID: {task_id}")

    return task_id


def poll_task(task_id):
    """Poll the operation until it completes."""
    print("Waiting for task to complete...")

    while True:
        response = requests.get(
            f"{BASE_URL}/documentintelligence/operations/{task_id}"
            f"?api-version={API_VERSION}"
        )

        if response.status_code != 200:
            print(f"Error polling: {response.status_code}")
            return None

        data = response.json()
        status = data.get("status")

        if status == "running":
            print("  Still processing...")
            time.sleep(2)
        elif status == "succeeded":
            print("Task completed!")
            return data
        else:
            error = data.get("error", {})
            print(f"Task failed with status {status}: {error.get('message', 'unknown error')}")
            return None


def download_pdf(task_id, input_filename):
    """Download the searchable archive PDF after processing completes."""
    base, _ = os.path.splitext(input_filename)
    output_path = f"{base}_ocr.pdf"

    print(f"\nDownloading archive PDF...")
    response = requests.get(
        f"{BASE_URL}/documentintelligence/documentModels/{MODEL_ID}"
        f"/analyzeResults/{task_id}/pdf"
    )

    if response.status_code != 200:
        print(f"PDF download failed: {response.status_code}")
        return None

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"PDF saved to: {output_path}")
    return output_path


def main():
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
        document_path = DOCUMENT_FILE

    # Submit document
    task_id = submit_document(document_path)
    if not task_id:
        return

    # Poll until complete
    task_data = poll_task(task_id)
    if not task_data:
        return

    # Download the searchable PDF
    download_pdf(task_id, document_path)

    # Get task details from the PaddleOCR endpoint
    task_response = requests.get(f"{BASE_URL}/ocr/tasks/{task_id}")
    if task_response.status_code != 200:
        print(f"Could not get task details: {task_response.status_code}")
        return

    task_details = task_response.json()

    print(f"\nTask Status: {task_details.get('status')}")
    print(f"File Type: {task_details.get('file_type')}")
    print(f"Page Count: {task_details.get('page_count')}")
    print()

    for page in task_details.get("pages", [])[:3]:  # Show first 3 pages
        page_num = page.get("page_index", 0) + 1
        status = page.get("status")

        print(f"\nPage {page_num}:")
        print(f"   Status: {status}")

        if status == "failed":
            print(f"   Error: {page.get('error_msg', 'Unknown error')}")
            continue

        # Image dimensions (pixels)
        image_width = page.get("width")
        image_height = page.get("height")
        print(f"   Image Size: {image_width} x {image_height} pixels")

        # PDF dimensions (points) - only present for PDF tasks
        pdf_width = page.get("pdf_width_pts")
        pdf_height = page.get("pdf_height_pts")
        if pdf_width and pdf_height:
            print(f"   PDF Size: {pdf_width} x {pdf_height} points (72 DPI)")
        else:
            print("   PDF Size: N/A (image task)")

        # Render scale
        render_scale = page.get("render_scale")
        print(f"   Render Scale: {render_scale:.4f}x" if render_scale else "   Render Scale: N/A")

        # Verify scale calculation (PDF tasks only)
        if pdf_width and image_width and render_scale:
            expected_image_width = pdf_width * render_scale
            print(f"   Scale Verification: {pdf_width} x {render_scale:.4f} = {expected_image_width:.1f} (actual: {image_width})")

        # Lines with bounding boxes
        lines = page.get("lines", [])
        print(f"   Lines: {len(lines)}")

        if lines:
            print(f"\n   Sample Lines with Bounding Boxes:")
            for i, line in enumerate(lines[:5]):
                text = line.get("text", "")
                box = line.get("box", [])

                if box:
                    # Calculate box dimensions
                    if len(box) >= 4:
                        x_min = min(box[0], box[2])
                        x_max = max(box[0], box[2])
                        y_min = min(box[1], box[3])
                        y_max = max(box[1], box[3])
                        box_width = x_max - x_min
                        box_height = y_max - y_min

                        print(f"   Line {i+1}:")
                        print(f"      Text: '{text[:50]}'")
                        print(f"      Box: {box}")
                        print(f"      Dimensions: {box_width:.1f} x {box_height:.1f}")
                        print(f"      Position: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")
                else:
                    print(f"   Line {i+1}: '{text[:50]}' (no box)")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful_pages = sum(1 for p in task_details.get("pages", []) if p.get("status") == "done")
    failed_pages = sum(1 for p in task_details.get("pages", []) if p.get("status") == "failed")

    print(f"Successful Pages: {successful_pages}")
    print(f"Failed Pages: {failed_pages}")

    if successful_pages > 0:
        sample_page = next((p for p in task_details.get("pages", []) if p.get("status") == "done"), None)
        if sample_page:
            render_scale = sample_page.get("render_scale", 0)
            pdf_width = sample_page.get("pdf_width_pts")
            pdf_height = sample_page.get("pdf_height_pts")
            print(f"\nCoordinate transformation is working!")
            if pdf_width and pdf_height:
                print(f"   PDF dimensions: {pdf_width} x {pdf_height} points")
                print(f"   Bounding boxes are in PDF point coordinates (72 DPI)")
            else:
                print(f"   Image task: bounding boxes are in pixel coordinates")
            if render_scale:
                print(f"   Render scale: {render_scale:.4f}x")
    else:
        print(f"\nAll pages failed. Check error messages above.")

    # Save JSON
    output_file = "task_analysis_full.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "azure_response": task_data,
                "paddleocr_response": task_details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nFull JSON saved to: {output_file}")


if __name__ == "__main__":
    main()
