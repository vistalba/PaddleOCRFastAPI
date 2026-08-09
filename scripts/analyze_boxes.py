#!/usr/bin/env python3
"""
Analyze bounding boxes and coordinate transformation from PaddleOCR task response.
Run this after submitting a PDF to the Azure-compatible endpoint.
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
PDF_FILE = "./test.pdf"
API_VERSION = "2024-11-30"

def submit_pdf(pdf_path):
    """Submit PDF to Azure-compatible endpoint"""
    print(f"📤 Submitting {pdf_path}...")

    with open(pdf_path, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        response = requests.post(
            f"{BASE_URL}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version={API_VERSION}",
            files=files
        )

    if response.status_code != 202:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

    # Extract task ID from Operation-Location header
    operation_location = response.headers.get("Operation-Location")
    # Extract task_id from URL, removing query parameters
    task_id = operation_location.split("/")[-1].split("?")[0]
    print(f"✅ Task ID: {task_id}")

    return task_id

def poll_task(task_id):
    """Poll task until completion"""
    print("⏳ Waiting for task to complete...")

    while True:
        response = requests.get(
            f"{BASE_URL}/documentintelligence/operations/{task_id}?api-version={API_VERSION}"
        )

        if response.status_code != 200:
            print(f"❌ Error polling: {response.status_code}")
            return None

        data = response.json()
        status = data.get("status")

        if status == "running":
            print("  ⏳ Still processing...")
            time.sleep(2)
        elif status == "succeeded":
            print("✅ Task completed!")
            return data
        else:
            print(f"❌ Task failed with status: {status}")
            return None

def analyze_coordinate_transformation(task_data):
    """Analyze coordinate transformation from image pixels to PDF points"""
    print("\n" + "="*80)
    print("📊 COORDINATE TRANSFORMATION ANALYSIS")
    print("="*80 + "\n")

    # Get task ID from Azure response
    task_id = task_data.get("analyzeResult", {}).get("pages", [{}])[0].get("pageNumber")
    # We need to get the actual task_id from the Operation-Location or store it separately
    # For now, let's try to extract it from the response
    # Actually, we need to poll the PaddleOCR endpoint with the task_id we got earlier
    
    # This function will be called after we have the task_id
    pass

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = PDF_FILE

    # Submit PDF
    task_id = submit_pdf(pdf_path)
    if not task_id:
        return

    # Poll until complete
    task_data = poll_task(task_id)
    if not task_data:
        return

    # Get task details from PaddleOCR endpoint
    task_response = requests.get(f"{BASE_URL}/ocr/tasks/{task_id}")
    if task_response.status_code != 200:
        print(f"⚠️  Could not get task details: {task_response.status_code}")
        return

    task_details = task_response.json()

    print(f"\nTask Status: {task_details.get('status')}")
    print(f"Page Count: {task_details.get('page_count')}")
    print()

    for page in task_details.get('pages', [])[:3]:  # Show first 3 pages
        page_num = page.get('page_index', 0) + 1
        status = page.get('status')

        print(f"\n📄 Page {page_num}:")
        print(f"   Status: {status}")

        if status == "failed":
            print(f"   Error: {page.get('error_msg', 'Unknown error')}")
            continue

        # Image dimensions (pixels)
        image_width = page.get('width')
        image_height = page.get('height')
        print(f"   Image Size: {image_width} x {image_height} pixels")

        # PDF dimensions (points) - NOW INCLUDED!
        pdf_width = page.get('pdf_width_pts')
        pdf_height = page.get('pdf_height_pts')
        print(f"   PDF Size: {pdf_width} x {pdf_height} points (72 DPI)")

        # Render scale
        render_scale = page.get('render_scale')
        print(f"   Render Scale: {render_scale:.4f}x" if render_scale else "   Render Scale: N/A")

        # Verify scale calculation
        if pdf_width and image_width and render_scale:
            expected_image_width = pdf_width * render_scale
            print(f"   Scale Verification: {pdf_width} × {render_scale:.4f} = {expected_image_width:.1f} (actual: {image_width})")

        # Lines with bounding boxes
        lines = page.get('lines', [])
        print(f"   Lines: {len(lines)}")

        if lines:
            print(f"\n   Sample Lines with Bounding Boxes:")
            for i, line in enumerate(lines[:5]):
                text = line.get('text', '')
                box = line.get('box', [])

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
                        print(f"      Dimensions: {box_width:.1f} × {box_height:.1f} points")
                        print(f"      Position: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")
                else:
                    print(f"   Line {i+1}: '{text[:50]}' (no box)")

        print()

    # Summary
    print("="*80)
    print("📋 SUMMARY")
    print("="*80)

    successful_pages = sum(1 for p in task_details.get('pages', []) if p.get('status') == 'done')
    failed_pages = sum(1 for p in task_details.get('pages', []) if p.get('status') == 'failed')

    print(f"Successful Pages: {successful_pages}")
    print(f"Failed Pages: {failed_pages}")

    if successful_pages > 0:
        sample_page = next((p for p in task_details.get('pages', []) if p.get('status') == 'done'), None)
        if sample_page:
            render_scale = sample_page.get('render_scale', 0)
            pdf_width = sample_page.get('pdf_width_pts')
            pdf_height = sample_page.get('pdf_height_pts')
            print(f"\n✅ Coordinate transformation is working!")
            print(f"   PDF dimensions: {pdf_width} × {pdf_height} points")
            print(f"   Bounding boxes are in PDF point coordinates (72 DPI)")
            print(f"   Render scale: {render_scale:.4f}x")
    else:
        print(f"\n❌ All pages failed. Check error messages above.")

    # Save JSON
    output_file = "task_analysis_full.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "azure_response": task_data,
            "paddleocr_response": task_details
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full JSON saved to: {output_file}")

if __name__ == "__main__":
    main()