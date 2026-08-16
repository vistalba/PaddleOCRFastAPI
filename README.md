# PaddleOCRFastAPI

![GitHub](https://img.shields.io/github/license/cgcel/PaddleOCRFastAPI)

[中文](./README_CN.md)

A Paddle OCR Web API based on `FastAPI`.

## Current Version

- PaddleOCR 3.7.0
- PaddlePaddle 3.3.0
- Default model: PP-OCRv6_small, with medium available
- Windows uses the CUDA 11.8 GPU package; non-Windows systems use the CPU package

## Features

- [x] Local path image recognition
- [x] Base64 data recognition
- [x] Upload file recognition
- [x] Single-file, multi-page PDF recognition
- [x] Ordered multi-image tasks
- [x] Per-page native PDF text or OCR routing
- [x] Per-page task results, page images, and partial failure handling
- [x] Retry failed tasks with a server-enforced cooldown
- [x] Azure Document Intelligence compatibility layer

## Azure Document Intelligence Compatibility

The API provides Azure Document Intelligence-compatible endpoints for integration with Paperless-ngx and other systems. The wire contract matches the `azure-ai-documentintelligence` 1.0.2 Python SDK (the version Paperless-ngx pins), so both PDFs and images are supported.

### Endpoints

**POST `/documentintelligence/documentModels/{model_id}:analyze`**
- Submit a PDF or image (PNG, JPEG, TIFF, BMP, GIF, WebP) for OCR analysis
- Request body: JSON `{"base64Source": "<base64>"}` (SDK format) or multipart `file` (legacy)
- Any `model_id` is accepted (Paperless-ngx sends `prebuilt-read`)
- Returns `202 Accepted` with `Operation-Location` header
- Parameters: `api-version` (query parameter, default: `2024-11-30`)

**GET `/documentintelligence/operations/{task_id}`**
- Poll task status and retrieve results
- Returns `{"status": "running"}` while processing
- Returns `{"status": "failed", "error": {"code", "message"}}` on failure
- Returns full Azure-compatible JSON (`analyzeResult` with `content`, `pages`, `stringIndexType`) when completed

**GET `/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}/pdf`**
- Download the searchable archive PDF (SDK path; `result_id` equals the task id)

**GET `/documentintelligence/operations/{task_id}/pdf`**
- Legacy archive download path, kept for backward compatibility

### Usage Example

```bash
# Submit a PDF for analysis (SDK format)
BASE64=$(base64 -w 0 document.pdf)
curl -X POST "http://localhost:8000/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30" \
  -H "Content-Type: application/json" \
  -d "{\"base64Source\": \"$BASE64\"}"

# Poll status
curl "http://localhost:8000/documentintelligence/operations/{task_id}?api-version=2024-11-30"

# Download searchable PDF
curl "http://localhost:8000/documentintelligence/documentModels/prebuilt-read/analyzeResults/{task_id}/pdf" -o output.pdf
```

The legacy multipart format also works:

```bash
curl -X POST "http://localhost:8000/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30" \
  -F "file=@document.pdf"
```

### Paperless-ngx Integration

Configure Paperless-ngx's remote OCR provider to use the Azure-compatible endpoint:

```env
PAPERLESS_REMOTE_OCR_ENGINE=azureai
PAPERLESS_REMOTE_OCR_ENDPOINT=http://your-server:8000
PAPERLESS_REMOTE_OCR_API_KEY=unused
```

The wire contract matches the `azure-ai-documentintelligence==1.0.2` SDK
exactly (the version Paperless-ngx resolves), including the
`poller.details["operation_id"]` lookup: the SDK's
`AnalyzeDocumentLROPoller` parses the last path segment of the
`Operation-Location` URL, which is this API's task id.

To verify end-to-end with the real SDK (it replays Paperless-ngx's exact
call sequence):

```bash
pip install azure-ai-documentintelligence==1.0.2
python scripts/sdk_smoke_test.py ./test.pdf http://your-server:8000
```

### Force OCR Configuration

By default, the Azure compatibility layer uses native PDF text if available. To force OCR on all PDFs:

```dotenv
FORCE_OCR_FOR_AZURE=true
```

When `FORCE_OCR_FOR_AZURE=true`:
- All pages are processed with OCR regardless of existing text
- Any pre-existing text layer in the input PDF is **removed** and replaced with the PaddleOCR text layer
- The output PDF contains exactly **one** text layer (the OCR-generated one)
- Images, graphics, and visual content are preserved unchanged

### Searchable PDF Output

The downloadable PDF (`/documentintelligence/operations/{task_id}/pdf`) contains:
- Original page content (images, graphics) unchanged
- Invisible text layer with correct bounding boxes in PDF point coordinates (72 DPI)
- Selectable and searchable text via `pdftotext` or PDF viewers

### Test Script

A test script is included to verify the full pipeline. It replays the exact
call sequence the Azure SDK performs (JSON `base64Source` submit, poll,
`analyzeResults` PDF download) and accepts PDFs and images:

```bash
python scripts/analyze_boxes.py ./test.pdf
python scripts/analyze_boxes.py ./scan.png
```

The script submits the document, polls for completion, downloads the
searchable PDF (saved as `<name>_ocr.pdf`), and prints the per-page
box/coordinate analysis.

### Unit Tests

The Azure compatibility layer has a unit test suite that stubs the internal
PaddleOCR API in-process (no OCR models or running server required):

```bash
uv run pytest tests/test_azure_api.py
```

### OCR Quality Configuration

The OCR quality can be configured via the `OCR_DPI` environment variable:

```dotenv
OCR_DPI=300  # Default: 300 DPI (high quality)
```

**Available values**: 72-600 DPI (default: 300)

**Impact**:
- Higher DPI = better OCR accuracy but slower processing
- Scale factor = `OCR_DPI / 72` (PDF native DPI is 72)
- 300 DPI → 4.17x scale (recommended by PaddleOCR)
- 150 DPI → 2.08x scale (faster, good quality)

**Example**:
```bash
# Run with 150 DPI for faster processing
docker run -e OCR_DPI=150 vistalba/paddleocrfastapi:test-azureai

# Run with 300 DPI for maximum accuracy
docker run -e OCR_DPI=300 vistalba/paddleocrfastapi:test-azureai
```

## Image and PDF Tasks

`POST /ocr/tasks` remains a single-file upload endpoint:

- An image creates a one-page PP-OCRv6 task.
- A PDF creates one task containing all PDF pages.
- Pages with sufficient native PDF text use the text layer directly.
- Other pages are rendered to PNG and processed with PP-OCRv6.
- `GET /ocr/tasks/{task_id}` returns per-page state and results in `pages`.
- `GET /ocr/tasks/{task_id}/pages/{page_index}/image` returns a page image.
- Fully failed tasks can reuse their original files through
  `POST /ocr/tasks/{task_id}/retry` after the cooldown. The task response exposes
  `retry_after_seconds`, and the server enforces the wait period.

Use `POST /ocr/tasks/multi-image` for an ordered multi-image task:

- Send images as repeated multipart `files` fields; upload order is page order.
- This endpoint accepts images only. Continue to send a single PDF to
  `/ocr/tasks`.
- All images share the task upload-size limit, and the task accepts up to
  `MAX_MULTI_IMAGE_PAGES` images.
- Document preprocessing, when enabled, is applied to each image page.

The limits are configurable in `.env`:

```dotenv
MAX_UPLOAD_SIZE_MB=50
MAX_PDF_PAGES=50
MAX_MULTI_IMAGE_PAGES=50
PDF_RENDER_SCALE=2.0
PDF_MAX_RENDER_PIXELS=40000000
PDF_NATIVE_TEXT_MIN_CHARS=20
RETRY_COOLDOWN_SECONDS=60
```

See
[docs/PDF_MULTIPAGE_IMPLEMENTATION_PLAN.md](docs/PDF_MULTIPAGE_IMPLEMENTATION_PLAN.md)
for the detailed design and progress checklist.

## Deployment

### Direct Deployment

1. Copy the project to the deployment path

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *The master branch is the most recent version of PaddleOCR supported by the project. To install a specific version, clone the branch with the corresponding version number.*

2. Install `uv`, then sync dependencies from `pyproject.toml` and `uv.lock`

   ```shell
   uv sync --frozen
   ```

   > Dependencies are defined by `pyproject.toml` and `uv.lock`. Do not use
   > the old `requirements.txt`.

3. Copy the environment variable template:

   ```shell
   cp .env.example .env
   ```

   For a Windows + NVIDIA GPU target, set:

   ```dotenv
   OCR_MODEL_TIER=small
   OCR_DEVICE=gpu:0
   ```

4. Download project-local models on an online preparation machine:

   ```shell
   uv run --frozen python -m scripts.prepare_models --with-doc-preprocessor
   ```

5. Run FastAPI

   ```shell
   uv run --frozen uvicorn main:app --host 0.0.0.0
   ```

For the full PP-OCRv6_small, Windows RTX 2080, CUDA version choice, and later
PP-OCRv6_medium switching notes, see
[docs/PP_OCRV6_WINDOWS_GPU.md](docs/PP_OCRV6_WINDOWS_GPU.md).

### Docker Deployment

Test completed in `Centos 7`, `Ubuntu 20.04`, `Ubuntu 22.04`, `Windows 10`, `Windows 11`, requires `Docker` to be installed.

> The current Dockerfile has been migrated to PP-OCRv6_small and uses the Linux
> CPU PaddlePaddle package by default. For Windows + RTX 2080, the direct uv
> deployment above is recommended. Docker GPU would require a CUDA base image
> and the Linux GPU PaddlePaddle dependency to be configured separately.

1. Copy the project to the deployment path

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *The master branch is the most recent version of PaddleOCR supported by the project. To install a specific version, clone the branch with the corresponding version number.*

2. Building a Docker Image

   ```shell
   cd PaddleOCRFastAPI
   # Use the host network. Add HTTP_PROXY / HTTPS_PROXY build args if needed.
   docker build -t paddleocrfastapi:latest --network host .
   ```

   > During image build,
   > `uv run --no-sync python -m scripts.prepare_models --with-doc-preprocessor`
   > prepares PP-OCRv6 models under `/app/.paddlex/official_models`. If the
   > Docker build environment has no Internet access, prepare `.paddlex/` on an
   > online machine first and make sure it is included in the Docker build
   > context.

3. Edit `docker-compose.yml`

    ```yaml
    services:

       PaddleOCR:
          build:
             context: .
             args:
                OCR_MODEL_TIER: small
          container_name: paddle_ocr_api # Custom container name
          image: paddleocrfastapi:latest # Custom image name and tag
          environment:
             - TZ=Asia/Hong_Kong
             - OCR_MODEL_TIER=small
             - OCR_DEVICE=cpu
             - OCR_MODEL_DIR=/app/.paddlex/official_models
             - PADDLE_PDX_CACHE_HOME=/app/.paddlex
             - PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
             - PADDLE_PDX_MODEL_SOURCE=bos
             - PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0
          ports:
             - "8000:8000" # Customize the service exposure port, 8000 is the default FastAPI port, do not modify
          restart: unless-stopped
    ```

4. Create the Docker container and run

   ```shell
   docker compose up -d --build
   ```

5. Swagger Page at `localhost:<port>/docs`

To test PP-OCRv6_medium in Docker, change both
`build.args.OCR_MODEL_TIER` and `environment.OCR_MODEL_TIER` in
`docker-compose.yml` to `medium`, then rebuild the image.

### Offline deployment notes

If you plan to move this service to an isolated intranet machine, prepare it like this:

1. On a machine with Internet access, install dependencies and download or preload the OCR models first.

2. Make sure the project directory contains at least the following items:

   ```text
   .paddlex/official_models/
   ├─ PP-OCRv6_small_det/
   ├─ PP-OCRv6_small_rec/
   └─ PP-LCNet_x1_0_textline_ori/
   .env
   pyproject.toml
   uv.lock
   ```

3. `.paddlex` is ignored by Git and must be explicitly included in the project
   archive. A macOS `.venv` and macOS uv wheel cache cannot be reused on
   Windows; the model directories are platform independent.

4. Populate the uv cache on an online Windows staging machine or provide an
   internal Python package index, then run on the offline target:

   ```shell
   uv sync --frozen --offline
   ```

## Language support

PP-OCRv6 uses one model for Chinese, English, Japanese, and many Latin-script
languages. The legacy `OCR_LANGUAGE` model switch is no longer used.

## Screenshots
API Docs: `/docs`

![Swagger](https://raw.githubusercontent.com/cgcel/PaddleOCRFastAPI/dev/screenshots/Swagger.png)

## Todo

- [x] PP-OCRv6 small / medium switching
- [x] Conditional Windows NVIDIA GPU dependency
- [x] Image URL recognition

## License

**PaddleOCRFastAPI** is licensed under the MIT license. Refer to [LICENSE](https://github.com/cgcel/PaddleOCRFastAPI/blob/master/LICENSE) for more information.
