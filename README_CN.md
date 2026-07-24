# PaddleOCRFastAPI

一个基于 `FastAPI` 的 Paddle OCR Web API。

## 当前版本

- PaddleOCR 3.7.0
- PaddlePaddle 3.3.0
- 默认模型 PP-OCRv6_small（可选 medium）
- Windows 使用 CUDA 11.8 GPU 包，非 Windows 使用 CPU 包

## 接口功能

- [x] 局域网范围内路径图片 OCR 识别
- [x] Base64 数据识别
- [x] 上传文件识别
- [x] 单个 PDF 多页识别
- [x] 多张图片按自定义顺序组成一个任务
- [x] PDF 原生文本页与扫描页自动分流
- [x] 异步任务逐页结果、页面图片与部分失败
- [x] 失败任务冷却后复用原文件重试

## 图片与 PDF 任务

异步接口 `POST /ocr/tasks` 保持单文件上传：

- 上传图片时，任务包含 1 页并执行 PP-OCRv6。
- 上传 PDF 时，任务包含 PDF 的全部页面。
- PDF 页面存在足够原生文本时直接使用文本层。
- 其余 PDF 页面渲染为 PNG 后执行 PP-OCRv6。
- `GET /ocr/tasks/{task_id}` 的 `pages` 字段返回逐页状态与结果。
- `GET /ocr/tasks/{task_id}/pages/{page_index}/image` 返回指定页图。
- 完全失败的任务可在冷却结束后通过
  `POST /ocr/tasks/{task_id}/retry` 复用原文件重新执行；服务端会返回
  `retry_after_seconds` 并强制校验冷却时间。

多图片任务使用 `POST /ocr/tasks/multi-image`：

- multipart 字段名为重复的 `files`，上传顺序即页面顺序。
- 仅接收图片，不接收 PDF；单个 PDF 继续使用 `/ocr/tasks`。
- 所有图片共享任务上传大小限制，单次最多上传
  `MAX_MULTI_IMAGE_PAGES` 张。
- 文档透视矫正设置会逐页作用于该任务中的图片。

相关限制可在 `.env` 中调整：

```dotenv
MAX_UPLOAD_SIZE_MB=50
MAX_PDF_PAGES=50
MAX_MULTI_IMAGE_PAGES=50
PDF_RENDER_SCALE=2.0
PDF_MAX_RENDER_PIXELS=40000000
PDF_NATIVE_TEXT_MIN_CHARS=20
RETRY_COOLDOWN_SECONDS=60
```

完整设计与实施进度见
[docs/PDF_MULTIPAGE_IMPLEMENTATION_PLAN.md](docs/PDF_MULTIPAGE_IMPLEMENTATION_PLAN.md)。

## 部署方式

### 直接部署

1. 复制项目至部署路径

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *master 分支为项目中支持的 PaddleOCR 的最新版本, 如需安装特定版本, 请克隆对应版本号的分支.*

2. 安装 `uv`，并按 `pyproject.toml` 与 `uv.lock` 同步依赖

   ```shell
   uv sync --frozen
   ```

   > 依赖以 `pyproject.toml` 和 `uv.lock` 为准，不要再使用旧的
   > `requirements.txt`。

3. 复制环境变量模板：

   ```shell
   cp .env.example .env
   ```

   Windows + NVIDIA GPU 目标环境改为：

   ```dotenv
   OCR_MODEL_TIER=small
   OCR_DEVICE=gpu:0
   ```

4. 在联网准备机下载项目本地模型：

   ```shell
   uv run --frozen python -m scripts.prepare_models --with-doc-preprocessor
   ```

5. 运行 FastAPI

   ```shell
   uv run --frozen uvicorn main:app --host 0.0.0.0
   ```

PP-OCRv6_small、Windows RTX 2080、CUDA 版本选择以及后续切换
PP-OCRv6_medium 的完整说明见
[docs/PP_OCRV6_WINDOWS_GPU.md](docs/PP_OCRV6_WINDOWS_GPU.md)。

### Docker 部署

在 `Centos 7`, `Ubuntu 20.04`, `Ubuntu 22.04`, `Windows 10`, `Windows 11` 中测试成功, 需要先安装好 `Docker`.

> 当前 Dockerfile 已同步到 PP-OCRv6_small，镜像内默认使用 Linux CPU 版
> PaddlePaddle。Windows + RTX 2080 推荐使用上方直接部署方式；如果后续需要
> Docker GPU，需要单独切换 CUDA 基础镜像和 Linux GPU 版 PaddlePaddle 依赖。

1. 复制项目至部署路径

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *master 分支为项目中支持的 PaddleOCR 的最新版本, 如需安装特定版本, 请克隆对应版本号的分支.*

2. 制作 Docker 镜像

   ```shell
   cd PaddleOCRFastAPI
   # 使用宿主机网络 build；如果需要代理，可按实际情况增加 HTTP_PROXY / HTTPS_PROXY
   docker build -t paddleocrfastapi:latest --network host .
   ```

   > 构建镜像时会执行
   > `uv run --no-sync python -m scripts.prepare_models --with-doc-preprocessor`，
   > 将 PP-OCRv6 模型准备到 `/app/.paddlex/official_models`。如果 Docker
   > 构建环境不能联网，请先在联网机器准备好 `.paddlex/`，并确保它随 Docker
   > build context 一起复制进镜像。

3. 编辑 `docker-compose.yml`

    ```yaml
    services:

       PaddleOCR:
          build:
             context: .
             args:
                OCR_MODEL_TIER: small
          container_name: paddle_ocr_api # 自定义容器名
          image: paddleocrfastapi:latest # 自定义镜像名与标签
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
             - "8000:8000" # 自定义服务暴露端口, 8000 为 FastAPI 默认端口, 不做修改
          restart: unless-stopped
    ```

4. 生成 Docker 容器并运行

   ```shell
   docker compose up -d --build
   ```

5. Swagger 页面请访问 localhost:\<port\>/docs

如需在 Docker 中测试 PP-OCRv6_medium，请同时把 `docker-compose.yml` 中的
`build.args.OCR_MODEL_TIER` 和 `environment.OCR_MODEL_TIER` 改为 `medium`，
然后重新构建镜像。

### 内网离线部署建议

如果需要将当前环境迁移到内网电脑，可按下面方式准备：

1. 在一台可联网电脑上先把项目依赖与模型准备好。

2. 确认项目目录下至少包含以下内容：

   ```text
   .paddlex/official_models/
   ├─ PP-OCRv6_small_det/
   ├─ PP-OCRv6_small_rec/
   └─ PP-LCNet_x1_0_textline_ori/
   .env
   pyproject.toml
   uv.lock
   ```

3. `.paddlex` 被 Git 忽略，必须随项目压缩包显式带上。

4. 若目标机器完全离线，先在联网 Windows 准备机填充 uv 缓存或准备内网
   Python 软件源，再在目标机执行：

   ```shell
   uv sync --frozen --offline
   ```

## 语言支持

PP-OCRv6 使用统一模型支持中文、英文、日文及多种拉丁语系语言，不再使用旧的
`OCR_LANGUAGE` 配置切换模型。

## 文本分段与可选 AI 整理

OCR 完成后，服务会使用文字框坐标、行距、缩进、标题和列表等规则自动恢复
段落，并将规则结果随逐页任务保存。结果页默认展示规则分段，同时可以随时切换
到原始 OCR 行。

AI 整理为可选能力，只在用户点击“AI 整理当前页”后运行，并且仅处理当前页。
模型只决定原始行的分组，不允许改写、遗漏或重排行。整理结果、模型名称和完成
时间会保存到页面记录中。普通客户端每页只能整理一次；来自 `127.0.0.1`
或 `::1` 的调试请求可以重复整理并覆盖该页上一次 AI 结果。

AI 推理使用 `llama-cpp-python`，支持包含聊天模板的自定义 GGUF 指令模型：

1. 安装可选依赖：

   ```shell
   uv sync --extra ai
   ```

2. 手动创建模型目录并放入 GGUF 文件，例如：

   ```text
   local_models/
   └─ Qwen3-0.6B-Q4_K_M.gguf
   ```

3. 在 `.env` 中配置：

   ```dotenv
   AI_TEXT_MODEL_PATH=local_models/Qwen3-0.6B-Q4_K_M.gguf
   AI_TEXT_MODEL_NAME=Qwen3-0.6B-Q4_K_M
   AI_TEXT_MODEL_CONTEXT_SIZE=4096
   AI_TEXT_MODEL_THREADS=4
   AI_TEXT_MODEL_GPU_LAYERS=0
   ```

`AI_TEXT_MODEL_PATH` 未配置、文件不存在或没有安装可选依赖时，AI 按钮会显示
不可用，但不会影响 OCR 与坐标规则分段。相对路径以 FastAPI 启动目录为基准。
服务只维护一份常驻模型实例，AI 请求按顺序执行，避免并发重复加载模型。

Docker 部署时还需把 `docker-compose.yml` 中的
`build.args.INSTALL_AI` 改为 `true`，取消 AI 模型路径和
`local_models` 只读挂载的注释，然后重新构建镜像。默认镜像保持不安装 AI
依赖，因此不会增加只使用规则分段的部署体积。

## 运行截图
API 文档：`/docs`

![Swagger](https://raw.githubusercontent.com/cgcel/PaddleOCRFastAPI/dev/screenshots/Swagger.png)

## Todo

- [x] PP-OCRv6 small / medium 模型切换
- [x] Windows NVIDIA GPU 条件依赖
- [x] Image URL recognition

## License

**PaddleOCRFastAPI** is licensed under the MIT license. Refer to [LICENSE](https://github.com/cgcel/PaddleOCRFastAPI/blob/master/LICENSE) for more information.
