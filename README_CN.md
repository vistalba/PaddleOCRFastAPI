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

## 运行截图
API 文档：`/docs`

![Swagger](https://raw.githubusercontent.com/cgcel/PaddleOCRFastAPI/dev/screenshots/Swagger.png)

## Todo

- [x] PP-OCRv6 small / medium 模型切换
- [x] Windows NVIDIA GPU 条件依赖
- [x] Image URL recognition

## License

**PaddleOCRFastAPI** is licensed under the MIT license. Refer to [LICENSE](https://github.com/cgcel/PaddleOCRFastAPI/blob/master/LICENSE) for more information.
