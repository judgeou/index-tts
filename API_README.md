# IndexTTS API 文档

本文档详细介绍了 IndexTTS API 服务器 (`api_server.py`) 提供的所有接口。

## 启动服务

通过项目根目录下的 `start_api.sh` 脚本启动 API 服务。您可以修改该脚本来指定端口号。

```bash
# 启动 API 服务 (默认端口 8000)
./start_api.sh

# 或者指定端口
PORT=8888 ./start_api.sh
```

服务启动后，您可以通过 `http://<服务器IP>:<端口号>` 访问 API。

## API 接口详解

---

### 1. 通用和状态接口

#### `GET /`

- **功能**: API 根路径，返回服务的基本信息。
- **请求**: `GET http://127.0.0.1:8000/`
- **响应**:
  ```json
  {
    "message": "IndexTTS API 服务运行中",
    "version": "1.0.0",
    "status": "running",
    "model_loaded": true
  }
  ```

#### `GET /health`

- **功能**: 健康检查接口，用于监控服务状态。
- **请求**: `GET http://127.0.0.1:8000/health`
- **响应**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "timestamp": 1678886400.0
  }
  ```

---

### 2. 核心语音合成接口

#### `POST /synthesize`

- **功能**: 全功能语音合成接口，一次性返回完整的 WAV 音频文件。支持所有推理参数。
- **请求**: `POST http://127.0.0.1:8000/synthesize`
- **请求体 (Body)**: `application/json`
  ```json
  {
    "text": "你好，欢迎使用 IndexTTS。",
    "reference_audio_index": 0,
    "use_fast_inference": false,
    "verbose": false,
    "max_text_tokens_per_sentence": 120,
    "do_sample": true,
    "top_p": 0.95,
    "top_k": 30,
    "temperature": 1.2,
    "length_penalty": 0.0,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 600,
    "sentences_bucket_max_size": 2
  }
  ```
- **参数说明**:
    - `text` (str, 必填): 要合成的文本。
    - `reference_audio_index` (int, 默认 0): 参考音频的序号。
    - `use_fast_inference` (bool, 默认 `false`): 是否使用快速推理模式。
    - `verbose` (bool, 默认 `false`): 是否在服务端打印详细日志。
    - `max_text_tokens_per_sentence` (int, 默认 120): 分句时每句话的最大 token 数量。
    - **生成参数**: `do_sample`, `top_p`, `top_k`, `temperature`, `length_penalty`, `num_beams`, `repetition_penalty`, `max_mel_tokens` 控制生成效果。
    - `sentences_bucket_max_size` (int, 默认 2): (仅快速推理) 分句分桶的最大容量。
- **响应**: `audio/wav` 格式的音频文件。

#### `POST /synthesize_simple`

- **功能**: 简化的语音合成接口，使用默认参数，方便快速调用。
- **请求**: `POST http://127.0.0.1:8000/synthesize_simple`
- **请求体 (Body)**: `application/json`
  ```json
  {
    "text": "这是一个简单的测试。",
    "reference_audio_index": 0
  }
  ```
- **响应**: `audio/wav` 格式的音频文件。

---

### 3. 流式语音合成接口

流式接口适合需要低延迟、实时返回音频流的场景。

#### `POST /synthesize_stream`

- **功能**: 流式语音合成，返回原始的 16-bit PCM 音频数据流。
- **请求**: `POST http://127.0.0.1:8000/synthesize_stream`
- **请求体 (Body)**: 与 `/synthesize` 类似，但不包含 `use_fast_inference` 和 `sentences_bucket_max_size`。
- **响应**: `application/octet-stream` 格式的原始 PCM 数据流。响应头中包含采样率等信息 (`X-Sample-Rate`, `X-Bit-Depth`, `X-Channels`)。

#### `POST /synthesize_stream_opus`

- **功能**: **(推荐)** 流式语音合成，返回 OGG 容器承载的 Opus 编码音频流。相比 PCM，流量消耗极小，非常适合网络传输。
- **注意**: 此接口内部有排队机制，服务器一次只处理一个请求，后续请求会排队等待。
- **请求**: `POST http://127.0.0.1:8000/synthesize_stream_opus`
- **请求体 (Body)**: 在 `/synthesize_stream` 的基础上增加了 Opus 编码参数。
  ```json
  {
    "text": "这是一个流式 Opus 合成测试。",
    "reference_audio_index": 0,
    "opus_bitrate": 32000,
    "opus_complexity": 10
    // ... 其他生成参数
  }
  ```
- **Opus 参数**:
    - `opus_bitrate` (int, 默认 32000): Opus 编码比特率 (范围: 8000-512000)。
    - `opus_complexity` (int, 默认 10): Opus 编码复杂度 (范围: 0-10)，越高计算量越大，质量越好。
- **响应**: `audio/ogg` 格式的 Opus 音频流。

#### `GET /opus_queue_status`

- **功能**: 获取 `/synthesize_stream_opus` 接口的排队状态。
- **请求**: `GET http://127.0.0.1:8000/opus_queue_status`
- **响应**:
  - 如果当前无任务:
    ```json
    {
      "is_processing": false,
      "queue_available": true,
      "timestamp": 1678886400.0
    }
    ```
  - 如果当前有任务在处理:
    ```json
    {
      "is_processing": true,
      "queue_available": false,
      "timestamp": 1678886400.0,
      "current_request": {
        "text_preview": "这是一个流式 Opus 合成测试...",
        "reference_audio_index": 0,
        "processing_duration_seconds": 5.23,
        "start_time": 1678886394.77
      }
    }
    ```

---

### 4. 参考音频管理

#### `GET /reference_audios`

- **功能**: 获取当前所有可用的参考音频列表。
- **请求**: `GET http://127.0.0.1:8000/reference_audios`
- **响应**:
  ```json
  {
    "reference_audios": [
      {
        "index": 0,
        "path": "/path/to/your/audios/sample1.wav",
        "exists": true,
        "filename": "sample1.wav"
      },
      {
        "index": 1,
        "path": "/path/to/your/audios/sample2.mp3",
        "exists": true,
        "filename": "sample2.mp3"
      }
    ],
    "total_count": 2
  }
  ```

#### `GET /download_reference_audio/{audio_index}`

- **功能**: 下载指定的参考音频文件。
- **请求**: `GET http://127.0.0.1:8000/download_reference_audio/0`
- **参数**:
    - `audio_index` (int, 路径参数): 要下载的音频序号。
- **响应**: 对应的音频文件。

---

### 5. 模型和测试接口

#### `GET /model_info`

- **功能**: 获取当前加载的 IndexTTS 模型信息。
- **请求**: `GET http://127.0.0.1:8000/model_info`
- **响应**:
  ```json
  {
    "model_loaded": true,
    "device": "cuda:0",
    "is_fp16": true,
    "use_cuda_kernel": false,
    "model_version": "1.5.0",
    "config_path": "checkpoints/config.yaml",
    "model_dir": "checkpoints"
  }
  ```

#### `GET /test`

- **功能**: 一个简单的测试接口，使用固定的文本 "欢迎使用TTS服务" 和默认参数进行快速推理，用于验证服务是否正常工作。
- **请求**: `GET http://127.0.0.1:8000/test`
- **响应**: `audio/wav` 格式的音频文件。

#### `GET /test_stream`

- **功能**: 一个用于测试服务器流式传输能力的接口。它会每秒发送一个数据块，持续10秒。
- **请求**: `GET http://127.0.0.1:8000/test_stream`
- **响应**: `text/plain` 格式的文本流。 