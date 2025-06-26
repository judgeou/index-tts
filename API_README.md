# IndexTTS HTTP API 使用说明

这是 IndexTTS 的 HTTP API 接口，为 Android TextToSpeechService 等应用提供语音合成服务。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 FastAPI 和 uvicorn
pip install fastapi uvicorn

# 或者如果已有 requirements.txt，直接安装
pip install -r requirements.txt
```

### 2. 启动服务

#### 方法一：使用启动脚本（推荐）
```bash
# 默认端口 8000
./start_api.sh

# 指定端口
./start_api.sh 8080

# 使用环境变量
API_PORT=9000 ./start_api.sh

# 查看帮助
./start_api.sh --help
```

#### 方法二：直接运行
```bash
python api_server.py
```

#### 方法三：使用 uvicorn 命令
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

### 3. 访问 API

- **服务地址**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs (Swagger UI)
- **交互式文档**: http://localhost:8000/redoc

## 📚 API 接口说明

### 1. 健康检查
```http
GET /health
```

返回服务状态和模型加载情况。

### 2. 模型信息
```http
GET /model_info
```

获取当前加载的模型详细信息。

### 3. 测试语音合成
```http
GET /test
```

快速测试语音合成服务，使用固定文本"欢迎使用TTS服务"进行测试。不需要任何参数，返回合成的音频文件。

### 4. 参考音频列表
```http
GET /reference_audios
```

获取可用的参考音频列表，包含序号、文件路径和是否存在等信息。

### 5. 语音合成（完整接口）
```http
POST /synthesize
```

**参数说明**:
- `text` (必需): 要合成的文本
- `reference_audio_index` (可选): 参考音频序号，默认 `0`
- `use_fast_inference` (可选): 是否使用快速推理，默认 `true`
- `verbose` (可选): 是否输出详细日志，默认 `false`
- `max_text_tokens_per_sentence` (可选): 每句话的最大token数，默认 `120`

**生成参数**:
- `do_sample`: 是否使用采样，默认 `true`
- `top_p`: top_p 采样参数，默认 `0.8`
- `top_k`: top_k 采样参数，默认 `30`
- `temperature`: 温度参数，默认 `1.0`
- `length_penalty`: 长度惩罚，默认 `0.0`
- `num_beams`: 束搜索数量，默认 `3`
- `repetition_penalty`: 重复惩罚，默认 `10.0`
- `max_mel_tokens`: 最大mel token数量，默认 `600`

**快速推理专用参数**:
- `sentences_bucket_max_size`: 分句分桶的最大容量，默认 `4`

### 6. 语音合成（简化接口）
```http
POST /synthesize_simple
```

**参数说明**:
- `text` (必需): 要合成的文本
- `reference_audio_index` (可选): 参考音频序号，默认 `0`

使用默认参数进行语音合成，适合简单的 TTS 调用。

## 🔧 客户端使用示例

### Python 客户端

我们提供了完整的 Python 客户端示例 `client_example.py`：

```bash
# 快速测试服务
python client_example.py --test

# 查看可用的参考音频
python client_example.py --list-audio

# 基本使用（使用默认参考音频序号0）
python client_example.py -t "你好，这是一个测试"

# 指定参考音频序号
python client_example.py -t "你好，这是一个测试" -r 0

# 使用简化接口
python client_example.py -t "你好，这是一个测试" --simple

# 详细输出
python client_example.py -t "你好，这是一个测试" -v

# 指定输出文件
python client_example.py -t "你好，这是一个测试" -o my_output.wav

# 连接到远程服务器
python client_example.py -t "你好，这是一个测试" --url http://192.168.1.100:8000
```

### cURL 示例

```bash
# 健康检查
curl -X GET "http://localhost:8000/health"

# 快速测试语音合成
curl -X GET "http://localhost:8000/test" --output test_output.wav

# 获取参考音频列表
curl -X GET "http://localhost:8000/reference_audios"

# 语音合成（简化接口）
curl -X POST "http://localhost:8000/synthesize_simple" \
     -F "text=你好，这是一个测试" \
     -F "reference_audio_index=0" \
     --output output.wav

# 语音合成（完整接口）
curl -X POST "http://localhost:8000/synthesize" \
     -F "text=你好，这是一个测试" \
     -F "reference_audio_index=0" \
     -F "use_fast_inference=true" \
     -F "temperature=1.0" \
     -F "top_p=0.8" \
     --output output.wav
```

### JavaScript/TypeScript 示例

```javascript
// 测试语音合成服务
async function testSpeech() {
    try {
        const response = await fetch('http://localhost:8000/test');
        if (response.ok) {
            const audioBlob = await response.blob();
            // 处理音频 blob
            return audioBlob;
        } else {
            console.error('测试失败:', response.statusText);
        }
    } catch (error) {
        console.error('测试请求失败:', error);
    }
}

async function synthesizeSpeech(text, referenceAudioIndex = 0) {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('reference_audio_index', referenceAudioIndex);
    formData.append('use_fast_inference', 'true');
    
    try {
        const response = await fetch('http://localhost:8000/synthesize_simple', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const audioBlob = await response.blob();
            // 处理音频 blob
            return audioBlob;
        } else {
            console.error('语音合成失败:', response.statusText);
        }
    } catch (error) {
        console.error('请求失败:', error);
    }
}

// 获取参考音频列表
async function getReferenceAudios() {
    try {
        const response = await fetch('http://localhost:8000/reference_audios');
        if (response.ok) {
            const data = await response.json();
            return data.reference_audios;
        }
    } catch (error) {
        console.error('获取参考音频列表失败:', error);
    }
    return [];
}
```

## 🤖 Android 集成示例

对于 Android TextToSpeechService，可以参考以下集成方式：

```java
// 使用 OkHttp 进行网络请求
public class IndexTTSService extends TextToSpeechService {
    private static final String API_URL = "http://your-server:8000/synthesize_simple";
    private static final String TEST_URL = "http://your-server:8000/test";
    
    // 测试语音合成服务
    private void testSpeech(Callback callback) {
        OkHttpClient client = new OkHttpClient();
        
        Request request = new Request.Builder()
            .url(TEST_URL)
            .get()
            .build();
            
        client.newCall(request).enqueue(callback);
    }
    
    private void synthesizeText(String text, int referenceAudioIndex, Callback callback) {
        OkHttpClient client = new OkHttpClient();
        
        RequestBody requestBody = new MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("text", text)
            .addFormDataPart("reference_audio_index", String.valueOf(referenceAudioIndex))
            .build();
            
        Request request = new Request.Builder()
            .url(API_URL)
            .post(requestBody)
            .build();
            
        client.newCall(request).enqueue(callback);
    }
    
    // 获取参考音频列表
    private void getReferenceAudios(Callback callback) {
        OkHttpClient client = new OkHttpClient();
        
        Request request = new Request.Builder()
            .url("http://your-server:8000/reference_audios")
            .get()
            .build();
            
        client.newCall(request).enqueue(callback);
    }
}
```

## ⚙️ 配置选项

### 参考音频配置

参考音频文件通过 `api_server.py` 中的 `REFERENCE_AUDIO_FILES` 数组配置：

```python
# 预设的参考音频文件列表
REFERENCE_AUDIO_FILES = [
    "/mnt/f/project/fish-speech/source/boke-male.mp3",
    # 可以在这里添加更多参考音频文件
    # "/path/to/your/audio1.wav",
    # "/path/to/your/audio2.mp3",
]
```

**添加新的参考音频**：
1. 将音频文件放置到服务器可访问的路径
2. 在 `REFERENCE_AUDIO_FILES` 数组中添加文件路径
3. 重启 API 服务
4. 使用 `GET /reference_audios` 确认新音频已添加

**音频格式要求**：
- 支持常见音频格式：wav, mp3, flac, m4a, ogg
- 建议使用清晰、无噪音的语音样本
- 推荐时长：3-10秒

### 服务器配置

可以通过修改 `api_server.py` 中的参数来调整服务器配置：

```python
# 修改监听地址和端口
uvicorn.run(
    "api_server:app",
    host="0.0.0.0",      # 监听所有接口
    port=8000,           # 端口号
    reload=False,        # 生产环境建议关闭
    workers=1            # worker 数量（TTS 模型建议为 1）
)
```

### 模型配置

可以通过修改模型初始化参数来调整性能：

```python
tts_model = IndexTTS(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=True,        # 是否使用半精度，可提升速度
    device="cuda:0",     # 指定设备
    use_cuda_kernel=True # 是否使用 CUDA 内核加速
)
```

## 🧪 快速测试

### 服务测试

使用内置的测试接口可以快速验证服务是否正常工作：

**API 调用**：
```bash
curl -X GET "http://localhost:8000/test" --output test.wav
```

**Python 客户端**：
```bash
python client_example.py --test
```

**JavaScript**：
```javascript
const audioBlob = await testSpeech();
```

**测试特点**：
- ✅ 使用固定文本："欢迎使用TTS服务"
- ✅ 使用默认参考音频（序号0）
- ✅ 使用快速推理模式
- ✅ 无需额外参数
- ✅ 直接返回音频文件

**测试用途**：
- 🔍 验证服务状态
- 🎯 检查音频质量
- ⚡ 测试推理速度
- 📱 Android 开发调试
- 🚀 部署后验证

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 `checkpoints/config.yaml` 是否存在
   - 确保模型文件完整下载
   - 查看详细错误信息

2. **CUDA 相关错误**
   - 设置 `use_cuda_kernel=False`
   - 检查 CUDA 环境是否正确安装

3. **内存不足**
   - 减少 `sentences_bucket_max_size`
   - 减少 `max_text_tokens_per_sentence`
   - 使用 CPU 推理：`device="cpu"`

4. **推理速度慢**
   - 使用 `use_fast_inference=True`
   - 启用 FP16：`is_fp16=True`
   - 使用 GPU 加速

5. **测试接口失败**
   - 首先运行 `GET /health` 检查服务状态
   - 检查参考音频文件是否存在
   - 查看服务器日志获取详细错误信息

### 性能优化

1. **快速推理模式**：`use_fast_inference=True`
2. **调整分句参数**：减少 `max_text_tokens_per_sentence`
3. **增加批处理大小**：增加 `sentences_bucket_max_size`
4. **使用 GPU**：确保 CUDA 环境正确
5. **启用 FP16**：在支持的硬件上使用半精度

## 📝 注意事项

1. **线程安全**：当前实现使用单个模型实例，建议使用单个 worker
2. **内存管理**：模型会自动清理 GPU 缓存，但长时间运行建议定期重启
3. **文件清理**：临时文件会自动清理，无需手动处理
4. **并发限制**：由于模型特性，建议限制并发请求数量

## 🔗 相关链接

- [IndexTTS 项目主页](https://github.com/index-tts/index-tts)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Android TextToSpeechService 文档](https://developer.android.com/reference/android/speech/tts/TextToSpeechService) 