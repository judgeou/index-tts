import os
import tempfile
import time
from typing import Optional
from pathlib import Path
import asyncio
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import struct

from indextts.infer import IndexTTS, InferenceCancelledError

# 初始化 FastAPI 应用
app = FastAPI(
    title="IndexTTS API",
    description="IndexTTS 语音合成 HTTP 接口",
    version="1.0.0"
)

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型实例
tts_model: Optional[IndexTTS] = None

# 用于流式 Opus 合成的排队锁，确保一次只处理一个请求
opus_synthesis_lock = asyncio.Lock()

# 用于跟踪当前正在处理的请求信息
current_opus_request = {
    "is_processing": False,
    "start_time": None,
    "text_preview": None,
    "reference_audio_index": None
}

# 参考音频配置
REFERENCE_AUDIO_DIRECTORY = Path("/mnt/f/project/fish-speech/source")
REFERENCE_AUDIO_FILES: list[str] = [] # 启动时从 REFERENCE_AUDIO_DIRECTORY 加载


def load_reference_audios():
    """从指定目录加载参考音频文件"""
    global REFERENCE_AUDIO_FILES
    if not REFERENCE_AUDIO_DIRECTORY.is_dir():
        print(f"⚠️ 参考音频目录不存在或不是一个目录: {REFERENCE_AUDIO_DIRECTORY}")
        REFERENCE_AUDIO_FILES = []
        return

    print(f"🎵 正在从 {REFERENCE_AUDIO_DIRECTORY} 加载参考音频...")
    allowed_extensions = ['.wav', '.mp3', '.opus']
    audio_files = []
    # 使用 sorted 确保文件顺序一致
    for p in sorted(REFERENCE_AUDIO_DIRECTORY.glob('*')):
        if p.is_file() and p.suffix.lower() in allowed_extensions:
            audio_files.append(str(p))
    
    REFERENCE_AUDIO_FILES = audio_files
    if REFERENCE_AUDIO_FILES:
        print(f"✅ 成功加载 {len(REFERENCE_AUDIO_FILES)} 个参考音频:")
        for i, f in enumerate(REFERENCE_AUDIO_FILES):
            print(f"  [{i}] {Path(f).name}")
    else:
        print("🟡 未找到任何参考音频文件。")


# Pydantic 模型定义
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_audio_index: int = Field(default=0, description="参考音频序号 (从0开始)")
    use_fast_inference: bool = Field(default=False, description="是否使用快速推理")
    verbose: bool = Field(default=False, description="是否输出详细日志")
    max_text_tokens_per_sentence: int = Field(default=120, description="每句话的最大token数")
    # 生成参数
    do_sample: bool = Field(default=True, description="是否使用采样")
    top_p: float = Field(default=0.95, description="top_p 采样参数")
    top_k: int = Field(default=30, description="top_k 采样参数")
    temperature: float = Field(default=1.2, description="温度参数")
    length_penalty: float = Field(default=0.0, description="长度惩罚")
    num_beams: int = Field(default=3, description="束搜索数量")
    repetition_penalty: float = Field(default=10.0, description="重复惩罚")
    max_mel_tokens: int = Field(default=600, description="最大mel token数量")
    # 快速推理专用参数
    sentences_bucket_max_size: int = Field(default=2, description="分句分桶的最大容量")


class SimpleSynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_audio_index: int = Field(default=0, description="参考音频序号 (从0开始)")


class StreamSynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_audio_index: int = Field(default=0, description="参考音频序号 (从0开始)")
    verbose: bool = Field(default=False, description="是否输出详细日志")
    max_text_tokens_per_sentence: int = Field(default=120, description="每句话的最大token数")
    # 生成参数
    do_sample: bool = Field(default=True, description="是否使用采样")
    top_p: float = Field(default=0.95, description="top_p 采样参数")
    top_k: int = Field(default=30, description="top_k 采样参数")
    temperature: float = Field(default=1.2, description="温度参数")
    length_penalty: float = Field(default=0.0, description="长度惩罚")
    num_beams: int = Field(default=3, description="束搜索数量")
    repetition_penalty: float = Field(default=10.0, description="重复惩罚")
    max_mel_tokens: int = Field(default=600, description="最大mel token数量")


class StreamOpusSynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_audio_index: int = Field(default=0, description="参考音频序号 (从0开始)")
    verbose: bool = Field(default=False, description="是否输出详细日志")
    max_text_tokens_per_sentence: int = Field(default=120, description="每句话的最大token数")
    # Opus 编码参数
    opus_bitrate: int = Field(default=32000, description="Opus编码比特率 (8000-512000)")
    opus_complexity: int = Field(default=10, description="Opus编码复杂度 (0-10)")
    # 生成参数
    do_sample: bool = Field(default=True, description="是否使用采样")
    top_p: float = Field(default=0.95, description="top_p 采样参数")
    top_k: int = Field(default=30, description="top_k 采样参数")
    temperature: float = Field(default=1.2, description="温度参数")
    length_penalty: float = Field(default=0.0, description="长度惩罚")
    num_beams: int = Field(default=3, description="束搜索数量")
    repetition_penalty: float = Field(default=10.0, description="重复惩罚")
    max_mel_tokens: int = Field(default=600, description="最大mel token数量")


def initialize_model():
    """初始化 IndexTTS 模型"""
    global tts_model
    try:
        print("正在初始化 IndexTTS 模型...")
        tts_model = IndexTTS(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            is_fp16=True,
            use_cuda_kernel=False
        )
        print("IndexTTS 模型初始化完成!")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise e


async def _synthesize_speech_core(request: SynthesizeRequest):
    """语音合成核心逻辑"""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    # 验证参考音频序号
    if request.reference_audio_index < 0 or request.reference_audio_index >= len(REFERENCE_AUDIO_FILES):
        raise HTTPException(
            status_code=400, 
            detail=f"参考音频序号无效: {request.reference_audio_index}. 有效范围: 0-{len(REFERENCE_AUDIO_FILES)-1}"
        )
    
    # 获取参考音频文件路径
    audio_prompt_path = REFERENCE_AUDIO_FILES[request.reference_audio_index]
    
    # 检查参考音频文件是否存在
    if not os.path.exists(audio_prompt_path):
        raise HTTPException(
            status_code=500, 
            detail=f"参考音频文件不存在: {audio_prompt_path}"
        )
    
    print(f"🎤 开始语音合成:")
    print(f"   文本: {request.text}")
    print(f"   参考音频: {audio_prompt_path}")
    print(f"   快速推理: {request.use_fast_inference}")
    
    try:
        # 创建临时文件（不自动删除）
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="tts_output_")
        os.close(temp_fd)  # 关闭文件描述符
        
        try:
            print(f"   临时输出路径: {temp_path}")
            
            # 准备生成参数
            generation_kwargs = {
                "do_sample": request.do_sample,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "temperature": request.temperature,
                "length_penalty": request.length_penalty,
                "num_beams": request.num_beams,
                "repetition_penalty": request.repetition_penalty,
                "max_mel_tokens": request.max_mel_tokens,
            }
            
            # 选择推理方法
            if request.use_fast_inference:
                print("🔄 使用快速推理模式...")
                result_path = tts_model.infer_fast(
                    audio_prompt=audio_prompt_path,
                    text=request.text,
                    output_path=temp_path,
                    verbose=request.verbose,
                    max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
                    sentences_bucket_max_size=request.sentences_bucket_max_size,
                    **generation_kwargs
                )
            else:
                print("🔄 使用标准推理模式...")
                result_path = tts_model.infer(
                    audio_prompt=audio_prompt_path,
                    text=request.text,
                    output_path=temp_path,
                    verbose=request.verbose,
                    max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
                    **generation_kwargs
                )
            
            print(f"   推理返回路径: {result_path}")
            # 检查生成是否成功
            if not result_path or not isinstance(result_path, str) or not os.path.exists(result_path):
                print(f"❌ 音频文件生成失败:")
                print(f"   返回路径: {result_path}")
                print(f"   输出路径: {temp_path}")
                print(f"   输出路径存在: {os.path.exists(str(temp_path)) if temp_path else False}")
                if result_path:
                    print(f"   返回路径存在: {os.path.exists(str(result_path))}")
                raise HTTPException(status_code=500, detail="音频生成失败：生成的音频文件不存在")
            
            print(f"✅ 音频生成成功: {result_path}")
            
            # 创建自定义 FileResponse 类，在发送后自动删除临时文件
            class TempFileResponse(FileResponse):
                def __init__(self, *args, **kwargs):
                    self.temp_file_path = kwargs.pop('temp_file_path', None)
                    super().__init__(*args, **kwargs)
                
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # 在响应发送完成后删除临时文件
                        if self.temp_file_path and os.path.exists(self.temp_file_path):
                            try:
                                os.unlink(self.temp_file_path)
                                print(f"🗑️ 临时文件已删除: {self.temp_file_path}")
                            except Exception as e:
                                print(f"⚠️ 删除临时文件失败: {e}")
            
            # 返回生成的音频文件
            return TempFileResponse(
                result_path,
                media_type="audio/wav",
                filename=f"synthesized_{int(time.time())}.wav",
                headers={
                    "Content-Disposition": "attachment; filename=synthesized_audio.wav"
                },
                temp_file_path=result_path
            )
            
        except Exception as e:
            # 如果发生错误，确保清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise e
            
    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except Exception as e:
        print(f"❌ 语音合成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型并加载参考音频"""
    initialize_model()
    load_reference_audios()


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "message": "IndexTTS API 服务运行中",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": tts_model is not None
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "timestamp": time.time()
    }


@app.post("/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    """
    语音合成接口
    
    Args:
        request: 包含所有合成参数的请求体
    
    Returns:
        合成的音频文件
    """
    return await _synthesize_speech_core(request)


@app.post("/synthesize_simple")
async def synthesize_speech_simple(request: SimpleSynthesizeRequest):
    """
    简化的语音合成接口，使用默认参数
    适合简单的TTS调用
    """
    # 转换为完整的请求对象
    full_request = SynthesizeRequest(
        text=request.text,
        reference_audio_index=request.reference_audio_index,
        use_fast_inference=False,
        verbose=False
    )
    return await _synthesize_speech_core(full_request)


@app.post("/synthesize_stream")
async def synthesize_stream(request: StreamSynthesizeRequest):
    """
    流式语音合成接口
    
    返回16bit PCM数据流，不包含WAV头
    适合实时音频处理和低延迟应用
    
    Args:
        request: 包含合成参数的请求体
    
    Returns:
        StreamingResponse: 16bit PCM音频数据流
    """
    if tts_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    # 验证参考音频序号
    if request.reference_audio_index < 0 or request.reference_audio_index >= len(REFERENCE_AUDIO_FILES):
        raise HTTPException(
            status_code=400, 
            detail=f"参考音频序号无效: {request.reference_audio_index}. 有效范围: 0-{len(REFERENCE_AUDIO_FILES)-1}"
        )
    
    # 获取参考音频文件路径
    audio_prompt_path = REFERENCE_AUDIO_FILES[request.reference_audio_index]
    
    # 检查参考音频文件是否存在
    if not os.path.exists(audio_prompt_path):
        raise HTTPException(
            status_code=500, 
            detail=f"参考音频文件不存在: {audio_prompt_path}"
        )
    
    print(f"🎤 开始流式语音合成:")
    print(f"   文本: {request.text}")
    print(f"   参考音频: {audio_prompt_path}")
    
    # 准备生成参数
    generation_kwargs = {
        "do_sample": request.do_sample,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "temperature": request.temperature,
        "length_penalty": request.length_penalty,
        "num_beams": request.num_beams,
        "repetition_penalty": request.repetition_penalty,
        "max_mel_tokens": request.max_mel_tokens,
    }
    
    def generate_pcm_stream():
        """生成16bit PCM数据流的生成器函数"""
        try:
            # 调用 infer_stream_pcm 获取16bit PCM音频片段
            assert tts_model is not None, "模型实例不应为None"
            for chunk_info in tts_model.infer_stream_pcm(
                audio_prompt=audio_prompt_path,
                text=request.text,
                verbose=request.verbose,
                max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
                **generation_kwargs
            ):
                # 获取PCM字节数据
                pcm_bytes = chunk_info['audio_pcm_bytes']
                
                print(f"📦 生成音频片段: {len(pcm_bytes)} 字节, 句子 {chunk_info['sentence_index'] + 1}/{chunk_info['total_sentences']}")
                
                yield pcm_bytes
                
        except HTTPException:
            # 重新抛出 HTTP 异常
            raise
        except Exception as e:
            print(f"❌ 流式语音合成过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"流式语音合成失败: {str(e)}")
    
    # 返回流式响应
    return StreamingResponse(
        generate_pcm_stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=synthesized_audio.pcm",
            "X-Sample-Rate": "24000",  # 采样率信息放在头部
            "X-Bit-Depth": "16",      # 位深度信息
            "X-Channels": "1",        # 声道数信息
        }
    )


@app.post("/synthesize_stream_opus")
async def synthesize_stream_opus(request: StreamOpusSynthesizeRequest, fastapi_request: Request):
    """
    流式 Opus 语音合成接口（带排队功能）
    
    返回 Opus 编码的音频数据流，相比 PCM 大幅减少流量消耗
    适合网络传输和实时音频应用
    
    注意：该接口使用排队机制，一次只处理一个请求，确保模型资源不被争用
    
    Args:
        request: 包含合成参数和 Opus 编码参数的请求体
    
    Returns:
        StreamingResponse: Opus 编码的音频数据流
    """
    if tts_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    # 验证参考音频序号
    if request.reference_audio_index < 0 or request.reference_audio_index >= len(REFERENCE_AUDIO_FILES):
        raise HTTPException(
            status_code=400, 
            detail=f"参考音频序号无效: {request.reference_audio_index}. 有效范围: 0-{len(REFERENCE_AUDIO_FILES)-1}"
        )
    
    # 获取参考音频文件路径
    audio_prompt_path = REFERENCE_AUDIO_FILES[request.reference_audio_index]
    
    # 检查参考音频文件是否存在
    if not os.path.exists(audio_prompt_path):
        raise HTTPException(
            status_code=500, 
            detail=f"参考音频文件不存在: {audio_prompt_path}"
        )
    
    async def generate_opus_stream():
        """
        生成 Opus 数据流的异步生成器函数。
        该函数包含完整的排队锁、状态管理和取消逻辑。
        """
        # 检查是否有其他请求正在处理
        if opus_synthesis_lock.locked():
            print(f"⏳ 有其他请求正在处理中，当前请求正在排队等待...")
            # 等待锁被释放
            await opus_synthesis_lock.acquire()
            opus_synthesis_lock.release()
        
        # 获取排队锁，确保一次只处理一个请求
        await opus_synthesis_lock.acquire()

        # 创建一个线程事件用于通知推理核心中断
        cancellation_event = threading.Event()

        async def check_disconnect():
            """在后台运行，定期检查客户端是否已断开连接"""
            while not cancellation_event.is_set():
                if await fastapi_request.is_disconnected():
                    print("🛑 客户端已断开连接，设置取消信号...")
                    cancellation_event.set()
                    break
                await asyncio.sleep(0.1)

        # 启动后台断开连接检查任务
        disconnect_task = asyncio.create_task(check_disconnect())

        try:
            # 更新当前处理状态
            current_opus_request["is_processing"] = True
            current_opus_request["start_time"] = time.time()
            current_opus_request["text_preview"] = request.text[:50] + ("..." if len(request.text) > 50 else "")
            current_opus_request["reference_audio_index"] = request.reference_audio_index
            
            print(f"🎤 开始流式 Opus 语音合成 (已获取处理锁):")
            print(f"   文本: {request.text}")
            print(f"   参考音频: {audio_prompt_path}")
            print(f"   Opus 比特率: {request.opus_bitrate}bps")
            print(f"   Opus 复杂度: {request.opus_complexity}")
            
            # 准备生成参数
            generation_kwargs = {
                "do_sample": request.do_sample,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "temperature": request.temperature,
                "length_penalty": request.length_penalty,
                "num_beams": request.num_beams,
                "repetition_penalty": request.repetition_penalty,
                "max_mel_tokens": request.max_mel_tokens,
            }

            # 直接调用 infer_opus，它现在返回 bytes 流
            assert tts_model is not None, "模型实例不应为None"
            chunk_count = 0
            for opus_bytes in tts_model.infer_opus(
                audio_prompt=audio_prompt_path,
                text=request.text,
                verbose=request.verbose,
                max_text_tokens_per_sentence=request.max_text_tokens_per_sentence,
                opus_bitrate=request.opus_bitrate,
                opus_complexity=request.opus_complexity,
                cancellation_event=cancellation_event,
                **generation_kwargs
            ):
                chunk_count += 1
                
                # 直接 yield Opus 字节数据
                yield opus_bytes
                
                # 强制刷新：让协程让出控制权，确保数据被发送
                await asyncio.sleep(0)
            
            print(f"Opus streaming completed: {chunk_count} chunks sent")
                
        except HTTPException:
            # 重新抛出 HTTP 异常
            raise
        except InferenceCancelledError:
            # 这是预期的取消操作，仅记录日志
            print("✅ 流式任务已被客户端成功取消。")
        except Exception as e:
            # 捕获任何其他异常
            if "Broken" in str(e) or "closed" in str(e):
                print(f"客户端可能已断开连接: {e}")
            else:
                print(f"❌ 流式 Opus 语音合成过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"流式 Opus 语音合成失败: {str(e)}")
        finally:
            # 确保后台任务被清理
            if not cancellation_event.is_set():
                cancellation_event.set() # 确保后台任务可以退出
            disconnect_task.cancel()
            
            # 清理处理状态
            current_opus_request["is_processing"] = False
            current_opus_request["start_time"] = None
            current_opus_request["text_preview"] = None
            current_opus_request["reference_audio_index"] = None

            # 释放锁
            opus_synthesis_lock.release()
            print("🔓 流式 Opus 语音合成完成或被取消，释放处理锁")

    # 返回流式 OGG 响应，添加实时传输头
    return StreamingResponse(
        generate_opus_stream(),
        media_type="audio/ogg",  # 修改为正确的OGG MIME类型
        headers={
            "Content-Disposition": "attachment; filename=synthesized_audio.ogg",
            "X-Opus-Sample-Rate": "48000",           # Opus 标准采样率
            "X-Original-Sample-Rate": "24000",       # 原始采样率
            "X-Opus-Bitrate": str(request.opus_bitrate),
            "X-Opus-Complexity": str(request.opus_complexity),
            "X-Channels": "1",                       # 单声道
            "Transfer-Encoding": "chunked",          # 分块传输编码
            "Cache-Control": "no-cache, no-store, must-revalidate",  # 禁用缓存
            "Pragma": "no-cache",                   # HTTP/1.0缓存控制
            "Expires": "0",                         # 立即过期
            "Connection": "keep-alive",             # 保持连接
            "X-Accel-Buffering": "no",             # 禁用nginx缓冲(如果有)
            "X-Queue-Info": "sequential-processing", # 标识使用排队处理
        }
    )


@app.get("/test")
async def test_speech():
    """
    测试语音合成接口
    使用固定文本"欢迎使用TTS服务"进行语音合成测试
    """
    test_text = "欢迎使用TTS服务"
    
    # 创建测试请求
    test_request = SynthesizeRequest(
        text=test_text,
        reference_audio_index=0,
        use_fast_inference=True,
        verbose=False
    )
    
    return await _synthesize_speech_core(test_request)


@app.get("/reference_audios")
async def get_reference_audios():
    """获取可用的参考音频列表"""
    audio_list = []
    for i, audio_path in enumerate(REFERENCE_AUDIO_FILES):
        audio_info = {
            "index": i,
            "path": audio_path,
            "exists": os.path.exists(audio_path),
            "filename": os.path.basename(audio_path)
        }
        audio_list.append(audio_info)
    
    return {
        "reference_audios": audio_list,
        "total_count": len(REFERENCE_AUDIO_FILES)
    }


@app.get("/download_reference_audio/{audio_index}")
async def download_reference_audio(audio_index: int):
    """
    下载指定的参考音频文件。

    Args:
        audio_index: 参考音频的序号 (从 /reference_audios 接口获取)。

    Returns:
        对应的参考音频文件。
    """
    if not (0 <= audio_index < len(REFERENCE_AUDIO_FILES)):
        raise HTTPException(
            status_code=404,
            detail=f"参考音频序号无效: {audio_index}。有效范围: 0-{len(REFERENCE_AUDIO_FILES) - 1}"
        )

    audio_path = REFERENCE_AUDIO_FILES[audio_index]

    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"参考音频文件不存在或已被删除: {audio_path}"
        )
    
    filename = os.path.basename(audio_path)

    return FileResponse(
        path=audio_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/model_info")
async def get_model_info():
    """获取模型信息"""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    return {
        "model_loaded": True,
        "device": str(tts_model.device),
        "is_fp16": tts_model.is_fp16,
        "use_cuda_kernel": tts_model.use_cuda_kernel,
        "model_version": getattr(tts_model, 'model_version', None),
        "config_path": "checkpoints/config.yaml",
        "model_dir": tts_model.model_dir
    }


@app.get("/opus_queue_status")
async def get_opus_queue_status():
    """获取流式 Opus 合成的排队状态"""
    status_info = {
        "is_processing": current_opus_request["is_processing"],
        "queue_available": not opus_synthesis_lock.locked(),
        "timestamp": time.time()
    }
    
    if current_opus_request["is_processing"]:
        # 计算处理时长
        processing_duration = time.time() - (current_opus_request["start_time"] or time.time())
        status_info.update({
            "current_request": {
                "text_preview": current_opus_request["text_preview"],
                "reference_audio_index": current_opus_request["reference_audio_index"],
                "processing_duration_seconds": round(processing_duration, 2),
                "start_time": current_opus_request["start_time"]
            }
        })
    
    return status_info


@app.get("/test_stream")
async def test_stream():
    """测试流式传输的端点，每秒发送一个数据块"""
    async def generate_test_stream():
        for i in range(10):
            test_data = f"数据块 {i+1}/10 - 时间戳: {time.time()}\n".encode()
            print(f"📤 [TEST] 发送测试数据块 {i+1}: {len(test_data)} bytes")
            yield test_data
            await asyncio.sleep(1)  # 等待1秒
        print("✅ [TEST] 测试流式传输完成")
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    # 通过 start_api.sh 脚本启动，支持端口配置
    # 也可以直接运行: uvicorn api_server:app --host 0.0.0.0 --port 8000
    print("提示：建议使用 ./start_api.sh 启动服务，支持端口配置")
    print("或者直接使用: uvicorn api_server:app --host 0.0.0.0 --port 8000")
    
    # 默认启动，配置实时传输
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议设为 False
        workers=1,     # TTS 模型通常不支持多进程，使用单个 worker
        # 实时传输优化配置
        http="httptools",           # 使用更快的HTTP解析器
        loop="uvloop",             # 使用更快的事件循环(Linux)
        timeout_keep_alive=65,     # 保持连接时间
        limit_concurrency=10,      # 限制并发连接数
        backlog=128               # 连接队列大小
    ) 