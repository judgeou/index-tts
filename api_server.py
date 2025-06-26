import os
import tempfile
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import struct

from indextts.infer import IndexTTS

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

# 预设的参考音频文件列表
REFERENCE_AUDIO_FILES = [
    "/mnt/f/project/fish-speech/source/boke-male.mp3",
    # 可以在这里添加更多参考音频文件
]


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
    """应用启动时初始化模型"""
    initialize_model()


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


if __name__ == "__main__":
    # 通过 start_api.sh 脚本启动，支持端口配置
    # 也可以直接运行: uvicorn api_server:app --host 0.0.0.0 --port 8000
    print("提示：建议使用 ./start_api.sh 启动服务，支持端口配置")
    print("或者直接使用: uvicorn api_server:app --host 0.0.0.0 --port 8000")
    
    # 默认启动
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议设为 False
        workers=1      # TTS 模型通常不支持多进程，使用单个 worker
    ) 