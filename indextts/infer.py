import os
import sys
import time
import io
import subprocess
from subprocess import CalledProcessError
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures

from indextts.utils.front import TextNormalizer, TextTokenizer


class IndexTTS:
    def __init__(
        self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens_list: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            squeezed_tokens = [t.squeeze(0) for t in tokens_list]
            return pad_sequence(squeezed_tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token, padding_side="right")
        max_len = max(t.size(1) for t in tokens_list)
        outputs = []
        for tensor in tokens_list:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        concatenated_tokens = torch.cat(outputs, dim=0)
        return concatenated_tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 快速推理：对于"多句长文本"，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token数，默认``100``，可以根据GPU硬件情况调整
                - 越小，batch 越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越大，batch 越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，默认``4``，可以根据GPU内存调整
                - 越大，bucket数量越少，batch越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越小，bucket数量越多，batch越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
        """
        print(">> start fast inference...")
        
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count:", len(text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_max_size: {bucket_max_size}", f"bucket_count: {bucket_count}" if bucket_max_size > 1 else "")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)

    # 流式推理模式 - 逐句生成音频片段
    def infer_stream(self, audio_prompt, text, verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        """
        流式推理函数，逐句生成音频片段
        
        Args:
            audio_prompt: 参考音频路径
            text: 要合成的文本
            verbose: 是否输出详细信息
            max_text_tokens_per_sentence: 每句最大token数
            **generation_kwargs: 生成参数
            
        Yields:
            dict: 包含音频片段信息的字典
                - audio_chunk: torch.Tensor, 音频数据 (float32, 确保兼容性)
                - sample_rate: int, 采样率
                - sentence_index: int, 当前句子索引 (从0开始)
                - total_sentences: int, 总句子数
                - sentence_text: str, 当前句子的token文本
                - timing_info: dict, 时间统计信息
        """
        print(">> start streaming inference...")
        print(f"🚀 [DEBUG] infer_stream 开始处理，设备: {self.device}")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        
        # 检查GPU内存使用情况
        if "cuda" in str(self.device):
            try:
                import torch
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                print(f"🔍 [DEBUG] GPU内存使用 (开始): {gpu_memory_before:.2f} GB")
            except:
                pass

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        cond_mel_start = time.perf_counter()
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            print(f"🎵 [DEBUG] 开始处理参考音频: {audio_prompt}")
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
            print(f"✅ [DEBUG] 参考音频处理完成，耗时: {time.perf_counter() - cond_mel_start:.3f}s")
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            print(f"🔄 [DEBUG] 使用缓存的参考音频，耗时: {time.perf_counter() - cond_mel_start:.3f}s")

        # 文本处理
        text_process_start = time.perf_counter()
        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        text_process_time = time.perf_counter() - text_process_start
        
        print(f"📝 [DEBUG] 文本处理完成，耗时: {text_process_time:.3f}s")
        print(f"📊 [DEBUG] 文本统计: 总token={len(text_tokens_list)}, 句子数={len(sentences)}, 每句最大token={max_text_tokens_per_sentence}")
        
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        
        # 提取生成参数
        params_start = time.perf_counter()
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        params_time = time.perf_counter() - params_start
        print(f"⚙️ [DEBUG] 参数处理完成，耗时: {params_time:.3f}s")
        
        total_sentences = len(sentences)
        has_warned = False
        cumulative_gpt_gen_time = 0
        cumulative_gpt_forward_time = 0
        cumulative_bigvgan_time = 0
        
        print(f"🔄 [DEBUG] 开始逐句处理 {total_sentences} 个句子")
        
        for sentence_idx, sent in enumerate(sentences):
            sentence_start_time = time.perf_counter()
            print(f"\n🔄 [DEBUG] === 处理第 {sentence_idx + 1}/{total_sentences} 句 ===")
            
            # 检查GPU内存
            if "cuda" in str(self.device):
                try:
                    gpu_memory_current = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"🔍 [DEBUG] 当前GPU内存: {gpu_memory_current:.2f} GB")
                except:
                    pass
            
            # 🔥 关键修复：清理GPT模型的KV缓存，防止跨句子累积
            if hasattr(self.gpt, 'inference_model') and hasattr(self.gpt.inference_model, 'cached_mel_emb'):
                # 重新设置mel embedding缓存，清理之前的状态
                self.gpt.inference_model.store_mel_emb(auto_conditioning)
                print(f"🧠 [DEBUG] 清理GPT KV缓存并重设mel embedding")
            
            # 文本token处理
            token_start = time.perf_counter()
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            token_time = time.perf_counter() - token_start
            print(f"📝 [DEBUG] Token处理耗时: {token_time:.3f}s, shape: {text_tokens.shape}")
            
            if verbose:
                print(f"\n--- Processing sentence {sentence_idx + 1}/{total_sentences} ---")
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")

            # GPT inference_speech 生成codes
            print(f"🧠 [DEBUG] 开始GPT inference_speech...")
            gpt_gen_start = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens,
                                                        **generation_kwargs)
            gpt_gen_time = time.perf_counter() - gpt_gen_start
            cumulative_gpt_gen_time += gpt_gen_time
            print(f"✅ [DEBUG] GPT inference_speech完成，耗时: {gpt_gen_time:.3f}s")
            
            # 🔍 检查GPT生成后的内存
            if "cuda" in str(self.device):
                try:
                    gpu_memory_after_gpt_gen = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"🔍 [DEBUG] GPT生成后GPU内存: {gpu_memory_after_gpt_gen:.2f} GB")
                except:
                    pass
            
            if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                warnings.warn(
                    f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                    f"Input text tokens: {text_tokens.shape[1]}. "
                    f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                    category=RuntimeWarning
                )
                has_warned = True

            # 🎯 修复codes处理逻辑 - 更安全的类型检查和处理
            codes_process_start = time.perf_counter()
            
            # 统一处理codes - 确保始终是Tensor格式
            # 使用更安全的类型检查方式，避免linter错误
            if hasattr(codes, 'sequences') and not isinstance(codes, torch.Tensor):
                # GenerateOutput对象，提取sequences
                codes = getattr(codes, 'sequences')
                print(f"🔧 [DEBUG] 从GenerateOutput提取sequences")
            elif isinstance(codes, torch.Tensor):
                # 已经是Tensor
                print(f"🔧 [DEBUG] codes已经是Tensor格式")
            else:
                # 其他情况，尝试转换
                codes = torch.tensor(codes, device=self.device)
                print(f"🔧 [DEBUG] 转换codes为Tensor")
            
            # 确保codes是Tensor并且维度正确
            if isinstance(codes, torch.Tensor) and codes.dim() == 1:
                codes = codes.unsqueeze(0)
            
            code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=torch.long)
            codes_process_time = time.perf_counter() - codes_process_start
            print(f"🔧 [DEBUG] Codes处理耗时: {codes_process_time:.3f}s, shape: {codes.shape}")
            
            if verbose:
                print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

            # 移除超长静音
            silence_start = time.perf_counter()
            codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
            silence_time = time.perf_counter() - silence_start
            print(f"🔇 [DEBUG] 静音移除耗时: {silence_time:.3f}s")
            if verbose:
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")
                
            # GPT forward 生成latent
            print(f"🧠 [DEBUG] 开始GPT forward...")
            gpt_forward_start = time.perf_counter()
            with torch.no_grad():  # 确保没有梯度计算
                with torch.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
            gpt_forward_time = time.perf_counter() - gpt_forward_start
            cumulative_gpt_forward_time += gpt_forward_time
            print(f"✅ [DEBUG] GPT forward完成，耗时: {gpt_forward_time:.3f}s")
            
            # 🔍 检查GPT forward后的内存
            if "cuda" in str(self.device):
                try:
                    gpu_memory_after_gpt_forward = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"🔍 [DEBUG] GPT forward后GPU内存: {gpu_memory_after_gpt_forward:.2f} GB")
                except:
                    pass

            # BigVGAN 生成wav
            print(f"🎵 [DEBUG] 开始BigVGAN生成...")
            bigvgan_start = time.perf_counter()
            with torch.no_grad():  # 确保没有梯度计算
                wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
            bigvgan_time = time.perf_counter() - bigvgan_start
            cumulative_bigvgan_time += bigvgan_time
            print(f"✅ [DEBUG] BigVGAN完成，耗时: {bigvgan_time:.3f}s")
            
            # 🔍 检查BigVGAN后的内存
            if "cuda" in str(self.device):
                try:
                    gpu_memory_after_bigvgan = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"🔍 [DEBUG] BigVGAN后GPU内存: {gpu_memory_after_bigvgan:.2f} GB")
                except:
                    pass
            
            # 🎯 修复音频后处理 - 确保Android兼容性
            postprocess_start = time.perf_counter()
            
            # 确保单声道输出 (Android兼容性)
            if wav.dim() > 2:
                wav = wav.squeeze()
            if wav.dim() == 2:
                if wav.shape[0] == 1:
                    wav = wav.squeeze(0)  # 移除批次维度
                elif wav.shape[1] == 1:
                    wav = wav.squeeze(1)  # 移除声道维度
                else:
                    wav = wav[0]  # 取第一个声道
            
            # 确保wav是一维的
            if wav.dim() != 1:
                wav = wav.flatten()
            
            # 音频归一化和类型转换
            wav = torch.clamp(wav, -1.0, 1.0)  # 首先归一化到[-1,1]
            
            # 转换为CPU并保持float32格式（更好的兼容性）
            wav_cpu = wav.cpu().float()
            
            if verbose:
                print(f"wav shape: {wav_cpu.shape}", "min:", wav_cpu.min(), "max:", wav_cpu.max())
            
            postprocess_time = time.perf_counter() - postprocess_start
            print(f"🔄 [DEBUG] 音频后处理耗时: {postprocess_time:.3f}s, 输出shape: {wav_cpu.shape}")
            
            sentence_total_time = time.perf_counter() - sentence_start_time
            
            # 准备时间统计信息
            timing_info = {
                'sentence_total_time': sentence_total_time,
                'gpt_gen_time': gpt_gen_time,
                'gpt_forward_time': gpt_forward_time,
                'bigvgan_time': bigvgan_time,
                'token_time': token_time,
                'codes_process_time': codes_process_time,
                'silence_time': silence_time,
                'postprocess_time': postprocess_time,
                'cumulative_total_time': time.perf_counter() - start_time,
                'cumulative_gpt_gen_time': cumulative_gpt_gen_time,
                'cumulative_gpt_forward_time': cumulative_gpt_forward_time,
                'cumulative_bigvgan_time': cumulative_bigvgan_time,
            }
            
            print(f"⏱️ [DEBUG] 句子 {sentence_idx + 1} 总耗时: {sentence_total_time:.3f}s")
            print(f"📊 [DEBUG] 详细耗时分布:")
            print(f"    - GPT生成: {gpt_gen_time:.3f}s ({gpt_gen_time/sentence_total_time*100:.1f}%)")
            print(f"    - GPT前向: {gpt_forward_time:.3f}s ({gpt_forward_time/sentence_total_time*100:.1f}%)")
            print(f"    - BigVGAN: {bigvgan_time:.3f}s ({bigvgan_time/sentence_total_time*100:.1f}%)")
            print(f"    - 其他处理: {(sentence_total_time-gpt_gen_time-gpt_forward_time-bigvgan_time):.3f}s")
            
            # yield 当前句子的音频片段 - 返回float32格式的tensor
            yield {
                'audio_chunk': wav_cpu,  # 现在返回torch.Tensor (float32)
                'sample_rate': sampling_rate,
                'sentence_index': sentence_idx,
                'total_sentences': total_sentences,
                'sentence_text': ' '.join(sent),  # 重建句子文本用于显示
                'timing_info': timing_info
            }
            
            # 🔥 强制清理GPU缓存和变量，防止内存泄漏
            cache_start = time.perf_counter()
            
            # 🎯 关键：创建临时引用列表，确保所有中间张量都被删除
            temp_tensors = [codes, latent, wav, wav_cpu, text_tokens]
            if 'code_lens' in locals():
                temp_tensors.append(code_lens)
            
            # 批量删除所有临时张量
            for tensor in temp_tensors:
                if isinstance(tensor, torch.Tensor):
                    del tensor
            del temp_tensors
            
            # 🧠 清理GPT模型的内部状态和缓存
            if hasattr(self.gpt, 'inference_model') and hasattr(self.gpt.inference_model, 'transformer'):
                # 清理transformer的past_key_values缓存
                for layer in self.gpt.inference_model.transformer.h:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'past_key_value'):
                        try:
                            setattr(layer.attn, 'past_key_value', None)
                        except (AttributeError, TypeError):
                            # 如果设置失败，跳过该层
                            pass
            
            # 强制清理GPU缓存
            self.torch_empty_cache()
            
            # 额外的强制清理
            if "cuda" in str(self.device):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 同步CUDA操作
                    # 清理CUDA内存分配器的内存碎片
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            
            cache_time = time.perf_counter() - cache_start
            print(f"🧹 [DEBUG] 强化GPU缓存清理耗时: {cache_time:.3f}s")
            
            # 检查清理后的GPU内存
            if "cuda" in str(self.device):
                try:
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"🔍 [DEBUG] 清理后GPU内存: {gpu_memory_after:.2f} GB")
                except:
                    pass

        total_time = time.perf_counter() - start_time
        print(f"\n📊 [DEBUG] === 流式推理完成总结 ===")
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> Total streaming inference time: {total_time:.2f} seconds")
        print(f">> Total gpt_gen_time: {cumulative_gpt_gen_time:.2f} seconds ({cumulative_gpt_gen_time/total_time*100:.1f}%)")
        print(f">> Total gpt_forward_time: {cumulative_gpt_forward_time:.2f} seconds ({cumulative_gpt_forward_time/total_time*100:.1f}%)")
        print(f">> Total bigvgan_time: {cumulative_bigvgan_time:.2f} seconds ({cumulative_bigvgan_time/total_time*100:.1f}%)")
        print(f">> 平均每句耗时: {total_time/total_sentences:.2f} seconds")
        
        # 最终GPU内存检查
        if "cuda" in str(self.device):
            try:
                gpu_memory_final = torch.cuda.memory_allocated() / 1024**3  # GB
                print(f"🔍 [DEBUG] 最终GPU内存: {gpu_memory_final:.2f} GB")
            except:
                pass

    def infer_stream_pcm(self, audio_prompt, text, verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        """
        流式推理函数，逐句生成音频片段并返回16bit PCM字节数据
        
        Args:
            audio_prompt: 参考音频路径
            text: 要合成的文本
            verbose: 是否输出详细信息
            max_text_tokens_per_sentence: 每句最大token数
            **generation_kwargs: 生成参数
            
        Yields:
            dict: 包含音频片段信息的字典
                - audio_pcm_bytes: bytes, 16bit PCM音频数据
                - sample_rate: int, 采样率
                - sentence_index: int, 当前句子索引 (从0开始)
                - total_sentences: int, 总句子数
                - sentence_text: str, 当前句子的token文本
                - timing_info: dict, 时间统计信息
        """
        for chunk_info in self.infer_stream(audio_prompt, text, verbose, max_text_tokens_per_sentence, **generation_kwargs):
            # 获取音频数据 (torch.Tensor, float32, 单声道)
            audio_chunk = chunk_info['audio_chunk']
            
            # 🎯 完全模仿infer函数的tensor处理逻辑
            # 确保是一维张量
            if audio_chunk.dim() != 1:
                audio_chunk = audio_chunk.flatten()
            
            # 确保单声道输出，添加声道维度以符合处理要求
            if audio_chunk.dim() == 1:
                wav = audio_chunk.unsqueeze(0)  # (length,) -> (1, length)
            else:
                wav = audio_chunk
            
            # 计算音频长度用于检查
            sampling_rate = 24000
            wav_length = wav.shape[-1] / sampling_rate
            
            # 🎯 音频质量验证和Android兼容性处理 (完全模仿infer函数)
            if wav_length > 0:
                # 音频质量检查
                wav_max = wav.abs().max()
                if wav_max > 1.0:
                    if verbose:
                        print(f"⚠️ [DEBUG] 音频超出范围 (max={wav_max:.4f})，进行归一化")
                    wav = wav / wav_max
                
                # 确保没有NaN或Inf
                if torch.isnan(wav).any() or torch.isinf(wav).any():
                    if verbose:
                        print(f"❌ [DEBUG] 检测到NaN或Inf，使用零替换")
                    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
                
                if verbose:
                    print(f"✅ [DEBUG] 音频质量验证完成: shape={wav.shape}, range=[{wav.min():.4f}, {wav.max():.4f}]")

            # 转换为int16格式 (完全模仿infer函数)
            wav_int16 = (wav * 32767).clamp(-32767, 32767).type(torch.int16)
            
            # 转换为字节数据
            pcm_bytes = wav_int16.numpy().tobytes()
            
            # 修改返回的字典，替换 audio_chunk 为 PCM 字节数据
            yield {
                'audio_pcm_bytes': pcm_bytes,
                'sample_rate': chunk_info['sample_rate'],
                'sentence_index': chunk_info['sentence_index'],
                'total_sentences': chunk_info['total_sentences'],
                'sentence_text': chunk_info['sentence_text'],
                'timing_info': chunk_info['timing_info']
            }

    # 原始推理模式 - 现在调用 infer_stream 并拼接结果
    def infer(self, audio_prompt, text, output_path, verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        """
        传统的推理函数，现在内部调用 infer_stream 并拼接所有音频片段
        
        Args:
            audio_prompt: 参考音频路径
            text: 要合成的文本
            output_path: 输出文件路径，如果为None则返回Gradio格式
            verbose: 是否输出详细信息
            max_text_tokens_per_sentence: 每句最大token数
            **generation_kwargs: 生成参数
            
        Returns:
            str: 输出文件路径（如果指定了output_path）
            tuple: (sample_rate, wav_data) Gradio格式（如果output_path为None）
        """
        start_time = time.perf_counter()
        
        # 收集所有音频片段
        wav_chunks = []
        total_sentences = 0
        final_timing_info = None
        
        # 使用流式推理收集所有音频片段
        for chunk_info in self.infer_stream(audio_prompt, text, verbose, max_text_tokens_per_sentence, **generation_kwargs):
            # 现在audio_chunk已经是float32的torch.Tensor，直接添加
            wav_chunks.append(chunk_info['audio_chunk'])
            total_sentences = chunk_info['total_sentences']
            final_timing_info = chunk_info['timing_info']
            
            # 更新进度条
            progress = (chunk_info['sentence_index'] + 1) / total_sentences
            self._set_gr_progress(0.1 + 0.8 * progress, f"Processing sentence {chunk_info['sentence_index'] + 1}/{total_sentences}")
            
            if verbose:
                print(f">> Collected chunk {chunk_info['sentence_index'] + 1}/{total_sentences}, "
                      f"shape: {chunk_info['audio_chunk'].shape}")

        self._set_gr_progress(0.9, "Concatenating audio...")
        
        # 🎯 修复音频拼接逻辑 - 确保Android兼容性
        if len(wav_chunks) > 0:
            print(f"🔧 [DEBUG] 拼接 {len(wav_chunks)} 个音频片段")
            
            # 确保所有片段都是一维的
            normalized_chunks = []
            for i, chunk in enumerate(wav_chunks):
                if chunk.dim() != 1:
                    chunk = chunk.flatten()
                normalized_chunks.append(chunk)
                if verbose:
                    print(f"  - 片段 {i+1}: shape={chunk.shape}, min={chunk.min():.4f}, max={chunk.max():.4f}")
            
            # 拼接所有音频片段
            wav = torch.cat(normalized_chunks, dim=0)  # 沿时间轴拼接
            
            # 确保单声道输出，添加声道维度以符合torchaudio.save要求
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (length,) -> (1, length)
            
            print(f"🎯 [DEBUG] 拼接完成: shape={wav.shape}, dtype={wav.dtype}")
        else:
            # 处理空结果的情况
            sampling_rate = 24000
            wav = torch.zeros((1, 0), dtype=torch.float32)
            print(f"⚠️ [DEBUG] 没有音频片段，创建空音频")
            
        end_time = time.perf_counter()
        sampling_rate = 24000
        wav_length = wav.shape[-1] / sampling_rate
        
        # 打印最终统计信息
        if final_timing_info:
            print(f">> Total inference time: {end_time - start_time:.2f} seconds")
            print(f">> Generated audio length: {wav_length:.2f} seconds")
            print(f">> RTF: {(end_time - start_time) / wav_length:.4f}" if wav_length > 0 else ">> RTF: N/A (no audio generated)")

        # 🎯 音频质量验证和Android兼容性处理
        if wav_length > 0:
            # 音频质量检查
            wav_max = wav.abs().max()
            if wav_max > 1.0:
                print(f"⚠️ [DEBUG] 音频超出范围 (max={wav_max:.4f})，进行归一化")
                wav = wav / wav_max
            
            # 确保没有NaN或Inf
            if torch.isnan(wav).any() or torch.isinf(wav).any():
                print(f"❌ [DEBUG] 检测到NaN或Inf，使用零替换")
                wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"✅ [DEBUG] 音频质量验证完成: shape={wav.shape}, range=[{wav.min():.4f}, {wav.max():.4f}]")

        # 保存或返回音频
        if output_path:
            # 保存WAV文件 - 使用Android兼容的格式
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换为int16格式并保存 (Android通用格式)
            wav_int16 = (wav * 32767).clamp(-32767, 32767).type(torch.int16)
            
            # 保存时明确指定编码格式，确保Android兼容性
            torchaudio.save(
                output_path, 
                wav_int16, 
                sampling_rate,
                encoding="PCM_S",  # 16-bit PCM
                bits_per_sample=16
            )
            print(f">> wav file saved to: {output_path} (format: 16-bit PCM, {sampling_rate}Hz, mono)")
            return output_path
        else:
            # 返回以符合Gradio的格式要求 (int16 array)
            wav_int16 = (wav * 32767).clamp(-32767, 32767).type(torch.int16)
            wav_data = wav_int16.numpy()
            if wav_data.ndim == 2:
                wav_data = wav_data.T  # Gradio期望 (time, channels) 格式
            return (sampling_rate, wav_data)

    def infer_opus(self, audio_prompt, text, verbose=False, max_text_tokens_per_sentence=120, 
                   opus_bitrate=32000, opus_complexity=10, **generation_kwargs):
        """
        流式推理函数，逐句生成音频片段并返回 OGG 容器中的 Opus 编码音频数据流
        
        Args:
            audio_prompt: 参考音频路径
            text: 要合成的文本
            verbose: 是否输出详细信息
            max_text_tokens_per_sentence: 每句最大token数
            opus_bitrate: Opus编码比特率 (默认32kbps，可选: 8000-512000)
            opus_complexity: Opus编码复杂度 (0-10，越高质量越好但编码越慢)
            **generation_kwargs: 生成参数
            
        Yields:
            bytes: OGG容器格式的音频数据块 (内含Opus编码音频，ExoPlayer完美支持)
        """
        import threading
        import queue
        
        print(">> Starting OGG (Opus) streaming inference...")
        
        # 检查 ffmpeg 是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg 未找到。请安装 FFmpeg 以支持 Opus 编码。")
        
        # 验证 Opus 参数
        if not (8000 <= opus_bitrate <= 512000):
            raise ValueError(f"Opus bitrate must be between 8000 and 512000, got {opus_bitrate}")
        if not (0 <= opus_complexity <= 10):
            raise ValueError(f"Opus complexity must be between 0 and 10, got {opus_complexity}")
        
        if verbose:
            print(f"🎵 [DEBUG] Opus 配置: bitrate={opus_bitrate}bps, complexity={opus_complexity}")
        
        # 🎵 在循环外初始化 FFmpeg 进程
        opus_sample_rate = 48000  # Opus 标准采样率
        original_sample_rate = 24000  # TTS 输出采样率
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-flags', 'low_delay',
            '-f', 'f32le',  # 输入格式：32-bit float little-endian
            '-ar', str(original_sample_rate),  # 输入采样率
            '-ac', '1',  # 单声道
            '-i', 'pipe:0',  # 从 stdin 读取
            '-c:a', 'libopus',  # 使用 Opus 编码器
            '-b:a', str(opus_bitrate),  # 设置比特率
            '-compression_level', str(opus_complexity),  # 设置复杂度
            '-frame_duration', '100',  # 100ms帧持续时间，适合实时传输
            '-application', 'lowdelay',  # 优先保证音频质量和完整性
            '-ar', str(opus_sample_rate),  # 输出采样率
            '-f', 'ogg',  # 使用OGG容器
            '-flush_packets', '1',  # 强制刷新包
            '-y',  # 覆盖输出
            'pipe:1'  # 输出到 stdout
        ]
        
        print(f"🔧 [DEBUG] 启动 FFmpeg: {' '.join(ffmpeg_cmd)}")
            
        
        # 启动 FFmpeg 进程
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # 无缓冲
            )
        except Exception as e:
            print(f"Failed to start FFmpeg: {e}")
            raise RuntimeError(f"Failed to start FFmpeg: {e}")
        
        # 检查进程是否立即退出
        import time
        time.sleep(0.1)
        if ffmpeg_process.poll() is not None:
            raise RuntimeError(f"FFmpeg process exited immediately with code: {ffmpeg_process.returncode}")
        
        # 创建输出队列用于异步读取
        output_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def read_output():
            """异步读取 FFmpeg 输出"""
            try:
                if ffmpeg_process.stdout is not None:
                    while True:
                        # 读取固定大小的数据块
                        data = ffmpeg_process.stdout.read(4096)
                        if not data:
                            break
                        output_queue.put(data)
            except Exception as e:
                error_queue.put(f"读取输出失败: {e}")
            finally:
                output_queue.put(None)  # 结束标记
        
        def read_error():
            """异步读取 FFmpeg 错误输出"""
            try:
                if ffmpeg_process.stderr is not None:
                    while True:
                        error_line = ffmpeg_process.stderr.readline()
                        if not error_line:
                            break
                        error_msg = error_line.decode('utf-8', errors='ignore').strip()
                        if error_msg:
                            print(f"🔴 [FFmpeg ERROR] {error_msg}")
                            error_queue.put(error_msg)
            except Exception as e:
                error_msg = f"读取错误输出失败: {e}"
                print(f"🔴 [ERROR] {error_msg}")
                error_queue.put(error_msg)
        
        # 启动异步读取线程
        output_thread = threading.Thread(target=read_output, daemon=True)
        error_thread = threading.Thread(target=read_error, daemon=True)
        output_thread.start()
        error_thread.start()
        
        total_opus_size = 0
        chunk_count = 0
        
        # FFmpeg will generate OGG header when it receives actual audio data
        
        try:
            # 🎵 先发送一小段静音来"预热"FFmpeg，这依然是一个好习惯
            try:
                silence_duration_ms = 20
                num_samples = int(original_sample_rate * silence_duration_ms / 1000)
                silence = torch.zeros(num_samples, dtype=torch.float32)
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.write(silence.numpy().tobytes())
                    ffmpeg_process.stdin.flush()
                if verbose:
                    print(f"🎤 [DEBUG] Primed FFmpeg with {silence_duration_ms}ms of silence.")
            except Exception as e:
                print(f"⚠️ [WARNING] Failed to send priming silence to FFmpeg: {e}")

            # 🔄 使用流式推理获取音频片段并发送给 FFmpeg
            for chunk_info in self.infer_stream(audio_prompt, text, verbose, max_text_tokens_per_sentence, **generation_kwargs):
                chunk_start_time = time.perf_counter()
                
                # 获取音频数据 (torch.Tensor, float32, 单声道)
                audio_chunk = chunk_info['audio_chunk']
                
                # 🎯 音频预处理
                if audio_chunk.dim() != 1:
                    audio_chunk = audio_chunk.flatten()
                
                # 检查音频长度，避免发送空音频
                if audio_chunk.numel() == 0:
                    if verbose:
                        print(f"⚠️ [DEBUG] 跳过空音频片段 (句子 {chunk_info['sentence_index'] + 1})")
                    continue
                
                # 转换为 numpy 数组并归一化
                audio_np = audio_chunk.clamp(-1.0, 1.0).numpy()
                
                try:
                    # 发送音频数据到 FFmpeg
                    if ffmpeg_process.stdin is not None:
                        ffmpeg_process.stdin.write(audio_np.tobytes())
                        ffmpeg_process.stdin.flush()
                    
                    chunk_count += 1
                    
                    # 🎵 读取所有已经可用的Opus输出数据
                    # 在非低延迟模式下，FFmpeg会缓冲数据，所以我们在这里非阻塞地拉取
                    while True:
                        try:
                            opus_data = output_queue.get_nowait()
                            if opus_data: # is not None
                                total_opus_size += len(opus_data)
                                yield opus_data
                            else: # is None, stream ended prematurely
                                break
                        except queue.Empty:
                            # 队列为空，表示当前没有可用的输出，继续处理下一个音频块
                            break
                    
                    # 检查FFmpeg进程状态
                    if ffmpeg_process.poll() is not None:
                        print(f"FFmpeg process exited with code: {ffmpeg_process.returncode}")
                        break
                        
                except BrokenPipeError:
                    print("❌ [ERROR] FFmpeg 进程意外终止")
                    break
                except Exception as e:
                    print(f"❌ [ERROR] 发送数据到 FFmpeg 失败: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    break
            
            # 🔚 完成所有音频后，关闭 stdin 并读取剩余输出
            if verbose:
                print("🔚 [DEBUG] 关闭 FFmpeg 输入流...")
                
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.close()
            
            # 读取剩余的输出数据
            final_start = time.perf_counter()
            while True:
                try:
                    opus_data = output_queue.get(timeout=5.0)  # 5秒超时
                    if opus_data is None:
                        break
                    elif isinstance(opus_data, bytes) and len(opus_data) > 0:
                        total_opus_size += len(opus_data)
                        if verbose:
                            print(f"📦 [DEBUG] 收到最终 Opus 数据: {len(opus_data):,} bytes")
                        yield opus_data
                except queue.Empty:
                    # 超时，可能没有更多数据
                    if verbose:
                        print("⏰ [DEBUG] 等待最终输出超时")
                    break
            
            if verbose:
                final_time = time.perf_counter() - final_start
                print(f"🔚 [DEBUG] 最终数据读取耗时: {final_time:.3f}s")
        
        finally:
            # 清理资源
            try:
                if ffmpeg_process.poll() is None:
                    ffmpeg_process.terminate()
                    ffmpeg_process.wait(timeout=5.0)
            except Exception as e:
                try:
                    ffmpeg_process.kill()
                except:
                    pass
        
        print(f">> OGG (Opus) streaming completed: {chunk_count} chunks, {total_opus_size:,} bytes")



