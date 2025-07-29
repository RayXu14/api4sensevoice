# @Time    : 2025/7/29 04:37
# @File    : test_noise_wss.py.py
# @text    : 增加降噪处理的vad+asr
# terminal: export ENABLE_NOISE_SUPPRESSION=true,调用降噪
# todo: 目前出现运行中中断的现象（未报错、未结束、单纯停下来），原因暂时不明，后期使用需要debug
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger
import sys
import json
import traceback
import time

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40, colorize=True)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40, colorize=True)

# ==== Metrics description block ====
metrics_description = """
实时音频参数说明:
chunk_rms_db: 当前 chunk 的 RMS 分贝 (dBFS)，表示瞬时能量大小。
noise_floor_db: 自适应估计的噪声底分贝 (dBFS)。
snr: 当前 chunk 相对于噪声底的信噪比 (dB)。
ema_rms_db: RMS 的指数滑动平均分贝，平滑后的能量。
peak_rms_db: 运行以来出现过的最大 RMS 分贝。
peak_amp: 运行以来出现过的最大幅度 (线性 0~1)。
chunk_peak_db: 当前 chunk 的峰值幅度分贝 (dBFS)。
crest_factor_db: 峰值与 RMS 之间的差值 (dB)，反映瞬态程度。
zcr: Zero Crossing Rate，零交叉率，反映频谱粗略特征/活跃度。
dc_offset: 直流偏移，理想情况下应接近 0。
"""
logger.info(metrics_description.strip())


class Config(BaseSettings):
    sv_thr: float = Field(0.3, description="Speaker verification threshold")
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")
    # 新增降噪配置
    enable_noise_suppression: bool = Field(False, description="Enable noise suppression")
    noise_suppression_model: str = Field("iic/speech_frcrn_ans_cirm_16k", description="Noise suppression model")
    enhanced_audio_rms_threshold: float = Field(0.015, description="Enhanced audio RMS threshold for VAD")


config = Config()

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    }

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
    }

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
    }

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
    }

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", }


def format_str(s):
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        # else:
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
    model_revision='v1.0.0'
    )

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    device="cuda:0",
    disable_update=True
    )

model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    disable_update=True
    )

# model_vad will be initialized after parsing command line arguments
model_vad = None
# noise suppression pipeline will be initialized if enabled
ans_pipeline = None

reg_spks_files = [
    "speaker/speaker1_a_cn_16k.wav"
    ]


def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr": sr,
            }
    return reg_spk


reg_spks = reg_spk_init(reg_spks_files)


def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow


def audio_norm(wav, target_dB=-25, eps=1e-6):
    """音频标准化"""
    if len(wav) == 0:
        return wav
    rms = (wav ** 2).mean() ** 0.5
    scalar = 10 ** (target_dB / 20) / (rms + eps)
    wav = wav * scalar
    return wav


def apply_noise_suppression(chunk_audio):
    """
    对音频chunk应用降噪处理

    :param chunk_audio: numpy array, float32 格式的音频数据
    :return: 降噪后的音频数据
    """
    global ans_pipeline

    if ans_pipeline is None:
        return chunk_audio

    try:
        start_time = time.time()

        # 先进行音频标准化
        normalized_audio = audio_norm(chunk_audio).astype(np.float32)

        # 转换为int16格式并创建WAV文件
        audio_int16 = (normalized_audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        wav_data = create_wav_header(audio_bytes, sample_rate=config.sample_rate,
                                     num_channels=config.channels, bits_per_sample=config.bit_depth)

        # 应用降噪
        result = ans_pipeline(wav_data)
        enhanced_audio_bytes = result['output_pcm']

        # 转换回float32格式
        enhanced_audio_int16 = np.frombuffer(enhanced_audio_bytes, dtype=np.int16)
        enhanced_audio_float32 = enhanced_audio_int16.astype(np.float32) / 32767.0

        # 确保输出长度与输入一致
        if len(enhanced_audio_float32) != len(chunk_audio):
            logger.warning(f"Audio length mismatch: input {len(chunk_audio)}, output {len(enhanced_audio_float32)}")
            if len(enhanced_audio_float32) > len(chunk_audio):
                enhanced_audio_float32 = enhanced_audio_float32[:len(chunk_audio)]
            else:
                # 如果输出较短，用原音频补齐
                enhanced_audio_float32 = np.pad(enhanced_audio_float32, (0, len(chunk_audio) - len(enhanced_audio_float32)), 'constant')

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        logger.debug(f"noise suppression elapsed: {elapsed_time:.2f} milliseconds")

        return enhanced_audio_float32

    except Exception as e:
        logger.error(f"Noise suppression failed: {e}")
        return chunk_audio


def speaker_verify(audio, sv_thr):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
            hit = True
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {k}: {res_sv}")
    return hit, k


def asr(audio, lang, cache, use_itn=False):
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    start_time = time.time()
    result = model_asr.generate(
        input=audio,
        cache=cache,
        language=lang.strip(),
        use_itn=use_itn,
        batch_size_s=60,
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
            ).model_dump()
        )


# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()

        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr = {}
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False

        # 添加长音频缓冲区（1秒）
        long_buffer_size = config.sample_rate  # 1秒=16000样本
        long_audio_buffer = np.array([], dtype=np.float32)
        speech_active = False

        # ====== Real-time audio metrics initialization ======
        noise_floor_rms = 1e-4  # initial noise floor estimate
        ema_rms = 0.0  # exponential moving average RMS
        peak_rms = 0.0  # peak RMS observed
        peak_amp = 0.0  # peak amplitude observed
        eps = 1e-10  # small epsilon to avoid log(0)

        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"received {len(data)} bytes")

            buffer += data
            if len(buffer) < 2:
                continue

            audio_buffer = np.append(
                audio_buffer,
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
                )

            # with open('buffer.pcm', 'ab') as f:
            #     logger.debug(f'write {f.write(buffer[:len(buffer) - (len(buffer) % 2)])} bytes to `buffer.pcm`')

            buffer = buffer[len(buffer) - (len(buffer) % 2):]

            # while len(audio_buffer) >= chunk_size:
            #     chunk = audio_buffer[:chunk_size]
            #     audio_buffer = audio_buffer[chunk_size:]
            #
            #     # 先进行VAD检测（原始音频）
            #     audio_vad = np.append(audio_vad, chunk)
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                # 维护长音频缓冲区
                long_audio_buffer = np.append(long_audio_buffer, chunk)
                if len(long_audio_buffer) > long_buffer_size:
                    long_audio_buffer = long_audio_buffer[-long_buffer_size:]

                # 对长音频缓冲区降噪（如果启用且缓冲区已满）
                vad_chunk = chunk  # 默认使用原始chunk

                if config.enable_noise_suppression and ans_pipeline is not None and len(long_audio_buffer) == long_buffer_size:
                    enhanced_long = apply_noise_suppression(long_audio_buffer)
                    vad_chunk = enhanced_long[-chunk_size:]  # 取最新300ms

                    # 降噪后音量判断，只在非语音状态检测音量
                    if not speech_active:
                        enhanced_rms = np.sqrt(np.mean(vad_chunk ** 2))
                        if enhanced_rms < config.enhanced_audio_rms_threshold:
                            logger.debug(f"Low volume, skipping VAD")
                            continue

                # VAD使用处理后的chunk
                audio_vad = np.append(audio_vad, vad_chunk)

                # ====== Real-time audio metrics computation & logging ======
                # Compute RMS of current chunk
                rms = float(np.sqrt(np.mean(np.square(chunk)) + eps))
                chunk_rms_db = 20 * np.log10(rms + eps)

                # Update noise floor (adaptive) using chunks likely to be noise
                noise_floor_rms = (0.95 * noise_floor_rms + 0.05 * rms) if rms < noise_floor_rms * 1.5 else noise_floor_rms
                noise_floor_db = 20 * np.log10(noise_floor_rms + eps)

                # Signal-to-Noise Ratio
                snr_db = chunk_rms_db - noise_floor_db

                # Exponential moving average RMS
                ema_rms = 0.9 * ema_rms + 0.1 * rms
                ema_rms_db = 20 * np.log10(ema_rms + eps)

                # Peak metrics
                if rms > peak_rms:
                    peak_rms = rms
                peak_rms_db = 20 * np.log10(peak_rms + eps)

                peak_amp_current = float(np.max(np.abs(chunk)))
                if peak_amp_current > peak_amp:
                    peak_amp = peak_amp_current

                # Other real-time metrics
                chunk_peak_amp = peak_amp_current
                chunk_peak_db = 20 * np.log10(chunk_peak_amp + eps)
                crest_factor_db = chunk_peak_db - chunk_rms_db
                # Zero Crossing Rate
                zcr = float(((chunk[:-1] * chunk[1:]) < 0).sum()) / len(chunk)
                # DC offset
                dc_offset = float(np.mean(chunk))

                # ANSI color codes
                COLOR_CODES = {
                    "chunk_rms_db": "\033[91m",  # Red
                    "noise_floor_db": "\033[92m",  # Green
                    "snr_db": "\033[93m",  # Yellow
                    "ema_rms_db": "\033[94m",  # Blue
                    "peak_rms_db": "\033[95m",  # Magenta
                    "peak_amp": "\033[96m",  # Cyan
                    "chunk_peak_db": "\033[90m",  # Bright Black (Grey)
                    "crest_factor_db": "\033[97m",  # Bright White
                    "zcr": "\033[90m",  # Grey
                    "dc_offset": "\033[97m",  # Bright White
                    }
                RESET_COLOR = "\033[0m"

                metrics_message = (
                    f"{COLOR_CODES['chunk_rms_db']}chunk_rms_db: {chunk_rms_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['noise_floor_db']}noise_floor_db: {noise_floor_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['snr_db']}snr: {snr_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['ema_rms_db']}ema_rms_db: {ema_rms_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['peak_rms_db']}peak_rms_db: {peak_rms_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['peak_amp']}peak_amp: {peak_amp:.3f}{RESET_COLOR} | "
                    f"\n"
                    f"{COLOR_CODES['chunk_peak_db']}chunk_peak_db: {chunk_peak_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['crest_factor_db']}crest_factor_db: {crest_factor_db:.2f} dB{RESET_COLOR} | "
                    f"{COLOR_CODES['zcr']}zcr: {zcr:.3f}{RESET_COLOR} | "
                    f"{COLOR_CODES['dc_offset']}dc_offset: {dc_offset:.4f}{RESET_COLOR}"
                )
                # logger.debug(metrics_message)

                # with open('chunk.pcm', 'ab') as f:
                #     logger.debug(f'write {f.write(chunk)} bytes to `chunk.pcm`')

                if last_vad_beg > 1:
                    if sv:
                        # speaker verify
                        # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                        # `hit` will reset after `asr`.
                        if not hit:
                            hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * config.sample_rate / 1000):], config.sv_thr)
                            if hit:
                                response = TranscriptionResponse(
                                    code=2,
                                    info="detect speaker",
                                    data=speaker
                                    )
                                await websocket.send_json(response.model_dump())
                    else:
                        response = TranscriptionResponse(
                            code=2,
                            info="detect speech",
                            data=''
                            )
                        await websocket.send_json(response.model_dump())

                # res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                res = model_vad.generate(input=vad_chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:  # speech begin
                            last_vad_beg = segment[0]
                            speech_active = True
                        if segment[1] > -1:  # speech end
                            last_vad_end = segment[1]
                            speech_active = False
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")

                            # 对语音段应用降噪（如果启用）
                            speech_segment = audio_vad[beg:end]
                            if config.enable_noise_suppression and ans_pipeline is not None:
                                original_segment = speech_segment.copy()
                                speech_segment = apply_noise_suppression(speech_segment)

                                # 调试：比较降噪前后的音频特征
                                orig_rms = np.sqrt(np.mean(original_segment ** 2))
                                enhanced_rms = np.sqrt(np.mean(speech_segment ** 2))
                                logger.debug(f"Speech segment RMS before: {orig_rms:.6f}, after: {enhanced_rms:.6f}")

                            result = None if sv and not hit else asr(speech_segment, lang.strip(), cache_asr, True)
                            logger.info(f"asr response: {result}")
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False

                            if result is not None:
                                response = TranscriptionResponse(
                                    code=0,
                                    info=json.dumps(result[0], ensure_ascii=False),
                                    data=format_str_v3(result[0]['text'])
                                    )
                                await websocket.send_json(response.model_dump())

                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    parser.add_argument('--vad_ms_after_speech', type=int, default=500, help='VAD max end silence time in milliseconds.')
    parser.add_argument('--speech_noise_thres', type=float, default=0.8, help='Speech noise threshold for VAD (0.0-1.0, higher = less sensitive).')
    # parser.add_argument('--certfile', type=str, default='path_to_your_SSL_certificate_file.crt', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_SSL_certificate_file.key', help='SSL key file')
    args = parser.parse_args()

    model_vad = AutoModel(
        model="fsmn-vad",
        model_revision="v2.0.4",
        disable_pbar=True,
        max_end_silence_time=args.vad_ms_after_speech,
        speech_noise_thres=args.speech_noise_thres,
        disable_update=True,
        )

    # 初始化降噪模型（如果启用）
    if config.enable_noise_suppression:
        try:
            logger.info(f"Initializing noise suppression model: {config.noise_suppression_model}")
            ans_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=config.noise_suppression_model
                )
            logger.info("Noise suppression model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize noise suppression model: {e}")
            ans_pipeline = None
    else:
        logger.info("Noise suppression is disabled")
        ans_pipeline = None

    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

