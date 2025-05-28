import asyncio
import websockets
import sounddevice
import json
import time

# 连接配置
PORT = 10001
URL = f"ws://localhost:{PORT}/ws/transcribe"

# 音频配置
CHANNELS = 1
SAMPLERATE = 16000
BLOCKSIZE = 512
DTYPE = "int16"

# 优化配置
SHOW_LATENCY = True

class AudioStreamer:
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.loop = None
        self.send_times = {}
        self.chunk_counter = 0
        
    def audio_callback(self, indata, frame_count, time_info, status):
        if status:
            print(f"音频状态警告: {status}")
        
        try:
            self.loop.call_soon_threadsafe(
                self.input_queue.put_nowait, 
                (bytes(indata), status, time.time())
            )
        except Exception as e:
            print(f"音频队列错误: {e}")

    async def stream_audio(self):
        self.loop = asyncio.get_event_loop()
        print("初始化音频流...")
        
        try:
            with sounddevice.RawInputStream(
                channels=CHANNELS, samplerate=SAMPLERATE,
                callback=self.audio_callback, blocksize=BLOCKSIZE, dtype=DTYPE
            ):
                print(f"音频流启动 - 块大小: {BLOCKSIZE} (~{BLOCKSIZE/SAMPLERATE*1000:.1f}ms)")
                
                while True:
                    try:
                        yield await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            print(f"音频流错误: {e}")
            raise

async def send_audio_data(websocket, audio_streamer):
    try:
        async for audio_data, status, timestamp in audio_streamer.stream_audio():
            if SHOW_LATENCY:
                chunk_id = audio_streamer.chunk_counter
                audio_streamer.send_times[chunk_id] = timestamp
                audio_streamer.chunk_counter += 1
            
            await websocket.send(audio_data)
    except (websockets.exceptions.ConnectionClosed, Exception) as e:
        print(f"发送音频数据错误: {e}")

def parse_info_data(info_str):
    """解析并打印详细信息"""
    try:
        info_data = json.loads(info_str)
        print(f"📋 详细信息:")
        for key, value in info_data.items():
            if key == 'avg_logprob':
                confidence = round((value + 1) * 100, 1)
                print(f"   📊 置信度: {confidence}% (logprob: {value:.3f})")
            elif key == 'emotion':
                print(f"   😊 情感标签: {value}")
            elif key == 'language':
                print(f"   🌍 语言标签: {value}")
            elif key == 'speaker':
                print(f"   👤 说话人: {value}")
            elif key == 'timestamp':
                print(f"   ⏰ 时间戳: {value}")
            else:
                print(f"   {key}: {value}")
        return info_data
    except json.JSONDecodeError:
        print(f"   ⚠️  Info字段不是有效JSON: {info_str}")
        return None

def calculate_latency(audio_streamer, receive_time, latencies):
    """计算并显示延迟"""
    if SHOW_LATENCY and audio_streamer.send_times:
        recent_sends = list(audio_streamer.send_times.values())[-10:]
        avg_send_time = sum(recent_sends) / len(recent_sends)
        latency = (receive_time - avg_send_time) * 1000
        latencies.append(latency)
        avg_latency = sum(latencies[-10:]) / len(latencies[-10:])
        print(f"   ⚡ 延迟: {latency:.1f}ms | 平均: {avg_latency:.1f}ms")

def handle_response(response_data, receive_time, audio_streamer, latencies):
    """处理服务器响应"""
    code = response_data.get('code', 'unknown')
    info_str = response_data.get('info', '')
    data = response_data.get('data', '')
    
    print(f"\n📨 完整响应信息:")
    print(f"   Code: {code}")
    print(f"   Info: {info_str}")
    print(f"   Data: {data}")
    
    if code == 0 and data and str(data).strip():
        # 转录结果
        print(f"\n🎤 转录文本: {data}")
        if info_str:
            parse_info_data(info_str)
            calculate_latency(audio_streamer, receive_time, latencies)
    elif code == 2:
        # 系统信息
        if 'detect speech' in info_str:
            print("🔊 检测到语音开始...")
        elif 'detect speaker' in info_str:
            print(f"👤 检测到说话人: {data}")
        else:
            print(f"ℹ️  系统信息: {info_str}")
            if data:
                print(f"   数据: {data}")
    else:
        print(f"❓ 未知响应类型")

async def receive_responses(websocket, audio_streamer):
    latencies = []
    
    try:
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                receive_time = time.time()
                
                try:
                    response_data = json.loads(response)
                    handle_response(response_data, receive_time, audio_streamer, latencies)
                except json.JSONDecodeError:
                    print(f"⚠️  收到非JSON响应: {response}")
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket连接已关闭")
                break
            except Exception as e:
                print(f"接收响应错误: {e}")
                break
    except Exception as e:
        print(f"响应接收任务错误: {e}")
    
    # 显示延迟统计
    if SHOW_LATENCY and latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n📊 延迟统计 (共{len(latencies)}次):")
        print(f"   平均延迟: {avg_latency:.1f}ms")
        print(f"   最小延迟: {min(latencies):.1f}ms") 
        print(f"   最大延迟: {max(latencies):.1f}ms")

async def connect_to_server():
    audio_streamer = AudioStreamer()
    
    try:
        print(f"连接到服务器: {URL}")
        
        async with websockets.connect(URL) as websocket:
            print("WebSocket连接成功")
            
            tasks = [
                asyncio.create_task(send_audio_data(websocket, audio_streamer)),
                asyncio.create_task(receive_responses(websocket, audio_streamer))
            ]
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                print("用户中断")
                for task in tasks:
                    task.cancel()
                
    except Exception as e:
        print(f"连接错误: {e}")

async def main():
    print("🎤 实时语音识别客户端")
    print("="*50)
    print(f"音频块大小: {BLOCKSIZE} (~{BLOCKSIZE/SAMPLERATE*1000:.1f}ms)")
    print(f"服务器地址: {URL}")
    print("按 Ctrl+C 停止")
    print("="*50)
    
    try:
        await connect_to_server()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    asyncio.run(main()) 