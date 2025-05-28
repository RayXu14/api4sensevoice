import asyncio
import websockets
import sounddevice
import json
import time
import argparse
import math
import re

# Upstream data format: PCM binary
CHANNELS = 1
SAMPLERATE = 16000
DTYPE = "int16"

class AudioStreamer:
    def __init__(self, block_ms):
        self.audio_queue = asyncio.Queue()

        self.send_times = []  # List of (chunk_id, send_time) tuples
        self.chunk_counter = 0
        self.last_speech_detection_time = None  # Track when speech was last detected
        self.detection_latencies = []  # Track audio-to-detection latencies

        print("Initializing audio stream...")
        loop = asyncio.get_event_loop()

        def audio_callback(indata, frame_count, time_info, status):
            if status:
                print(f"Audio status warning: {status}")
            
            loop.call_soon_threadsafe(
                self.audio_queue.put_nowait, 
                bytes(indata)
            )

        blocksize = int(SAMPLERATE * block_ms / 1000)
        self.stream = sounddevice.RawInputStream(
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            callback=audio_callback,
            blocksize=blocksize,
            dtype=DTYPE
        )
        print(f"Audio stream created - block duration: {block_ms}ms ({blocksize} frames)")


    async def streaming(self, websocket):
        self.stream.start()
        print(f"Audio stream started")

        while True:
            audio_data = await self.audio_queue.get()
            send_time = time.time()
            chunk_id = self.chunk_counter
            self.send_times.append((chunk_id, send_time))
            self.chunk_counter += 1
            await websocket.send(audio_data)

def calculate_detection_latency(audio_streamer, detection_time):
    """Calculate approximate latency from recent audio activity to speech detection"""
    if audio_streamer.send_times:
        # This is an approximation since we can't precisely correlate 
        # which audio chunks triggered this detection
        # Since send_times is ordered by time, just get the last one
        recent_send_time = audio_streamer.send_times[-1][1]
        detection_latency = (detection_time - recent_send_time) * 1000
        
        if detection_latency > 0:
            audio_streamer.detection_latencies.append(detection_latency)
            avg_detection_latency = sum(audio_streamer.detection_latencies[-10:]) / len(audio_streamer.detection_latencies[-10:])
            print(f"   🎯 Speech detection response time: {detection_latency:.1f}ms | Average: {avg_detection_latency:.1f}ms")
        else:
            print(f"   ⚠️  Detection timing anomaly (negative latency: {detection_latency:.1f}ms)")
            assert False, 'Are you sure we have negative latency?'

def calculate_latency(audio_streamer, receive_time, latencies):
    """Calculate and display latency from last speech detection to transcription result"""
    if audio_streamer.last_speech_detection_time:
        # Calculate latency from the last speech detection time
        latency = (receive_time - audio_streamer.last_speech_detection_time) * 1000
        
        if latency > 0:
            latencies.append(latency)
            avg_latency = sum(latencies[-10:]) / len(latencies[-10:])
            print(f"   ⚡ Transcription latency: {latency:.1f}ms | last 10 average: {avg_latency:.1f}ms")
        else:
            print(f"   ⚠️  ASR timing anomaly (negative latency: {latency:.1f}ms)")
            assert False, 'Are you sure we have negative latency?'
        
        # Clean up old send times to prevent memory growth
        # Since list is ordered by time, just keep the most recent entries
        if len(audio_streamer.send_times) > 100:
            audio_streamer.send_times = audio_streamer.send_times[-50:]

def extract_tags(text):
    """提取语种标签、情感标签和事件标签"""
    # 找到所有的标签
    tags = re.findall(r'<\|([^|]+)\|>', text)
    
    # 按顺序提取：第一个是语言，第二个是情感，第三个是事件
    language = tags[0] if len(tags) > 0 else None
    emotion = tags[1] if len(tags) > 1 else None
    event = tags[2] if len(tags) > 2 else None
    
    return language, emotion, event

def handle_response(response_data, receive_time, audio_streamer, latencies):
    """Handle server response"""
    code = response_data.get('code', 'unknown')
    info_str = response_data.get('info', '')
    data = response_data.get('data', '')
    
    print(f"\n📨 Complete response info:")
    print(f"   Code: {code}")
    print(f"   Info: {info_str}")
    print(f"   Data: {data}")
    
    if code == 0 and data and str(data).strip():
        if info_str:
            try:
                info_json = json.loads(info_str)
                text_field = info_json.get('text', '')
                if text_field:
                    language, emotion, event = extract_tags(text_field)
                    print(f"🎤 Transcribed text: {data}")
                    if language:
                        print(f"   🌐 Language: {language}")
                    if emotion:
                        print(f"   😊 Emotion: {emotion}")
                    if event:
                        print(f"   🎵 Event: {event}")
            except json.JSONDecodeError:
                pass
            calculate_latency(audio_streamer, receive_time, latencies)
    elif code == 2:
        if 'detect speech' in info_str:
            calculate_detection_latency(audio_streamer, receive_time)
            audio_streamer.last_speech_detection_time = receive_time
        else:
            print(f"❓ Unknown System info: {info_str}")
            assert False
    else:
        print(f"❓ Unknown response type: {code}")
        assert False

async def receive_responses(websocket, audio_streamer):
    latencies = []
    
    while True:
        response = await websocket.recv()
        receive_time = time.time()
        
        try:
            response_data = json.loads(response)
            handle_response(response_data, receive_time, audio_streamer, latencies)
        except json.JSONDecodeError:
            print(f"⚠️  Received non-JSON response: {response}")

async def connect_to_server(url, block_ms):
    audio_streamer = AudioStreamer(block_ms)

    print(f"Connecting to server: {url}")
    
    websocket = await websockets.connect(url)
    print("WebSocket connection successful")
        
    await asyncio.gather(
            audio_streamer.streaming(websocket),
            receive_responses(websocket, audio_streamer)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host address (default: localhost)')
    parser.add_argument('--port', '-p', type=int, default=27000, 
                       help='Server port number (default: 27000)')
    parser.add_argument('--block-ms', type=int, default=10,
                       help='Audio block duration in milliseconds (default: 10ms). Use 300ms for better latency measurement accuracy (aligns with server processing), maybe...')
    args = parser.parse_args()
    
    print("🎤 Real-time Speech Recognition Client")
    print("="*50, "Press Ctrl+C to stop", "="*50, sep='\n')
    try:
        url = f"ws://{args.host}:{args.port}/ws/transcribe"
        asyncio.run(connect_to_server(url, args.block_ms))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")