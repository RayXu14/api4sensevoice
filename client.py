import asyncio
import websockets
import sounddevice
import json
import time
import argparse

# Upstream data format: PCM binary
CHANNELS = 1
SAMPLERATE = 16000
DTYPE = "int16"

class AudioStreamer:
    def __init__(self, block_ms):
        self.input_queue = asyncio.Queue()
        self.loop = None
        self.send_times = {}
        self.chunk_counter = 0
        self.last_speech_detection_time = None  # Track when speech was last detected
        self.detection_latencies = []  # Track audio-to-detection latencies
        self.block_ms = block_ms
        self.blocksize = int(SAMPLERATE * block_ms / 1000)  # Convert milliseconds to frames

    async def stream_audio(self):
        self.loop = asyncio.get_event_loop()
        print("Initializing audio stream...")

        def audio_callback(indata, frame_count, time_info, status):
            if status:
                print(f"Audio status warning: {status}")
            
            self.loop.call_soon_threadsafe(
                self.input_queue.put_nowait, 
                bytes(indata)
            )

        try:
            with sounddevice.RawInputStream(
                channels=CHANNELS,
                samplerate=SAMPLERATE,
                callback=audio_callback,
                blocksize=self.blocksize,
                dtype=DTYPE
            ):
                print(f"Audio stream started - block duration: {self.block_ms}ms ({self.blocksize} frames)")
                
                while True:
                    try:
                        yield await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            print(f"Audio stream error: {e}")
            raise

async def send_audio_data(websocket, audio_streamer):
    try:
        async for audio_data in audio_streamer.stream_audio():
            chunk_id = audio_streamer.chunk_counter
            # Record the actual send time, not the capture time
            send_time = time.time()
            audio_streamer.send_times[chunk_id] = send_time
            audio_streamer.chunk_counter += 1
            
            await websocket.send(audio_data)
    except (websockets.exceptions.ConnectionClosed, Exception) as e:
        print(f"Send audio data error: {e}")

def parse_info_data(info_str):
    """Parse and print detailed information"""
    try:
        info_data = json.loads(info_str)
        print(f"📋 Detailed info:")
        for key, value in info_data.items():
            if key == 'avg_logprob':
                confidence = round((value + 1) * 100, 1)
                print(f"   📊 Confidence: {confidence}% (logprob: {value:.3f})")
            elif key == 'emotion':
                print(f"   😊 Emotion label: {value}")
            elif key == 'language':
                print(f"   🌍 Language label: {value}")
            elif key == 'speaker':
                print(f"   👤 Speaker: {value}")
            elif key == 'timestamp':
                print(f"   ⏰ Timestamp: {value}")
            else:
                print(f"   {key}: {value}")
    except json.JSONDecodeError:
        print(f"   ⚠️  Info field is not valid JSON: {info_str}")

def calculate_detection_latency(audio_streamer, detection_time):
    """Calculate approximate latency from recent audio activity to speech detection"""
    if audio_streamer.send_times:
        # This is an approximation since we can't precisely correlate 
        # which audio chunks triggered this detection
        recent_send_time = max(audio_streamer.send_times.values())
        detection_latency = (detection_time - recent_send_time) * 1000
        
        if detection_latency > 0:
            audio_streamer.detection_latencies.append(detection_latency)
            avg_detection_latency = sum(audio_streamer.detection_latencies[-10:]) / len(audio_streamer.detection_latencies[-10:])
            print(f"   🎯 Detection response time: {detection_latency:.1f}ms | Average: {avg_detection_latency:.1f}ms")
        else:
            print(f"   ⚠️  Detection timing anomaly (negative latency: {detection_latency:.1f}ms)")

def calculate_latency(audio_streamer, receive_time, latencies):
    """Calculate and display latency from last speech detection to transcription result"""
    if audio_streamer.last_speech_detection_time:
        # Calculate latency from the last speech detection time
        latency = (receive_time - audio_streamer.last_speech_detection_time) * 1000
        
        # Only add positive latencies (avoid negative values due to timing issues)
        if latency > 0:
            latencies.append(latency)
            avg_latency = sum(latencies[-10:]) / len(latencies[-10:])
            print(f"   ⚡ Transcription latency: {latency:.1f}ms | Average: {avg_latency:.1f}ms")
            print(f"   📏 (from last speech detection to transcription result)")
        
        # Clean up old send times to prevent memory growth
        if len(audio_streamer.send_times) > 100:
            # Keep only the most recent 50 entries
            sorted_items = sorted(audio_streamer.send_times.items())
            audio_streamer.send_times = dict(sorted_items[-50:])

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
        # Transcription result - this is where we calculate the real latency
        print(f"\n🎤 Transcribed text: {data}")
        if info_str:
            parse_info_data(info_str)
            calculate_latency(audio_streamer, receive_time, latencies)
    elif code == 2:
        # System information - track speech detection
        if 'detect speech' in info_str:
            print("🔊 Speech detection started...")
            # Calculate detection latency
            calculate_detection_latency(audio_streamer, receive_time)
            # Update the last speech detection time
            audio_streamer.last_speech_detection_time = receive_time
        elif 'detect speaker' in info_str:
            print(f"👤 Speaker detected: {data}")
        else:
            print(f"ℹ️  System info: {info_str}")
            if data:
                print(f"   Data: {data}")
    else:
        print(f"❓ Unknown response type")

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
                    print(f"⚠️  Received non-JSON response: {response}")
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                break
            except Exception as e:
                print(f"Receive response error: {e}")
                break
    except Exception as e:
        print(f"Response receiving task error: {e}")
    
    # Display latency statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n📊 Transcription Latency Statistics (total {len(latencies)} times):")
        print(f"   Average latency: {avg_latency:.1f}ms")
        print(f"   Minimum latency: {min(latencies):.1f}ms") 
        print(f"   Maximum latency: {max(latencies):.1f}ms")
    
    if audio_streamer.detection_latencies:
        avg_detection = sum(audio_streamer.detection_latencies) / len(audio_streamer.detection_latencies)
        print(f"\n🎯 Detection Response Time Statistics (total {len(audio_streamer.detection_latencies)} times):")
        print(f"   Average response time: {avg_detection:.1f}ms")
        print(f"   Minimum response time: {min(audio_streamer.detection_latencies):.1f}ms")
        print(f"   Maximum response time: {max(audio_streamer.detection_latencies):.1f}ms")
        print(f"   Note: These are approximate values due to audio-detection correlation limitations")

async def connect_to_server(url, block_ms):
    audio_streamer = AudioStreamer(block_ms)

    print(f"Connecting to server: {url}")
    
    async with websockets.connect(url) as websocket:
        print("WebSocket connection successful")
        
        tasks = [
            asyncio.create_task(send_audio_data(websocket, audio_streamer)),
            asyncio.create_task(receive_responses(websocket, audio_streamer))
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nUser interrupted")
            for task in tasks:
                task.cancel()
            # Wait for tasks to be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise to be caught by main()

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