import asyncio
from client_wss import AudioStreamer, receive_responses
import websockets
import argparse
import wave
import time

# Upstream data format: PCM binary
CHANNELS = 1
SAMPLERATE = 16000
DTYPE = "int16"

class AudioTester(AudioStreamer):
    def __init__(self, block_ms, wav_path):
        self.audio_queue = asyncio.Queue()

        self.send_times = []  # List of (chunk_id, send_time) tuples
        self.chunk_counter = 0
        self.last_speech_detection_time = None  # Track when speech was last detected
        self.detection_latencies = []  # Track audio-to-detection latencies
        self.block_ms = block_ms

        self.all_audio_data = self._read_wav_file(wav_path)
        self.audio_index = 0
        self.block_size = int(SAMPLERATE * block_ms / 1000) * 2  # 2 bytes per sample for int16

    @staticmethod
    def _read_wav_file(file_path):
        with wave.open(file_path, 'rb') as wf:
            assert wf.getnchannels() == CHANNELS
            assert wf.getframerate() == SAMPLERATE
            assert wf.getsampwidth() == 2  # 16-bit PCM
            all_audio_data = wf.readframes(wf.getnframes())
        return all_audio_data

    async def streaming(self, websocket):

        for i in range(0, len(self.all_audio_data), self.block_size):
            audio_data = self.all_audio_data[i:min(i + self.block_size, len(self.all_audio_data))]
            send_time = time.time()
            chunk_id = self.chunk_counter
            self.send_times.append((chunk_id, send_time))
            self.chunk_counter += 1
            await websocket.send(audio_data)
            await asyncio.sleep(self.block_ms / 1000)

async def connect_to_server(url, block_ms, wav_path):
    audio_streamer = AudioTester(block_ms, wav_path)

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
    parser.add_argument('--wav-path', type=str, required=True,
                       help='Path to the input WAV file')
    args = parser.parse_args()
    
    print("🎤 Real-time Speech Recognition Client")
    print("="*50, "Press Ctrl+C to stop", "="*50, sep='\n')
    try:
        url = f"ws://{args.host}:{args.port}/ws/transcribe"
        asyncio.run(connect_to_server(url, args.block_ms, args.wav_path))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")