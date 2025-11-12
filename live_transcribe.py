"""
live_transcribe.py 
Real-time call transcription with  error handling
"""

import os
import soundfile as sf
import queue
import sys
import time
import whisper
import sounddevice as sd
import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path

# Audio settings
SAMPLE_RATE = 16000
BLOCK_DURATION = 15
MAX_QUEUE_SIZE = 100

# Output file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"live_transcript_{timestamp}.txt"

# Initialize
print("="*60)
print("🎙️  LIVE CALL TRANSCRIPTION SYSTEM")
print("="*60)
print(f"\n📝 Output: {output_file}")
print("🔄 Loading Whisper model (small for speed)...")

try:
    model = whisper.load_model("small")
    print("✅ Model loaded successfully\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Audio queue with size limit
q = queue.Queue(maxsize=MAX_QUEUE_SIZE)
chunk_count = 0
error_count = 0

def callback(indata, frames, time_info, status):
    """Audio callback - runs in separate thread"""
    if status:
        print(f"⚠️ Audio status: {status}", file=sys.stderr)
    try:
        q.put(indata.copy(), block=False)
    except queue.Full:
        print("⚠️ Buffer full, dropping frame", file=sys.stderr)

# Write header to transcript file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Live Transcription Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")

print("🎙️  Recording... (Ctrl+C to stop)\n")
print("💡 Tip: Speak clearly, wait for transcription between chunks\n")
print("-"*60 + "\n")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=1, 
        dtype="float32", 
        callback=callback
    ):
        buffer = []
        start_time = time.time()
        
        while True:
            # Collect audio chunks
            try:
                buffer.append(q.get(timeout=1))
            except queue.Empty:
                continue
            
            duration = time.time() - start_time
            
            # Process every BLOCK_DURATION seconds
            if duration >= BLOCK_DURATION:
                chunk_count += 1
                audio_block = np.concatenate(buffer, axis=0).flatten()
                buffer = []
                start_time = time.time()
                
                print(f"🧠 Transcribing chunk #{chunk_count}...", end="", flush=True)
                
                try:
                    # Transcribe directly from numpy array (no file I/O!)
                    result = model.transcribe(
                        audio_block, 
                        language="en",
                        fp16=False
                    )
                    
                    text = result["text"].strip()
                    
                    if not text:
                        print(" [No speech detected]")
                        continue
                    
                    # Calculate confidence (optional)
                    if result.get('segments'):
                        confidences = [seg.get('avg_logprob', 0) 
                                      for seg in result['segments']]
                        avg_confidence = np.mean(confidences) if confidences else 0
                    else:
                        avg_confidence = 0
                    
                    print(" ✓")
                    print(f"📝 {text}\n")
                    
                    # Save with metadata
                    timestamp_str = datetime.now().strftime('%H:%M:%S')
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f"[Chunk {chunk_count} | {timestamp_str}]\n")
                        f.write(f"{text}\n\n")
                    
                    print(f"💾 Saved (Total chunks: {chunk_count})\n")
                    print("-"*60 + "\n")
                    
                except Exception as e:
                    error_count += 1
                    print(f" ❌ Error")
                    print(f"⚠️ Transcription failed: {e}")
                    print("⏭️  Continuing...\n")
                    
                    # Log error
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f"[Chunk {chunk_count} | {datetime.now().strftime('%H:%M:%S')} | ERROR]\n")
                        f.write(f"Transcription failed: {str(e)}\n\n")

except KeyboardInterrupt:
    print("\n" + "="*60)
    print("🛑 Stopped by user")
    print("="*60)
    
    # Write summary footer
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Chunks: {chunk_count}\n")
        f.write(f"Errors: {error_count}\n")
    
    print(f"\n📊 Statistics:")
    print(f"   Chunks processed: {chunk_count}")
    print(f"   Errors: {error_count}")
    print(f"   Success rate: {((chunk_count-error_count)/chunk_count*100):.1f}%" if chunk_count > 0 else "   Success rate: N/A")
    print(f"\n💾 Transcript saved: {output_file}")
    print()

except Exception as e:
    print(f"\n❌ Fatal error: {e}")
    sys.exit(1)
