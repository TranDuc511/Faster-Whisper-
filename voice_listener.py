"""
Faster-Whisper Voice Listener for Real-time Speech-to-Text
Uses GPU acceleration (CUDA) if available, with Voice Activity Detection.
Supports both English and Vietnamese.
"""

import threading
import numpy as np
import sounddevice as sd
from typing import Callable, Optional
from collections import deque
import time

# Lazy load to avoid slow startup
_whisper_model = None
_model_lock = threading.Lock()

def get_whisper_model(model_size: str = "small", device: str = "cuda"):
    """Lazy load Faster-Whisper model (singleton pattern)"""
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            print(f"[Whisper] Loading model '{model_size}' on {device}...")
            _whisper_model = WhisperModel(
                model_size, 
                device=device,
                compute_type="float16" if device == "cuda" else "int8"
            )
            print(f"[Whisper] Model loaded successfully!")
        return _whisper_model


class FasterWhisperListener:
    """
    Real-time voice listener using Faster-Whisper.
    Records audio in chunks, detects speech, and transcribes.
    """

    def __init__(
        self,
        on_speech_recognized: Callable[[str], None],
        on_status_change: Optional[Callable[[str], None]] = None,
        on_partial_result: Optional[Callable[[str], None]] = None,
        model_size: str = "small",
        device: str = "cuda",
        language: Optional[str] = None  # None = auto-detect (supports EN + VI)
    ):
        """
        Args:
            on_speech_recognized: Callback when final text is recognized
            on_status_change: Optional callback for status updates
            on_partial_result: Optional callback for partial/interim results
            model_size: Whisper model size (tiny/base/small/medium/large)
            device: "cuda" for GPU or "cpu"
            language: Language code or None for auto-detection
        """
        self.on_speech_recognized = on_speech_recognized
        self.on_status_change = on_status_change
        self.on_partial_result = on_partial_result
        self.model_size = model_size
        self.device = device
        self.language = language  # None = auto (multi-lingual)

        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_duration = 0.5  # seconds per chunk
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # VAD settings (simple energy-based)
        self.silence_threshold = 0.01  # RMS threshold for silence
        self.speech_timeout = 1.5  # seconds of silence to end utterance
        self.min_speech_duration = 0.3  # minimum seconds to consider valid speech
        
        # State
        self._running = False
        self._recording_thread = None
        self._audio_buffer = deque(maxlen=int(30 / self.chunk_duration))  # Max 30 seconds
        self._speech_chunks = []
        self._last_speech_time = 0
        self._is_speaking = False

    def _update_status(self, msg: str):
        if self.on_status_change:
            self.on_status_change(msg)

    def start(self):
        """Start listening in background"""
        if self._running:
            return

        self._running = True
        self._update_status("Loading Whisper model...")
        
        # Pre-load model in background
        def init_and_start():
            try:
                get_whisper_model(self.model_size, self.device)
                self._update_status("Listening...")
                self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
                self._recording_thread.start()
            except Exception as e:
                self._update_status(f"Error: {e}")
                self._running = False
        
        threading.Thread(target=init_and_start, daemon=True).start()

    def stop(self):
        """Stop listening"""
        self._running = False
        self._update_status("Mic Stopped")

    def _recording_loop(self):
        """Main recording loop with VAD"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size
            ) as stream:
                while self._running:
                    audio_chunk, _ = stream.read(self.chunk_size)
                    audio_chunk = audio_chunk.flatten()
                    
                    # Simple energy-based VAD
                    rms = np.sqrt(np.mean(audio_chunk ** 2))
                    is_speech = rms > self.silence_threshold
                    current_time = time.time()
                    
                    if is_speech:
                        self._last_speech_time = current_time
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_chunks = []
                            self._update_status("🎤 Speaking...")
                        
                        self._speech_chunks.append(audio_chunk)
                    
                    elif self._is_speaking:
                        # Still collecting after speech ends (for trailing audio)
                        self._speech_chunks.append(audio_chunk)
                        
                        # Check if silence timeout reached
                        if current_time - self._last_speech_time > self.speech_timeout:
                            self._is_speaking = False
                            self._process_speech()
                            self._update_status("Listening...")

        except Exception as e:
            self._update_status(f"Recording error: {e}")

    def _process_speech(self):
        """Process collected speech chunks"""
        if not self._speech_chunks:
            return
        
        # Combine chunks
        audio = np.concatenate(self._speech_chunks)
        duration = len(audio) / self.sample_rate
        
        # Skip if too short
        if duration < self.min_speech_duration:
            self._speech_chunks = []
            return
        
        self._update_status("Processing...")
        
        # Transcribe in background
        def transcribe():
            try:
                model = get_whisper_model(self.model_size, self.device)
                
                segments, info = model.transcribe(
                    audio,
                    beam_size=5,
                    language=self.language,  # None = auto-detect
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
                
                # Combine all segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                full_text = " ".join(text_parts).strip()
                
                if full_text:
                    detected_lang = info.language if hasattr(info, 'language') else "unknown"
                    print(f"[Whisper] Detected: '{full_text}' (lang={detected_lang})")
                    self.on_speech_recognized(full_text)
                    
            except Exception as e:
                print(f"[Whisper] Transcription error: {e}")
            finally:
                self._update_status("Listening...")
        
        threading.Thread(target=transcribe, daemon=True).start()
        self._speech_chunks = []


# Backward compatibility alias
VoiceListener = FasterWhisperListener
