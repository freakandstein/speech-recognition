import sys
import io
import collections
import numpy as np
import torch
import whisper
import pyaudio
import webrtcvad


def load_whisper_model(model_name, device):
    """Load Whisper model, handle sparse tensor untuk MPS."""
    print(f"📦 Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, device="cpu")

    if device == "mps":
        for name, buf in list(model.named_buffers()):
            if buf.is_sparse:
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent.register_buffer(parts[-1], buf.to_dense())
        model = model.to(device)

    print(f"✅ Model loaded on {device}\n")
    return model


def is_loud_enough(chunk, threshold=750):
    """Cek apakah audio cukup keras berdasarkan RMS energy."""
    audio = np.frombuffer(chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms > threshold


def record_with_vad(vad, sample_rate=16000):
    """
    Rekam audio hanya saat ada suara menggunakan VAD + energy detection.
    """
    chunk_duration_ms = 30
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunk_size = chunk_samples * 2

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_samples,
    )

    print("\n👂 Mendengarkan... (berbicara untuk mulai merekam)")
    print("=" * 50)

    ring_buffer = collections.deque(maxlen=20)
    triggered = False
    voiced_frames = []
    silent_chunks = 0
    max_silent_chunks = 40

    # Flush stream dulu sebelum mulai mendengarkan
    for _ in range(10):
        stream.read(chunk_samples, exception_on_overflow=False)

    try:
        while True:
            chunk = stream.read(chunk_samples, exception_on_overflow=False)

            if len(chunk) != chunk_size:
                continue

            try:
                has_voice = vad.is_speech(chunk, sample_rate)
            except Exception:
                has_voice = False

            if not triggered:
                ring_buffer.append((chunk, has_voice))
                num_voiced = len([f for f, v in ring_buffer if v])

                if num_voiced > 0.8 * ring_buffer.maxlen:
                    triggered = True
                    print("🎤 Suara terdeteksi! Merekam...")
                    voiced_frames.extend([f for f, _ in ring_buffer])
                    ring_buffer.clear()
                    silent_chunks = 0
            else:
                voiced_frames.append(chunk)

                if not has_voice:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > max_silent_chunks:
                    print("🔇 Selesai berbicara, memproses...")
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return b"".join(voiced_frames)


def main():
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Load model
    model = load_whisper_model("small", device)

    # Setup VAD mode 2 = balance antara sensitif dan ketat
    vad = webrtcvad.Vad()
    vad.set_mode(3)

    print("🔁 Tekan Ctrl+C untuk berhenti\n")
    try:
        while True:
            audio_bytes = record_with_vad(vad)

            if len(audio_bytes) < 1000:
                continue

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            print("🔍 Memproses audio...")
            result = model.transcribe(
                audio_np,
                language="en",
                fp16=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
            )
            text = result["text"].strip()

            print("=" * 50)
            if text and text.replace(".", "").replace(" ", "") != "":
                print(f"📝 Hasil: {text}")
            else:
                print("📝 Hasil: (tidak terdeteksi ucapan)")
            print("=" * 50)
            print()

    except KeyboardInterrupt:
        print("\n\n👋 Selesai!")


if __name__ == "__main__":
    main()