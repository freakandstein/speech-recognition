import collections
import numpy as np
import torch
import whisper
import pyaudio
from silero_vad import load_silero_vad


# ── Silero VAD chunk size harus 512 samples pada 16kHz ──────────────────────
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms per chunk, required by Silero VAD


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


def get_audio_rms(chunk_bytes):
    """Hitung RMS dari raw audio bytes."""
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(audio ** 2))


def bytes_to_tensor(chunk_bytes):
    """Konversi raw bytes ke float32 tensor untuk Silero VAD."""
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def record_with_silero(silero_model, sample_rate=SAMPLE_RATE):
    """
    Rekam audio menggunakan Silero VAD (neural network) untuk deteksi
    suara manusia yang akurat.

    Silero mengembalikan confidence score 0.0–1.0:
      > 0.5  → kemungkinan besar suara manusia
      < 0.35 → bukan suara manusia / silence
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    print("\n👂 Mendengarkan... (berbicara untuk mulai merekam)")
    print("=" * 50)

    # Ring buffer: simpan ~20 chunk terakhir (~640ms)
    ring_buffer = collections.deque(maxlen=20)
    triggered = False
    voiced_frames = []
    silent_chunks = 0

    # 1200ms / 32ms per chunk ≈ 37 chunks diam sebelum berhenti rekam
    max_silent_chunks = 37

    SPEECH_THRESHOLD = 0.5    # di atas ini → suara manusia
    SILENCE_THRESHOLD = 0.35  # di bawah ini → silence / non-human

    # Flush stream dulu
    for _ in range(10):
        stream.read(CHUNK_SAMPLES, exception_on_overflow=False)

    # Kalibrasi noise floor untuk energy gate
    print("🔧 Kalibrasi noise floor... (diam sebentar)")
    noise_rms_samples = []
    for _ in range(50):
        cal_chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
        noise_rms_samples.append(get_audio_rms(cal_chunk))
    # Buang outlier dulu: hapus sampel yang > median × 3
    # (jika ada suara keras saat kalibrasi, sampelnya dibuang sebelum hitung threshold)
    arr = np.array(noise_rms_samples)
    median = np.median(arr)
    clean = arr[arr <= median * 3]  # hanya pakai sampel yang wajar
    if len(clean) < 5:
        clean = arr  # fallback jika terlalu banyak yang dibuang
    noise_floor = np.percentile(clean, 75)  # P75 dari sampel bersih
    energy_threshold = max(200, noise_floor * 1.5)
    print(f"📊 Noise median: {median:.1f} | Clean P75: {noise_floor:.1f} | Energy gate: {energy_threshold:.1f}")
    print(f"   ({len(arr) - len(clean)} sampel outlier dibuang dari {len(arr)} total)\n")

    try:
        while True:
            chunk_bytes = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)

            if len(chunk_bytes) != CHUNK_SAMPLES * 2:
                continue

            # ── Silero VAD: confidence score 0.0–1.0 ────────────────────────
            with torch.no_grad():
                tensor = bytes_to_tensor(chunk_bytes)
                confidence = silero_model(tensor, sample_rate).item()

            # ── Energy gate ──────────────────────────────────────────────────
            chunk_rms = get_audio_rms(chunk_bytes)
            energy_ok = chunk_rms > energy_threshold

            # Valid speech = Silero yakin DAN energy cukup
            is_speech = (confidence > SPEECH_THRESHOLD) and energy_ok

            # ── Debug real-time ──────────────────────────────────────────────
            status = "🗣 " if is_speech else "   "
            print(
                f"  {status} Conf: {confidence:.2f} | RMS: {chunk_rms:6.1f} | "
                f"{'RECORDING' if triggered else f'buf: {sum(1 for _,v in ring_buffer if v)}/{ring_buffer.maxlen}'}",
                end="\r"
            )

            if not triggered:
                ring_buffer.append((chunk_bytes, is_speech))
                num_voiced = sum(1 for _, v in ring_buffer if v)

                # Trigger jika 75% dari ring buffer adalah speech
                if num_voiced >= 0.75 * ring_buffer.maxlen:
                    triggered = True
                    print(f"\n🎤 Suara manusia terdeteksi! (conf: {confidence:.2f}) Merekam...")
                    voiced_frames.extend([c for c, _ in ring_buffer])
                    ring_buffer.clear()
                    silent_chunks = 0
            else:
                voiced_frames.append(chunk_bytes)

                is_silent = (confidence < SILENCE_THRESHOLD) or not energy_ok
                if is_silent:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > max_silent_chunks:
                    print("\n🔇 Selesai berbicara, memproses...")
                    break

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return b"".join(voiced_frames)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Load Silero VAD — neural network, jauh lebih akurat dari WebRTC VAD
    print("📦 Loading Silero VAD...")
    silero_model = load_silero_vad()
    silero_model.eval()
    print("✅ Silero VAD loaded\n")

    # Load Whisper
    whisper_model = load_whisper_model("small", device)

    print("🔁 Tekan Ctrl+C untuk berhenti\n")
    try:
        while True:
            audio_bytes = record_with_silero(silero_model)

            duration_s = len(audio_bytes) / (SAMPLE_RATE * 2)
            overall_rms = np.sqrt(
                np.mean(np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) ** 2)
            )

            # Filter: terlalu pendek
            if duration_s < 0.5:
                print("⏭️  SKIP: terlalu pendek\n")
                continue

            # Filter: energi terlalu rendah
            if overall_rms < 300:
                print("⏭️  SKIP: energi terlalu rendah\n")
                continue

            print("🔍 Memproses...")
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            result = whisper_model.transcribe(
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
            if text and text.replace(".", "").replace(",", "").replace(" ", ""):
                print(f"📝 {text}")
            else:
                print("📝 (tidak terdeteksi ucapan)")
            print("=" * 50)
            print()

    except KeyboardInterrupt:
        print("\n\n👋 Selesai!")


if __name__ == "__main__":
    main()