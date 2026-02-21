import collections
import numpy as np
import torch
import whisper
import pyaudio
import pyautogui

import re
from difflib import get_close_matches
from silero_vad import load_silero_vad


# ── Silero VAD chunk size harus 512 samples pada 16kHz ──────────────────────
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms per chunk, required by Silero VAD
VOICE_COMMANDS = {
    "kamera satu": ["command", "1"],
    "kamera dua": ["command", "2"],
    "kamera tiga": ["command", "3"],
    "kamera empat": ["command", "4"],
    "merekam": ["command", "r"],
    # Add more mappings here
}

DIGIT_TO_WORD = {
    "0": "nol", "1": "satu", "2": "dua", "3": "tiga", "4": "empat",
    "5": "lima", "6": "enam", "7": "tujuh", "8": "delapan", "9": "sembilan",
}

def normalize_text(text):
    # Lowercase, remove punctuation, collapse whitespace
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]+', '', text)
    # Konversi digit ke kata (misal: "kamera 1" → "kamera satu")
    text = re.sub(r'\b(\d)\b', lambda m: DIGIT_TO_WORD.get(m.group(1), m.group(1)), text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_shortcut(keys):
    pyautogui.hotkey(*keys)

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


def list_input_devices():
    """Tampilkan semua input device yang tersedia."""
    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append((i, info["name"], int(info["maxInputChannels"])))
    pa.terminate()
    return devices


def select_input_device():
    """Interaktif: tampilkan daftar device dan minta user pilih."""
    devices = list_input_devices()
    print("\n🎙️  Input devices tersedia:")
    for idx, name, ch in devices:
        print(f"  [{idx}] {name} (ch: {ch})")

    default_idx = None
    for idx, name, ch in devices:
        if "macbook" in name.lower() or "built-in" in name.lower():
            default_idx = idx
            break
    if default_idx is None:
        default_idx = devices[0][0]

    try:
        choice = input(f"\nPilih device index (Enter = [{default_idx}] default): ").strip()
        selected = int(choice) if choice else default_idx
        # Validasi
        valid_ids = [d[0] for d in devices]
        if selected not in valid_ids:
            print(f"⚠️  Index tidak valid, pakai default [{default_idx}]")
            selected = default_idx
    except ValueError:
        selected = default_idx

    chosen = next((d for d in devices if d[0] == selected), None)
    print(f"✅ Menggunakan: [{chosen[0]}] {chosen[1]}\n")
    return selected, chosen[2]  # return (device_index, channels)


def get_audio_rms(chunk_bytes):
    """Hitung RMS dari raw audio bytes."""
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(audio ** 2))


def bytes_to_tensor(chunk_bytes):
    """Konversi raw bytes ke float32 tensor untuk Silero VAD."""
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def calibrate_energy_gate(stream, sample_rate=SAMPLE_RATE):
    """Kalibrasi awal: baca 50 chunk dari stream, return energy_threshold."""
    noise_rms_samples = []
    for _ in range(50):
        cal_chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
        noise_rms_samples.append(get_audio_rms(cal_chunk))
    return compute_energy_threshold(noise_rms_samples, label="Kalibrasi awal")


def compute_energy_threshold(rms_samples, label="Recalibrate"):
    """Hitung energy threshold dari list RMS tanpa membaca stream baru."""
    arr = np.array(rms_samples)
    median = np.median(arr)
    clean = arr[arr <= median * 3]
    if len(clean) < 5:
        clean = arr
    noise_floor = np.percentile(clean, 75)
    energy_threshold = max(200, noise_floor * 1.0)
    outliers = len(arr) - len(clean)
    print(f"\n📊 [{label}] Noise median: {median:.1f} | Clean P75: {noise_floor:.1f} | Energy gate: {energy_threshold:.1f} ({outliers} outlier dibuang)")
    return energy_threshold


def record_with_silero(silero_model, device_index=None, sample_rate=SAMPLE_RATE):
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
        input_device_index=device_index,
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

    # Kalibrasi noise floor awal
    print("🔧 Kalibrasi noise floor... (diam sebentar)")
    energy_threshold = calibrate_energy_gate(stream, sample_rate)

    # Auto-recalibrate setelah 2 detik idle tanpa speech
    # 2000ms / 32ms per chunk = 62 chunks
    IDLE_RECAL_CHUNKS = 62
    idle_chunks = 0          # counter chunk idle (tidak triggered, tidak speech)
    idle_rms_buffer = []     # kumpulkan RMS dari chunk idle untuk recalibrate

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

                # ── Auto-recalibrate saat idle terlalu lama ──────────────────
                if not is_speech:
                    idle_chunks += 1
                    idle_rms_buffer.append(chunk_rms)  # kumpulkan RMS idle
                    if idle_chunks >= IDLE_RECAL_CHUNKS:
                        # Recalibrate dari RMS yang sudah dikumpulkan — TANPA baca stream baru
                        # sehingga tidak ada audio yang terlewat
                        energy_threshold = compute_energy_threshold(idle_rms_buffer, label="Auto-recal")
                        idle_chunks = 0
                        idle_rms_buffer = []
                        ring_buffer.clear()
                else:
                    idle_chunks = 0
                    idle_rms_buffer = []

                # Trigger jika 75% dari ring buffer adalah speech
                if num_voiced >= 0.75 * ring_buffer.maxlen:
                    triggered = True
                    idle_chunks = 0
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

    # Pilih input device
    device_index, _ = select_input_device()

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
            audio_bytes = record_with_silero(silero_model, device_index=device_index)

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
                language="id",
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
                # Normalize and fuzzy match
                cmd = normalize_text(text)
                print(f"🔗 Looking for shortcut for: '{cmd}'")
                match = None
                if cmd in VOICE_COMMANDS:
                    match = cmd
                else:
                    # Fuzzy match: allow minor typo or punctuation difference
                    close = get_close_matches(cmd, VOICE_COMMANDS.keys(), n=1, cutoff=0.8)
                    if close:
                        match = close[0]
                if match:
                    print(f"🔗 Executing shortcut for: {match}")
                    run_shortcut(VOICE_COMMANDS[match])
            else:
                print("📝 (tidak terdeteksi ucapan)")
            print("=" * 50)
            print()

    except KeyboardInterrupt:
        print("\n\n👋 Selesai!")


if __name__ == "__main__":
    main()