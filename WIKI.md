# Technical Wiki: Speech Recognition CLI

## Overview

A Python CLI for real-time voice command recognition via microphone. The pipeline is:

```
Microphone → PyAudio stream → Silero VAD → Whisper ASR → Fuzzy match → pyautogui shortcut
```

Designed to run on macOS with Apple Silicon (M1/M2/M3/M4), leveraging MPS (Metal Performance Shaders) as a hardware accelerator.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                         main()                           │
│                                                          │
│  1. Device selection (select_input_device)               │
│  2. Load Silero VAD                                      │
│  3. Load Whisper model (load_whisper_model)              │
│                                                          │
│  ┌─────── Main loop ──────────────────────────────────┐  │
│  │                                                    │  │
│  │  record_with_silero()  ←── PyAudio stream          │  │
│  │       │                                            │  │
│  │       ▼                                            │  │
│  │  [calibrate_energy_gate]  ←── 50 chunk baseline    │  │
│  │       │                                            │  │
│  │       ▼                                            │  │
│  │  [VAD loop]  ─── Silero confidence + RMS gate      │  │
│  │       │                                            │  │
│  │       ▼                                            │  │
│  │  Whisper transcribe (language=id)                  │  │
│  │       │                                            │  │
│  │       ▼                                            │  │
│  │  normalize_text → exact/fuzzy match VOICE_COMMANDS │  │
│  │       │                                            │  │
│  │       ▼                                            │  │
│  │  run_shortcut → pyautogui.hotkey(...)              │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` Hz | Standard rate for Silero VAD & Whisper |
| `CHUNK_SAMPLES` | `512` | ~32ms per chunk, **required** by Silero VAD |
| `SPEECH_THRESHOLD` | `0.35` | Minimum Silero confidence to classify as speech |
| `SILENCE_THRESHOLD` | `0.25` | Confidence below this is treated as silence |
| `max_silent_chunks` | `70` | ~2.2 seconds of silence before recording stops |
| `IDLE_RECAL_CHUNKS` | `62` | ~2 seconds idle before auto-recalibrating energy gate |
| Ring buffer `maxlen` | `20` | ~640ms look-back window before trigger |
| Trigger ratio | `0.60` | 60% of ring buffer must be speech to start recording |

---

## VAD Pipeline Detail

### Dual-Gate Speech Detection

Every 512-sample chunk is validated through two gates simultaneously:

```
is_speech = (silero_confidence > SPEECH_THRESHOLD) AND (chunk_rms > energy_threshold)
```

- **Gate 1 — Silero VAD**: A neural network producing a 0.0–1.0 confidence score indicating the likelihood that the audio contains human speech.
- **Gate 2 — Energy gate**: An RMS-based filter to discard low-amplitude noise that passes Silero (e.g., fan hum, electrical hiss).

Both gates must pass for a chunk to count as valid speech.

### Recording State Machine

```
         ┌─────────────────────────────────────┐
         │            IDLE / LISTENING          │
         │                                     │
         │  - Fill ring buffer (20 chunks)      │
         │  - Count num_voiced                  │
         │  - If idle > 62 chunks → recal       │
         └─────────────┬───────────────────────┘
                       │ num_voiced >= 60% of ring buffer
                       ▼
         ┌─────────────────────────────────────┐
         │             RECORDING               │
         │                                     │
         │  - Append all chunks to             │
         │    voiced_frames[]                  │
         │  - Track silent_chunks              │
         │    (Silero only, not RMS)           │
         └─────────────┬───────────────────────┘
                       │ silent_chunks > 70
                       ▼
                   [STOP → process]
```

**Important:** During `RECORDING`, the stop trigger uses only `silero_confidence < SILENCE_THRESHOLD`, **not** the energy gate. This is intentional — short inter-word gaps (breath, phrase breaks) should not cut the recording.

### Auto-Recalibrate

When there is no speech for ≥62 consecutive chunks, the energy threshold is recomputed from already-collected RMS values **without re-reading the stream**, so no audio is missed.

`compute_energy_threshold` algorithm:
1. Take the median of all RMS samples
2. Filter outliers: discard samples > 3× the median
3. Use P75 (75th percentile) of the clean samples as the noise floor
4. `energy_threshold = max(100, noise_floor × 1.0)`

---

## Whisper Transcription

Model used: `small` (244M parameters), transcribing in Indonesian (`language="id"`).

| Parameter | Value | Reason |
|---|---|---|
| `fp16` | `False` | MPS does not stably support FP16 |
| `no_speech_threshold` | `0.6` | Filters segments likely containing no speech |
| `logprob_threshold` | `-1.0` | Tolerates low-confidence transcription output |
| `compression_ratio_threshold` | `2.4` | Filters repetitive/hallucinated output |
| `condition_on_previous_text` | `False` | Prevents prior context from biasing transcription |

**Pre-filters before Whisper:**
- Recording duration < 0.5 seconds → skip
- Overall RMS < 150 → skip (energy too low)

---

## Voice Command Matching

### Text Normalization

`normalize_text()` applies three transformations:
1. Lowercase all characters
2. Strip all characters except `[a-z0-9 ]`
3. Convert single digits to Indonesian words (`"1"` → `"satu"`, etc.)

This ensures inputs like `"Kamera 1!"` and `"kamera satu"` resolve to the same string.

### Matching Strategy

```python
# 1. Exact match (O(1))
if cmd in VOICE_COMMANDS:
    match = cmd

# 2. Fuzzy match via difflib (cutoff=0.8)
else:
    close = get_close_matches(cmd, VOICE_COMMANDS.keys(), n=1, cutoff=0.8)
```

`cutoff=0.8` means sequence similarity must be ≥80% to be accepted. This handles minor Whisper transcription errors like `"kameraa satu"` vs `"kamera satu"`.

### Adding Voice Commands

Edit the `VOICE_COMMANDS` dictionary in [speech_recognition.py](speech_recognition.py#L16):

```python
VOICE_COMMANDS = {
    "kamera satu": ["command", "1"],
    "kamera dua":  ["command", "2"],
    # Add here:
    "fullscreen":  ["command", "f"],
    "mute":        ["command", "shift", "m"],
}
```

**Key:** The normalized phrase (lowercase, no punctuation, digits already converted to words).  
**Value:** List of arguments passed to `pyautogui.hotkey()`.

---

## MPS / Apple Silicon Handling

Whisper is loaded to CPU first, then moved to MPS:

```python
model = whisper.load_model(model_name, device="cpu")

if device == "mps":
    # Convert sparse tensors → dense before moving to MPS
    for name, buf in list(model.named_buffers()):
        if buf.is_sparse:
            parent.register_buffer(parts[-1], buf.to_dense())
    model = model.to(device)
```

MPS does not support sparse tensors, so all sparse buffers must be densified first. Silero VAD runs on CPU (default from `load_silero_vad()`).

---

## Audio Data Flow

```
PyAudio stream (int16, mono, 16kHz, 512 samples/chunk)
    │
    ├─→ bytes_to_tensor()   → float32 tensor [-1.0, 1.0] → Silero VAD
    │
    ├─→ get_audio_rms()     → float RMS value → energy gate
    │
    └─→ voiced_frames[]     → b"".join() → np.frombuffer() → float32 array
                                                              → Whisper
```

Key conversions:
- `np.frombuffer(bytes, dtype=np.int16)` — raw bytes to int16 array
- `/ 32768.0` — normalize to range [-1.0, 1.0] for Silero & Whisper
- `torch.from_numpy(audio)` — to tensor for Silero VAD

---

## Dependencies

| Library | Min Version | Role |
|---|---|---|
| `torch` | — | Tensor ops + MPS backend |
| `whisper` | — | OpenAI Whisper ASR |
| `pyaudio` | — | Microphone audio stream |
| `silero_vad` | — | Neural VAD |
| `pyautogui` | — | Keyboard shortcut simulation |
| `numpy` | — | Array ops (RMS, normalization) |
| `difflib` | stdlib | Fuzzy string matching |

**System requirements:**
- macOS (for MPS & pyautogui)
- Python 3.8+
- Accessibility permission for pyautogui (`System Settings → Privacy & Security → Accessibility`)
- portaudio (`brew install portaudio`) for PyAudio

---

## Tuning Guide

### Too many false triggers from noise

- Raise `SPEECH_THRESHOLD` to `0.45–0.55`
- Raise the multiplier in `compute_energy_threshold`: `noise_floor * 1.5`
- Raise the trigger ratio from `0.60` to `0.75`

### Quiet or distant voice not detected

- Lower `SPEECH_THRESHOLD` to `0.25–0.30`
- Lower the `max(100, ...)` floor in `compute_energy_threshold`
- Increase microphone gain in System Settings

### Recording cuts off mid-sentence

- Increase `max_silent_chunks` (default 70 ≈ 2.2 seconds)
- Lower `SILENCE_THRESHOLD` to `0.15`

### Whisper frequently mistranscribes

- Switch to `"medium"` or `"large"` model in `load_whisper_model`
- Use a higher-quality external microphone
- Ensure low background noise during recording
