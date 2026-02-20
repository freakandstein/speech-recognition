# Speech Recognition CLI with Whisper & Silero VAD

This script records audio from your microphone, automatically detects segments containing human speech using Silero VAD (neural network), and transcribes them using OpenAI Whisper.

## Main Features
- **Input device selection**: Choose any microphone (USB, iPhone, Behringer, etc) or use the default MacBook mic.
- **Human speech detection**: Uses Silero VAD for highly accurate voice activity detection.
- **Auto energy gate**: Calibrates automatically to ignore room noise.
- **Automatic transcription**: Uses Whisper (model `small`).
- **Realtime feedback**: Shows confidence, RMS, and recording status in the terminal.

## Installation

1. **Install Python 3.8+**
2. Install dependencies:
   ```bash
   pip install torch numpy pyaudio whisper silero-vad
   ```
   - If you get a pyaudio error on Mac, install portaudio first:
     ```bash
     brew install portaudio
     pip install pyaudio
     ```

## Usage

1. Run the script:
   ```bash
   python3 speech_recognition.py
   ```
2. Select your input device from the list (USB mic, iPhone, Behringer, etc). Press Enter to use the default.
3. Stay silent during noise floor calibration.
4. Speak; the script will only record when human speech is detected.
5. Transcription results will appear in the terminal.

## Key Parameters
- **SAMPLE_RATE**: 16000 Hz (standard for VAD & Whisper)
- **CHUNK_SAMPLES**: 512 (chunk size for Silero VAD)
- **SPEECH_THRESHOLD**: 0.5 (minimum confidence to be considered human speech)
- **SILENCE_THRESHOLD**: 0.35 (below this is considered silence)
- **Energy gate**: Automatically set based on room noise during calibration (uses P75 of clean samples, robust to outliers)

## Tips
- Stay completely silent during noise floor calibration for best results.
- For more sensitivity, lower `SPEECH_THRESHOLD` or the energy gate multiplier in the script.
- For stricter detection (only loud, clear speech), increase the thresholds.
- For best results, use a high-quality external microphone.

## References
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)

---

Main script: [speech_recognition.py](speech_recognition.py)
