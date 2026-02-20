# Speech Recognition CLI with Whisper & Silero VAD

> **Note:** This script was tested and runs well on a MacBook M4 Max using Whisper and Silero VAD. Performance is excellent on Apple Silicon (M4/M3/M2/M1) for real-time speech recognition and transcription.

This script records audio from your microphone, automatically detects segments containing human speech using Silero VAD (neural network), and transcribes them using OpenAI Whisper.

## Main Features
- **Input device selection**: Choose any microphone (iPhone, Audio Interface, etc) or use the default MacBook mic.
- **Human speech detection**: Uses Silero VAD for highly accurate voice activity detection.
- **Auto energy gate**: Calibrates automatically to ignore room noise.
- **Automatic transcription**: Uses Whisper (model `small`).
- **Realtime feedback**: Shows confidence, RMS, and recording status in the terminal.

## Installation

1. **Install Python 3.8+**
2. Install dependencies:
    ```bash
    pip3 install torch numpy pyaudio whisper silero-vad
    ```
    - If you get a pyaudio error on Mac, install portaudio first:
       ```bash
       brew install portaudio
       pip3 install pyaudio
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

## Voice Command Mapping

You can map recognized phrases to keyboard shortcuts using the `VOICE_COMMANDS` dictionary in the script. Example (from the current script):

```
VOICE_COMMANDS = {
    "first camera": ["command", "3"],
    "second camera": ["command", "4"],
    "record": ["command", "r"],
    # Add more mappings here
}
```

When you say "First camera", the script will send Command+3 using `pyautogui` (make sure Python has Accessibility permissions in System Settings). "Second camera" will send Command+4, and "Record" will send Command+R.

You can add more commands by editing the dictionary. The matching is case-insensitive, ignores punctuation, and supports fuzzy matching for minor typos.

**Note:** Some system shortcuts may not work in all contexts due to macOS security restrictions.
