# Kinyarwanda Voice Assistant

A mini voice assistant for Kinyarwanda language, built for the Intelligent Robotics course (April 2025).

---

## 🧠 Overview

This project implements a Kinyarwanda voice assistant capable of:

- 🗣️ Transcribing Kinyarwanda speech to text using `benax-rw/KinyaWhisper` (Automatic Speech Recognition).
- 🤖 Understanding queries using an intent-based NLP engine.
- 🔊 Responding with synthesized speech using `Coqui TTS`.

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/kinyarwanda-voice-assistant.git
cd kinyarwanda-voice-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token in `.env`:

```
HUGGINGFACE_HUB_TOKEN=your_token
```

4. Install `ffmpeg` for audio conversion:

```bash
sudo apt-get install ffmpeg
```

5. Run the assistant:

```bash
python main.py
```

## Files

* `asr/transcribe.py`: Speech-to-text using KinyaWhisper
* `nlp/qa_engine.py`: NLP for question-answer matching
* `nlp/intents.json`: QA pairs for NLP
* `tts/speech_synth.py`: Text-to-speech using Coqui TTS
* `main.py`: Main script to run the assistant
* `test_audio/`: 5 Kinyarwanda audio files (`audio1.wav` to `audio5.wav`)
* `transcriptions.txt`: Transcriptions of audio files
* `requirements.txt`: Python dependencies

## Usage

* Place Kinyarwanda audio files in `test_audio/` (WAV or MP3, 16 kHz mono preferred)
* Run `main.py` to process audio files, generate responses, and save spoken outputs
* Check `transcriptions.txt` for transcriptions and `response_*.wav` for TTS outputs

## Features

* **Automatic Speech Recognition (ASR)**: Accurately transcribes Kinyarwanda speech using a fine-tuned Whisper model
* **Natural Language Processing (NLP)**: Identifies intents from transcribed text and matches to appropriate responses
* **Text-to-Speech (TTS)**: Converts text responses to natural-sounding Kinyarwanda speech

## Limitations

* The `xtts_v2` TTS model may not fully support Kinyarwanda pronunciation due to limited training data
* Audio files must be clear and in Kinyarwanda for accurate transcription
* The NLP component relies on predefined intents and may not handle complex or out-of-scope queries

## Future Improvements

* Implement real-time audio processing for live conversations
* Expand the intent database with more Kinyarwanda phrases and responses
* Fine-tune TTS models specifically for Kinyarwanda pronunciation
* Add support for multiple languages (Kinyarwanda-English code-switching)


## License

[MIT License](LICENSE)

## Acknowledgments

* `benax-rw/KinyaWhisper` for the Kinyarwanda ASR model
* Coqui TTS for text-to-speech capabilities