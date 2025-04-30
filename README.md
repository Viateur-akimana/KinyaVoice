# Kinyarwanda Voice Assistant

A mini voice assistant for Kinyarwanda language, built for the Intelligent Robotics course (April 2025). This project implements a complete voice interaction pipeline with Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Text-to-Speech (TTS) capabilities.

## 🚀 Features

- **🗣️ Speech Recognition**: Transcribes Kinyarwanda speech to text using `benax-rw/KinyaWhisper` (ASR)
- **🤖 NLP Processing**: Understands queries using intent-based pattern matching with fuzzy matching capabilities
- **🔊 Speech Synthesis**: Generates natural-sounding Kinyarwanda speech using custom TTS models
- **🌐 Web Interface**: User-friendly Gradio interface for interactive conversations
- **💻 CLI Support**: Command-line interface for batch processing of audio files

## 📂 Project Structure


```
kinyarwanda-voice-assistant/
├── gradio_app.py                  # Main Gradio web interface
├── main.py                 # Command-line application
├── asr/
│   └── transcribe.py       # Speech recognition module
├── nlp/
│   ├── qa_engine.py        # Question answering system
│   └── intents.json        # Intent patterns and responses
├── tts/
│   └── speech_synth.py     # Text-to-speech engine
├── Inference/
│   └── kinyatts/           # TTS model and utilities
├── test_audio/             # Sample audio files
└── transcription_output/                # Generated audio outputs
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Viateur-akimana/KinyaVoice.git
   cd kinyarwanda-voice-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the TTS model:
   - Download the TTS model file and place it at the correct location
   - Verify the paths in `tts/speech_synth.py`

## Usage


You can use the system from the command line:

```bash
python main.py
```

This will:
1. Transcribe audio from a predefined file
2. Process the transcription and generate a response
3. Convert the response to speech

## Components

### Speech Recognition (ASR)

Uses the KinyaWhisper model for accurate Kinyarwanda speech recognition.

### Natural Language Processing (NLP)

Pattern matching with fuzzy matching capabilities to understand user input and generate appropriate responses.

### Text-to-Speech (TTS)

Uses a custom KinyaTTS model trained specifically for Kinyarwanda to generate kinyarwanda-sounding speech.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers
- Gradio
- Torchaudio
- Additional dependencies in requirements.txt

## ⚠️ Limitations

### Speech Recognition (ASR)
- 🎙️ Requires clear audio input (16kHz mono recommended)
- 🌫️ Background noise may reduce transcription accuracy
- 🔠 Struggles with rare Kinyarwanda dialects/accents

### Natural Language Processing
- 📜 Currently limited to predefined intents (expandable in `intents.json`)
- 🔄 No contextual conversation memory
- ❓ Handles only direct questions (no follow-up questions)

### Speech Synthesis (TTS)
- 🔊 Pronunciation may be imperfect for some Kinyarwanda words
- 🐢 Slight latency in voice generation (~1-2 seconds)
- 🎚️ Limited voice customization options

## 🔮 Future Improvements

### Core Technology
- ⚡ Real-time conversation capabilities
- 🎯 Fine-tuned TTS specifically for Kinyarwanda pronunciation
- 🤖 Integration with robotic platforms (ROS, Arduino)

### Language Capabilities
- 🌍 Multilingual support (Kinyarwanda-English code-switching)
- 📚 Expanded intent database (50+ Kinyarwanda phrases)
- 💬 Advanced NLP for follow-up questions

### User Experience
- 🎨 Customizable voice characteristics

### Performance
- 🚀 GPU acceleration for faster inference
- 🧠 Machine learning model optimizations
- 📊 Detailed performance analytics

## Credits

- KinyaWhisper ASR model by [benax-rw](https://huggingface.co/benax-rw/KinyaWhisper)
- KinyaTTS for text-to-speech capabilities