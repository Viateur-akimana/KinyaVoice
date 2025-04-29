# Kinyarwanda Voice Assistant

An interactive voice assistant for Kinyarwanda language that can transcribe speech, respond to queries, and generate spoken responses.

## Features

- **Speech Recognition**: Transcribes Kinyarwanda speech to text using KinyaWhisper
- **Natural Language Processing**: Pattern-matching and intent-based response system
- **Text-to-Speech**: Generates natural-sounding Kinyarwanda speech from text
- **Web Interface**: Easy-to-use Gradio interface for interaction

## Project Structure

```
kinyarwanda-voice-assistant/
├── app.py                  # Main Gradio web interface
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
└── outputs/                # Generated audio outputs
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kinyarwanda-voice-assistant.git
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

### Web Interface

Run the Gradio web interface:

```bash
python app.py
```

This will launch a web interface where you can:
- Speak into your microphone in Kinyarwanda
- See the transcription of your speech
- Read the assistant's response
- Hear the spoken response

### Command-line Interface

You can also use the system from the command line:

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

Uses a custom TTS model trained specifically for Kinyarwanda to generate natural-sounding speech.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers
- Gradio
- Torchaudio
- Additional dependencies in requirements.txt

## License

[Specify your license here]

## Credits

- KinyaWhisper ASR model by [benax-rw](https://huggingface.co/benax-rw/KinyaWhisper)
- [Add other credits as appropriate]