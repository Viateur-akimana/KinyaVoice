from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from pathlib import Path
import os

AUDIO_DIR = "test_audio"
AUDIO_FILE = "kinyarwanda.mp3"
OUTPUT_DIR = "transcription_output"
MODEL_NAME = "benax-rw/KinyaWhisper"

def validate_paths():
    """
    Ensure all directories and files exist.
    
    Returns:
        Path: Path to the audio file to transcribe.
    """
    if not Path(AUDIO_DIR).exists():
        raise FileNotFoundError(f"Audio directory '{AUDIO_DIR}' not found")
    
    audio_path = Path(AUDIO_DIR) / AUDIO_FILE
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file '{audio_path}' not found")
    
    return audio_path

def load_model():
    """
    Load the KinyaWhisper model and processor.
    
    Returns:
        tuple: (model, processor) for transcription.
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        forced_decoder_ids=None,
        suppress_tokens=None
    )
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    return model, processor

def transcribe_audio(model, processor, audio_path):
    """
    Handle audio processing and transcription.
    
    Args:
        model: The WhisperForConditionalGeneration model.
        processor: The WhisperProcessor for audio processing.
        audio_path (Path): Path to the audio file.
        
    Returns:
        str: The transcribed text.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
            
        # Process with explicit language code
        inputs = processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language="rw",  # Explicitly set Kinyarwanda
            task="transcribe"  # Force transcription mode
        )
        
        # Generate transcription
        predicted_ids = model.generate(
            inputs.input_features,
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True
        )
        
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

def main():
    """
    Main function for standalone ASR usage.
    """
    audio_path = validate_paths()
    model, processor = load_model()
    transcription = transcribe_audio(model, processor, audio_path)
    
    print("Transcription:", transcription)
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"{Path(AUDIO_FILE).stem}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()