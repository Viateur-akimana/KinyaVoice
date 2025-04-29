import os
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login, HfApi
import torchaudio
from dotenv import load_dotenv
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KinyarwandaTranscriber:
    def __init__(self):
        try:
            # Get token from environment variable
            token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not token:
                logger.warning("HUGGINGFACE_HUB_TOKEN not found in environment variables")
                logger.info("Attempting to use cached credentials instead")
            else:
                logger.info("Authenticating with Hugging Face using token")
                login(token=token)
            
            # Initialize HfApi - will use cached token if available
            self.api = HfApi()
            
            logger.info("Loading KinyaWhisper model and processor...")
            # Load the fine-tuned KinyaWhisper model and processor
            self.model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
            self.processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KinyarwandaTranscriber: {str(e)}")
            raise
    
    def convert_to_wav(self, audio_path):
        """Convert MP3 file to WAV if needed."""
        # Check if the file is already a WAV file
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        # Create a temporary WAV file
        wav_path = os.path.splitext(audio_path)[0] + '.wav'
        
        try:
            logger.info(f"Converting {audio_path} to WAV format at {wav_path}")
            
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                # Use ffmpeg to convert
                subprocess.run([
                    'ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', 
                    '-c:a', 'pcm_s16le', wav_path
                ], check=True, capture_output=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fallback to torchaudio if ffmpeg is not available
                logger.info("ffmpeg not available, using torchaudio for conversion")
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    )(waveform)
                torchaudio.save(wav_path, waveform, 16000)
            
            logger.info(f"Conversion completed successfully")
            return wav_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            # Return the original file if conversion fails
            return audio_path
    
    def transcribe(self, audio_path):
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Convert MP3 to WAV if needed
            audio_path = self.convert_to_wav(audio_path)
            
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.info(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
            
            # Ensure the sample rate is 16 kHz
            if sample_rate != 16000:
                logger.info(f"Resampling audio to 16 kHz")
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            
            # Ensure the waveform is mono
            if waveform.shape[0] > 1:
                logger.info("Converting stereo audio to mono")
                waveform = waveform.mean(dim=0, keepdim=True)
            
            inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
            
            # Generate transcription
            predicted_ids = self.model.generate(inputs["input_features"])
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            logger.info("Transcription completed successfully")
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise