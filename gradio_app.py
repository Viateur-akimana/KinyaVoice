import os
import logging
import re
import gradio as gr
import tempfile
from pathlib import Path
import torch
import torchaudio
import json
import random
from difflib import get_close_matches
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from Inference.kinyatts.tts.commons import intersperse
from Inference.kinyatts.tts.models import SynthesizerTrn
from Inference.kinyatts.tts.text import symbols
from Inference.kinyatts.tts.text.kinya import text_to_sequence
from Inference.kinyatts.tts.utils import HParams, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KinyarwandaTranscriber:
    """
    ASR component using KinyaWhisper model for Kinyarwanda speech recognition.
    """
    MODEL_NAME = "benax-rw/KinyaWhisper"
    
    def __init__(self):
        logger.info("Loading KinyaWhisper model...")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.MODEL_NAME,
            forced_decoder_ids=None,
            suppress_tokens=None
        )
        self.processor = WhisperProcessor.from_pretrained(self.MODEL_NAME)
        logger.info("KinyaWhisper model loaded successfully")
    
    def transcribe(self, audio_path):
        """
        Transcribe Kinyarwanda audio to text.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.dim() > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000
                
            # Process with explicit language code
            inputs = self.processor(
                waveform.squeeze().numpy(),  # Convert to numpy array
                sampling_rate=sample_rate,
                return_tensors="pt",
                language="rw",
                task="transcribe"
            )
            
            # Generate transcription
            predicted_ids = self.model.generate(
                inputs.input_features,
                max_new_tokens=256,
                num_beams=5,
                early_stopping=True
            )
            
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

class KinyarwandaQA:
    """
    Question answering component for Kinyarwanda language.
    """
    def __init__(self, intents_file="nlp/intents.json"):
        with open(intents_file) as f:
            self.intents = json.load(f)
            
    def get_response(self, query, threshold=0.6):
        """
        Get a response for the given query by matching it with patterns in intents.
        
        Args:
            query (str): The user's query/input.
            threshold (float): Minimum similarity threshold for fuzzy matching.
            
        Returns:
            str: The response to the query.
        """
        query = query.lower().strip()
        
        # First, try direct pattern matching within intents
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                if pattern.lower() in query:
                    return random.choice(data["responses"])
        
        # If no direct match, try fuzzy matching the entire query
        # against patterns in all intents
        all_patterns = []
        pattern_to_intent = {}
        
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                all_patterns.append(pattern.lower())
                pattern_to_intent[pattern.lower()] = intent
        
        matches = get_close_matches(query, all_patterns, n=1, cutoff=threshold)
        if matches:
            matched_intent = pattern_to_intent[matches[0]]
            return random.choice(self.intents[matched_intent]["responses"])
        
        # If still no match, return default response
        return "Saa mbiri z'umugoroba"  # "I don't know the answer to that question"

class KinyarwandaSpeaker:
    """
    TTS component for Kinyarwanda speech synthesis.
    """
    def __init__(self):
        logger.info("Initializing Kinyarwanda TTS engine...")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Paths to TTS configuration and model
        path_to_tts_config = '/home/viateur/kinyarwanda-voice-assistant/Inference/kinyatts/ms_ktjw_istft_vits2_base.json'
        path_to_tts_model = '/home/viateur/Downloads/TTS_MODEL_ms_ktjw_istft_vits2_base_1M.pt'
        
        # Load TTS hyperparameters
        self.tts_hps = self.get_hparams_from_file(path_to_tts_config)
        
        # Initialize the TTS model
        self.tts_model = SynthesizerTrn(
            len(symbols),
            self.get_posterior_channels(),
            self.tts_hps.train.segment_size // self.tts_hps.data.hop_length,
            n_speakers=self.tts_hps.data.n_speakers,
            **self.tts_hps.model
        ).to(self.device)
        _ = self.tts_model.eval()
        _ = load_checkpoint(path_to_tts_model, self.tts_model, None)
        
        # Volume adjustment
        self.louder_vol = torchaudio.transforms.Vol(gain=3.0, gain_type="amplitude")
        logger.info("TTS engine ready!")
    
    def get_posterior_channels(self):
        """Determine posterior encoder type and channels."""
        if "use_mel_posterior_encoder" in self.tts_hps.model.keys() and self.tts_hps.model.use_mel_posterior_encoder:
            logger.info("Using mel posterior encoder for VITS2")
            posterior_channels = 80  # VITS2
            self.tts_hps.data.use_mel_posterior_encoder = True
        else:
            logger.info("Using lin posterior encoder for VITS1")
            posterior_channels = self.tts_hps.data.filter_length // 2 + 1
            self.tts_hps.data.use_mel_posterior_encoder = False
        return posterior_channels
    
    def get_hparams_from_file(self, config_path):
        """Load hyperparameters from config file."""
        with open(config_path, "r") as f:
            data = f.read()
        config = json.loads(data)
        hps = HParams(**config)
        return hps
    
    def get_text(self, text):
        """Convert input text to sequence of phonetic IDs."""
        text_norm = text_to_sequence(text)
        if self.tts_hps.data.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    
    def speak(self, text, output_wav_file='output.wav'):
        """
        Generate Kinyarwanda speech from text.
        
        Args:
            text (str): Input text in Kinyarwanda
            output_wav_file (str): Output file path
            
        Returns:
            tuple: (output_path, duration)
        """
        try:
            # Clean input text
            fltstr = re.sub(r"[\[\](){}]", "", text)
            
            # Convert text to phonetic sequence
            stn_tst = self.get_text(fltstr)
            
            # Generate audio
            with torch.no_grad():
                x_tst = stn_tst.to(self.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
                audio = self.tts_model.infer(
                    x_tst,
                    x_tst_lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0
                )[0][0, 0].data.cpu().float()
            
            # Calculate duration
            duration = audio.size(0) / self.tts_hps.data.sampling_rate
            
            # Increase volume and save
            audio = self.louder_vol(audio.unsqueeze(0))
            torchaudio.save(output_wav_file, audio, self.tts_hps.data.sampling_rate)
            
            return output_wav_file, duration
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            raise

class KinyarwandaVoiceAssistant:
    """
    Main class that integrates all components of the Kinyarwanda Voice Assistant.
    """
    
    def __init__(self, intents_file="nlp/intents.json"):
        """
        Initialize all components of the voice assistant.
        """
        logger.info("Initializing Kinyarwanda Voice Assistant components...")
        
        # Initialize components
        try:
            self.transcriber = KinyarwandaTranscriber()
            logger.info("Transcriber initialized")
            
            self.qa_engine = KinyarwandaQA(intents_file)
            logger.info("QA engine initialized")
            
            self.speaker = KinyarwandaSpeaker()
            logger.info("Speaker initialized")
            
            # Create output directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def process_audio(self, audio_file):
        """
        Process an audio file through the entire pipeline.
        """
        try:
            logger.info(f"Processing audio file: {audio_file}")
            
            # Step 1: Transcribe the audio
            transcription = self.transcriber.transcribe(audio_file)
            logger.info(f"Transcription: {transcription}")
            
            # Step 2: Get response from QA engine
            response = self.qa_engine.get_response(transcription)
            logger.info(f"Response: {response}")
            
            # Step 3: Generate speech for the response
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, "kinyarwanda_response.wav")
            
            output_path, duration = self.speaker.speak(response, output_file)
            logger.info(f"Speech generated: {output_path} (Duration: {duration:.2f}s)")
            
            return transcription, response, output_path
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return f"Error: {str(e)}", f"Error: {str(e)}", None

def create_gradio_interface():
    """
    Create the Gradio web interface for the Kinyarwanda Voice Assistant.
    """
    try:
        # Initialize the voice assistant
        assistant = KinyarwandaVoiceAssistant()
        
        # Define Gradio interface
        demo = gr.Interface(
            fn=assistant.process_audio,
            inputs=gr.Audio(source="microphone", type="filepath"),
            outputs=[
                gr.Textbox(label="Transcription"),
                gr.Textbox(label="Response"),
                gr.Audio(label="Spoken Response")
            ],
            title="Kinyarwanda Voice Assistant",
            description="Speak in Kinyarwanda and get a response in your language.",
            theme=gr.themes.Soft(),
            examples=[["test_audio/kinyarwanda.wav"]],
            allow_flagging="never"
        )
        
        return demo
        
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Create and launch the Gradio interface
        demo = create_gradio_interface()
        logger.info("Launching Gradio interface...")
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Error launching application: {str(e)}")
        print(f"Error: {str(e)}")