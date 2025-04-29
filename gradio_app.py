import os
import logging
import gradio as gr
import tempfile
from pathlib import Path

from asr.transcribe import KinyarwandaTranscriber
from nlp.qa_engine import KinyarwandaQA
from tts.speech_synth import KinyarwandaSpeaker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KinyarwandaVoiceAssistant:
    """
    Main class that integrates all components of the Kinyarwanda Voice Assistant.
    """
    
    def __init__(self, intents_file="nlp/intents.json"):
        """
        Initialize all components of the voice assistant.
        
        Args:
            intents_file (str): Path to the intents JSON file for the QA engine.
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
        
        Args:
            audio_file (str): Path to the audio file to process.
            
        Returns:
            tuple: (transcription, response, output_audio_path)
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
            # Create temporary file for Gradio compatibility
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
    
    Returns:
        gr.Interface: The Gradio interface object.
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
            examples=[["test_audio/kinyarwanda.mp3"]],
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