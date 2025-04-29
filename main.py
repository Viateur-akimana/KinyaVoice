import os
import logging
import traceback
from asr.transcribe import validate_paths, load_model, transcribe_audio
from nlp.qa_engine import KinyarwandaQA
from tts.speech_synth import kinya_tts_setup, kinya_tts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Step 1: Transcription
        logger.info("Validating paths and loading model...")
        audio_path = validate_paths()
        model, processor = load_model()
        
        logger.info("Starting transcription...")
        transcription = transcribe_audio(model, processor, audio_path)
        logger.info(f"Transcription: {transcription}")
        print(f"🗣️ Transcription: {transcription}")
        
        # Step 2: NLP - Get response
        logger.info("Initializing KinyarwandaQA...")
        qa_engine = KinyarwandaQA("nlp/intents.json")
        response = qa_engine.get_response(transcription)
        logger.info(f"Response: {response}")
        print(f"🤖 Response: {response}")
        
        # Step 3: TTS - Convert response to speech
        logger.info("Initializing TTS engine...")
        kinya_tts_setup()  # Make sure to initialize TTS system
        
        logger.info("Generating Kinyarwanda speech...")
        output_file, duration = kinya_tts(
            inputstr=response, 
            output_folder="outputs"
        )
        logger.info(f"Spoken response saved to: {output_file} (Duration: {duration:.2f} seconds)")
        print(f"🔊 Spoken response saved to: {output_file} (Duration: {duration:.2f} seconds)")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        print("See log for detailed traceback")

if __name__ == "__main__":
    main()