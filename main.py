import os
import logging
import traceback
from asr.transcribe import KinyarwandaTranscriber
from nlp.qa_engine import KinyarwandaQA
from tts.speech_synth import KinyarwandaSpeaker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize components with verbose debugging
        logger.info("Initializing Kinyarwanda Voice Assistant components...")
        logger.info("Creating KinyarwandaTranscriber...")
        transcriber = KinyarwandaTranscriber()
        logger.info("Creating KinyarwandaQA...")
        qa_engine = KinyarwandaQA()
        logger.info("Creating KinyarwandaSpeaker...")
        speaker = KinyarwandaSpeaker()
        
        # Use the provided test audio file
        audio_file = "test_audio/kinyarwanda.wav"
        logger.info(f"Audio file path: {audio_file}")
        logger.info(f"Audio file exists: {os.path.exists(audio_file)}")
        
        # Transcribe with extra error handling
        try:
            logger.info("Starting transcription...")
            transcription = transcriber.transcribe(audio_file)
            logger.info(f"Transcription completed: {transcription}")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        print(f"Transcription: {transcription}")
        
        # Get response
        response = qa_engine.get_response(transcription)
        logger.info(f"Response: {response}")
        print(f"Response: {response}")
        
        # Speak response
        output_file = "response.wav" # Specify the output file name
        speaker.speak(response, output_file)
        logger.info(f"Spoken response generated as '{output_file}'")
        print(f"Spoken response generated as '{output_file}'")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        print("See log for detailed traceback")

if __name__ == "__main__":
    main()