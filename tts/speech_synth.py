from TTS.api import TTS

class KinyarwandaSpeaker:
    def __init__(self):
        self.tts = TTS(model_name="tts_models/multilingual/mms")
        self.speaker_id = "spk_10"

    def speak(self, text, output_file="response.wav"):  
        self.tts.tts_to_file(text=text, file_path=output_file, speaker=self.speaker_id)