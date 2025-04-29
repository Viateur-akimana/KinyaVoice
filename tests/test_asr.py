import unittest
from asr.transcribe import KinyarwandaTranscriber

class TestASR(unittest.TestCase):
    def test_transcription(self):
        transcriber = KinyarwandaTranscriber()
        result = transcriber.transcribe("/test_audio/kinyarwanda.mp3")  # Replace with a valid audio file path
        self.assertIsInstance(result, str)