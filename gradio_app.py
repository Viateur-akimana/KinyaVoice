import gradio as gr
from asr.transcribe import KinyarwandaTranscriber
from nlp.qa_engine import KinyarwandaQA
from tts.speech_synth import KinyarwandaSpeaker

transcriber = KinyarwandaTranscriber()
qa_engine = KinyarwandaQA("nlp/intents.json")
speaker = KinyarwandaSpeaker()

def process_audio(audio_file):
    transcription = transcriber.transcribe(audio_file)
    response = qa_engine.get_response(transcription)
    output_file = "response.wav"
    speaker.speak(response, output_file)
    return transcription, response, output_file

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Response"),
        gr.Audio(label="Spoken Response")
    ],
    title="Kinyarwanda Voice Assistant",
    description="Speak in Kinyarwanda and get a response in your language."
)

if __name__ == "__main__":
    demo.launch()