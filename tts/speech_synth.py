from __future__ import print_function, division

import warnings
from typing import Tuple
import os
import torch
import torchaudio
import re
from pathlib import Path
import logging
from kinyatts.tts.commons import intersperse
from kinyatts.tts.utils import get_hparams_from_file, load_checkpoint
from kinyatts.tts.models import SynthesizerTrn
from kinyatts.tts.text import text_to_sequence
from kinyatts.tts.text.symbols import symbols

warnings.filterwarnings("ignore")

# Global inference engine
inference_engine = (None, None, None, None)

def kinya_tts_setup():
    """
    Set up the Kinyarwanda TTS engine.
    This function initializes the TTS model, loads the configuration, and prepares the inference engine.
    """
    global inference_engine

    # Select device (GPU if available, otherwise CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Paths to TTS configuration and model
    path_to_tts_config = '/home/viateur/kinyarwanda-voice-assistant/Inference/kinyatts/ms_ktjw_istft_vits2_base.json'
    path_to_tts_model = '/home/viateur/Downloads/TTS_MODEL_ms_ktjw_istft_vits2_base_1M.pt'

    # Load TTS hyperparameters
    tts_hps = get_hparams_from_file(path_to_tts_config)

    # Determine posterior encoder type
    if "use_mel_posterior_encoder" in tts_hps.model.keys() and tts_hps.model.use_mel_posterior_encoder:
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # VITS2
        tts_hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = tts_hps.data.filter_length // 2 + 1
        tts_hps.data.use_mel_posterior_encoder = False

    # Initialize the TTS model
    tts_model = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        tts_hps.train.segment_size // tts_hps.data.hop_length,
        n_speakers=tts_hps.data.n_speakers,  # >0 for multi-speaker
        **tts_hps.model
    ).to(device)
    _ = tts_model.eval()
    _ = load_checkpoint(path_to_tts_model, tts_model, None)

    # Volume adjustment
    louder_vol = torchaudio.transforms.Vol(gain=3.0, gain_type="amplitude")

    # Set up the inference engine
    inference_engine = (device, tts_model, tts_hps, louder_vol)
    print('TTS API engine ready!', flush=True)

def get_text(text, hps):
    """
    Convert input text to a sequence of phonetic IDs for TTS processing.
    
    Args:
        text (str): Input text in Kinyarwanda.
        hps: Model hyperparameters.
        
    Returns:
        torch.LongTensor: Tensor of token IDs representing the input text.
    """
    text_norm = text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def kinya_tts(inputstr, output_folder=None, output_wav_file='igihe.wav') -> Tuple[str, float]:
    """
    Generate Kinyarwanda speech from text input.
    
    Args:
        inputstr (str): Input text in Kinyarwanda.
        output_folder (str, optional): Folder to save the generated audio. Will be created if it doesn't exist.
        output_wav_file (str): Filename or path to save the generated audio.
        
    Returns:
        Tuple[str, float]: Path to the generated audio file and its duration in seconds.
    """
    global inference_engine
    (device, tts_model, tts_hps, louder_vol) = inference_engine

    # Handle output folder if provided
    if output_folder:
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Combine the folder and filename
        full_output_path = os.path.join(output_folder, os.path.basename(output_wav_file))
    else:
        full_output_path = output_wav_file

    # Remove brackets and other special characters
    fltstr = re.sub(r"[\[\](){}]", "", inputstr)

    # Convert text to phonetic sequence
    stn_tst = get_text(fltstr, tts_hps)

    # Set speech generation speed
    speed = 1.0

    # Generate audio using the TTS model
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = tts_model.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1 / speed
        )[0][0, 0].data.cpu().float()

    # Calculate audio duration
    AUDIO_TIME = audio.size(0) / tts_hps.data.sampling_rate

    # Increase volume
    audio = louder_vol(audio.unsqueeze(0))

    # Save audio to WAV file
    torchaudio.save(full_output_path, audio, tts_hps.data.sampling_rate)

    return full_output_path, AUDIO_TIME

# Example usage
if __name__ == "__main__":
    try:
        # Set up the TTS engine
        kinya_tts_setup()

        # Generate speech
        output_file, duration = kinya_tts(
            inputstr="Muraho neza, urakomeye?",  # "Hello, how are you?"
            output_folder="outputs"
        )
        print(f"Generated audio: {output_file} (Duration: {duration:.2f} seconds)")
    except Exception as e:
        logging.error(f"Error generating speech: {e}")