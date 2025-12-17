from pydub import AudioSegment
import os
import uuid

def convert_to_wav(input_path: str) -> str:
    """
    Converts any audio format to WAV (16kHz, mono, PCM)
    Returns the converted WAV path
    """
    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    output_path = f"temp_{uuid.uuid4().hex}.wav"
    audio.export(output_path, format="wav")

    return output_path
