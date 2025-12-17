import json
import wave
import os
from vosk import Model, KaldiRecognizer
from audio.audio_utils import convert_to_wav

MODEL_PATH = "vosk-model-en-us-0.22-lgraph"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {MODEL_PATH}")

model = Model(MODEL_PATH)

def transcribe(audio_path: str) -> str:
    """
    Transcribe WAV / MP3 / M4A / FLAC safely using Vosk
    """
    # 🔑 ALWAYS convert to WAV first
    wav_path = convert_to_wav(audio_path)

    wf = wave.open(wav_path, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text += result.get("text", "") + " "

    final_result = json.loads(rec.FinalResult())
    text += final_result.get("text", "")

    wf.close()

    # Optional cleanup
    try:
        os.remove(wav_path)
    except:
        pass

    return text.strip()
