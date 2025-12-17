import pyaudio
import wave

def record_audio(filename: str, duration: int = 5):
    """
    Records voice from microphone and saves as WAV
    """
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000

    p = pyaudio.PyAudio()

    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk
    )

    frames = []
    for _ in range(int(rate / chunk * duration)):
        frames.append(stream.read(chunk))

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
