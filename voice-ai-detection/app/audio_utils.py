# app/audio_utils.py
import base64
import io
import librosa
from pydub import AudioSegment

def decode_and_preprocess(base64_audio: str):
    try:
        audio_bytes = base64.b64decode(base64_audio)

        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        signal, _ = librosa.load(wav_io, sr=16000)
        return signal

    except Exception as e:
        raise ValueError("Invalid or corrupted MP3 audio")
