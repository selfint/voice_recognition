from pathlib import Path
from voice_recogntion.speech_to_text import SpeechToText


MODEL = Path("models/english/deepspeech-0.9.3-models.pbmm")


def test_load_model():
    stt = SpeechToText(model_path=MODEL)

    assert stt.model is not None
