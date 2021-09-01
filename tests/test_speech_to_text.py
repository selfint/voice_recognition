from pathlib import Path
from voice_recogntion.speech_to_text import SpeechToText


MODEL = Path("models/english/deepspeech-0.9.3-models.pbmm")
SCORER = Path("models/english/deepspeech-0.9.3-models.scorer")


def test_load_model():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)

    assert stt.model is not None
