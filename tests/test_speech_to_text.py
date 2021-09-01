from collections import deque
from pathlib import Path

import numpy as np

from voice_recogntion.speech_to_text import SpeechToText

MODEL = Path("models/english/deepspeech-0.9.3-models.pbmm")
SCORER = Path("models/english/deepspeech-0.9.3-models.scorer")


def test_load_model():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)

    assert stt.model is not None

def test_speech_to_text():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)

    # create a buffer of one second of silence (with 16000 sample rate)
    audio_buffer = np.zeros(16000, dtype=np.int16)

    # stt should not recognize a word, but should still work
    assert stt.stt(audio_buffer) == ""

def test_speech_to_text_to_queue():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    queue = deque()

    audio_buffer = np.zeros(16000, dtype=np.int16)

    stt.stt_to_queue(audio_buffer, queue)

    assert list(queue) == [""]

