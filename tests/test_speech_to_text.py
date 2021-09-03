from collections import deque
from pathlib import Path

import numpy as np

from voice_recognition.speech_to_text import SpeechToText

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


def test_multiple_audio_buffers():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)

    audio_buffers = [np.zeros(16000, dtype=np.int16), np.zeros(16000, dtype=np.int16)]

    assert stt.stt(audio_buffers) == ""


def test_speech_to_text_to_queue():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    queue = deque()

    audio_buffer = np.zeros(16000, dtype=np.int16)

    stt.stt_to_queue(audio_buffer, queue)

    assert list(queue) == [""]


def test_speech_to_text_from_queue_to_queue():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    input_queue = deque()
    output_queue = deque()

    audio_buffer = np.zeros(16000, dtype=np.int16)

    input_queue.append(audio_buffer)

    stt.stt_from_queue_to_queue(input_queue, output_queue)

    assert list(output_queue) == [""]

def test_from_queue_greedy():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    input_queue = deque()
    output_queue = deque()

    audio_buffers = [np.zeros(16000, dtype=np.int16), np.zeros(16000, dtype=np.int16)]

    input_queue.extend(audio_buffers)

    stt.stt_from_queue_to_queue(input_queue, output_queue, greedy=True)

    assert list(output_queue) == [""]

    # make sure input queue is empty
    assert not input_queue

def test_from_queue_greedy_no_pop():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    input_queue = deque()
    output_queue = deque()

    audio_buffers = [np.zeros(16000, dtype=np.int16), np.zeros(16000, dtype=np.int16)]

    input_queue.extend(audio_buffers)

    stt.stt_from_queue_to_queue(input_queue, output_queue, greedy=True, pop=False)

    assert list(output_queue) == [""]

    # make sure input queue is not empty
    assert list(input_queue) == audio_buffers

def test_from_queue_no_greedy_no_pop():
    stt = SpeechToText(model_path=MODEL, scorer_path=SCORER)
    input_queue = deque()
    output_queue = deque()

    audio_buffers = [np.zeros(16000, dtype=np.int16), np.zeros(16000, dtype=np.int16)]

    input_queue.extend(audio_buffers)

    stt.stt_from_queue_to_queue(input_queue, output_queue, greedy=False, pop=False)

    assert list(output_queue) == [""]

    # make sure input queue is not empty
    assert list(input_queue) == audio_buffers
