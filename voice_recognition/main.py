import tempfile
import time
import wave
from collections import deque
from multiprocessing.dummy import Process
from pathlib import Path
from typing import Callable, Deque

import numpy as np
from voice_recognition.sox_recorder import SoxRecorder
from voice_recognition.speech_to_text import SpeechToText

MODEL = Path("models/english/deepspeech-0.9.3-models.pbmm")
SCORER = Path("models/english/deepspeech-0.9.3-models.scorer")


def start_sox(audio_dir: Path, audio_queue: Deque[Path]):
    """Start the SoxRecorder.

    Args:
        audio_dir (Path): Directory to generate audio files in
        audio_queue (Deque[Path]): Queue to push generated audio files into
    """
    sr = SoxRecorder(Path(audio_dir), audio_queue)
    sr.start()

    return sr


def load_wav(wav_file: Path) -> np.ndarray:
    """Load .wav file into a numpy array

    Args:
        wav_file (Path): .wav file to load into numpy array

    Returns:
        np.array: .wav file content as a numpy array
    """

    # load .wav file data
    try:
        fin = wave.open(str(wav_file.resolve()), "rb")
    except EOFError:
        return np.zeros(16000, np.int16)
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    fin.close()

    return audio

def load_raw(raw_file: Path) -> np.ndarray:
    """Load raw audio file into a numpy array

    Args:
        raw_file: Path to raw audio file

    Returns:
        np.ndarray: Raw audio buffer in file
    """

    with raw_file.open("rb") as audio_file:
        audio_buffer = np.frombuffer(audio_file.read(), np.int16)

    return audio_buffer


def file_loader(
    load_func: Callable[[Path], np.ndarray],
    audio_files_queue: Deque[Path],
    audio_buffers_queue: Deque[np.ndarray]
):
    """
    Load WAV files in audio files queue into audio buffers queue.
    Args:
        load_func: Function to call to load the file into an audio buffer
        audio_files_queue: Queue containing paths of audio files to load
        audio_buffers_queue: Queue to upload audio buffers to
    """

    print("File loader started")
    while True:
        if audio_files_queue:
            audio_file = audio_files_queue.pop()
            audio_buffer = load_func(audio_file)
            audio_file.unlink()

            audio_buffers_queue.append(audio_buffer)

            time.sleep(0.5)


def recognizer(
    stt: SpeechToText, audio_buffers_queue: Deque[np.ndarray], text_queue: Deque[str]
):
    """Recognize text in audio buffers queue, save results in the text queue.

    Use ``stt`` to recognize speech in audio buffers from the ``audio_buffers_queue``,
    push recognized text to the ``text_queue``.

    Args:
        stt (SpeechToText): SpeechToText to use for voice recognition
        audio_buffers_queue (Deque[np.ndarray]): Queue with audio buffers to recognize
        text_queue (Deque[str]): Queue to push recognized text to
    """

    print("Recognizer started")
    while True:
        stt.stt_from_queue_to_queue(
            audio_buffers_queue,
            text_queue,
            greedy=True,
            pop=False
        )

        time.sleep(0.5)


def run_continuous_asynchronously():
    audio_files_queue: Deque[Path] = deque()
    audio_buffers_queue: Deque[np.ndarray] = deque(maxlen=2)
    text_queue: Deque[str] = deque()

    stt = SpeechToText(MODEL, SCORER)

    file_loader_process = Process(
        target=file_loader, args=(load_raw, audio_files_queue, audio_buffers_queue)
    )
    file_loader_process.daemon = True

    recognizer_process = Process(
        target=recognizer, args=(stt, audio_buffers_queue, text_queue)
    )
    recognizer_process.daemon = True

    with tempfile.TemporaryDirectory() as audio_dir:
        print(f"Voice io started!")
        sr = start_sox(Path(audio_dir), audio_files_queue)
        file_loader_process.start()
        recognizer_process.start()

        try:
            while True:
                while text_queue:
                    print(f"{text_queue.pop()!r}")

                time.sleep(0.5)

        except KeyboardInterrupt:
            sr.stop()


def run_continuous_interactively():
    sr = SoxRecorder()
    stt = SpeechToText(MODEL, SCORER)
    while True:
        print(stt.stt(sr.record_once()))


def run_one_interactively():
    sr = SoxRecorder()
    stt = SpeechToText(MODEL, SCORER)
    print(stt.stt(sr.record_once()))


if __name__ == "__main__":
    run_continuous_asynchronously()
