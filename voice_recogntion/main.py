import tempfile
import time
import wave
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np

from voice_recogntion.sox_recorder import SoxRecorder
from voice_recogntion.speech_to_text import SpeechToText

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


def main():
    audio_files_queue: Deque[Path] = deque()
    audio_buffers_queue = deque(maxlen=3)
    text_queue = deque(maxlen=3)
    stt = SpeechToText(MODEL, SCORER)

    with tempfile.TemporaryDirectory() as audio_dir:
        print(f"Recording!")
        sr = start_sox(Path(audio_dir), audio_files_queue)

        try:
            while True:
                if audio_files_queue:
                    print(f"Loading audio file")

                    wav_file = audio_files_queue.pop()
                    audio_buffer = load_wav(wav_file)
                    wav_file.unlink()

                    audio_buffers_queue.append(audio_buffer)

                print(f"Recognizing {len(audio_buffers_queue)} buffers")
                stt.stt_from_queue_to_queue(audio_buffers_queue, text_queue, greedy=True, pop=False)

                while text_queue:
                    print(f"Recognized f{text_queue.pop()!r}")

                time.sleep(0.5)

        except KeyboardInterrupt:
            sr.stop()


def recognize():
    stt = SpeechToText(MODEL, SCORER)
    audio = load_wav(Path("a.wav"))
    print(stt.stt(audio))


if __name__ == "__main__":
    main()
