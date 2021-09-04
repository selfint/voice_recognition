import subprocess
import tempfile
import time
import wave
from collections import deque
from multiprocessing.dummy import Process
from pathlib import Path
from typing import Callable, Deque

import numpy as np

from voice_recognition.sound_detector import SoundDetector
from voice_recognition.sox_recorder import SoxRecorder
from voice_recognition.speech_to_text import SpeechToText

MODEL = Path("models/english/deepspeech-0.9.3-models.pbmm")
SCORER = Path("models/english/deepspeech-0.9.3-models.scorer")


def start_sox(audio_dir: Path, duration: float, audio_queue: Deque[Path]):
    """Start the SoxRecorder.

    Args:
        audio_dir (Path): Directory to generate audio files in
        audio_queue (Deque[Path]): Queue to push generated audio files into
    """
    sr = SoxRecorder(Path(audio_dir), audio_queue)
    sr.start(duration)

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
        return np.zeros(16000)
    audio = np.frombuffer(fin.readframes(fin.getnframes()))
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
        audio_buffer = np.frombuffer(audio_file.read())

    return audio_buffer


def file_loader(
    load_func: Callable[[Path], np.ndarray],
    audio_files: Deque[Path],
    output_queue: Deque[np.ndarray],
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
        if audio_files:
            audio_file = audio_files.pop()
            audio_buffer = load_func(audio_file)
            audio_file.unlink()

            output_queue.append(audio_buffer)

            time.sleep(0.5)


def audio_bucketer(
    sd: SoundDetector,
    audio_buffers: Deque[np.ndarray],
    output_queue: Deque[np.ndarray],
):
    """Bucket audio samples by sound continuation.

    Read and expand audio buffers from the ``audio_buffers`` queue.
    When a period of "silence, sound, silence" is detected, then it is merged
    into a single audio buffer and pushed to the ``output_queue``.

    Args:
        sd: SoundDetector to use for sound detection
        audio_buffers: Queue with audio buffers to bucket
        output_queue: Queue to push audio buckets to
    """

    print("Audio bucketer started")

    audio_buffer = np.array([])

    while True:
        if audio_buffers:
            # append next buffer to the audio buffer
            next_buffer = audio_buffers.pop()
            audio_buffer = np.concatenate((audio_buffer, next_buffer))

            sounds = sd.detect_sound(audio_buffer, keep_silence=True)
            print(sounds)

            new_start = 0
            for start, end in sounds:
                speak_buffer(audio_buffer[start:end])
                sound = np.frombuffer(audio_buffer[start:end].tobytes(), dtype=np.int16)
                new_start = end

                # do not push the last sound if it is not finished yet
                if end == len(audio_buffer):
                   new_start = start

                   # this also means that it is the last sound and we can break
                   break

                output_queue.append(sound)

            # remove all parts of the buffer that have been processed
            audio_buffer = audio_buffer[new_start:].copy()

            max_buffer_size = SoundDetector.ms_to_byte_index(10000, 16000, 16)
            if len(audio_buffer) >= max_buffer_size:
                speak_buffer(audio_buffer)
                output_queue.append(np.array(audio_buffer, dtype=np.int16))
                audio_buffer = np.array([])

            time.sleep(0.5)


def speak_buffer(audio_buffer: np.ndarray):
    audio_buffer.tofile("sound.raw")
    print("## speaking ##")
    speaker = subprocess.Popen(
        "play -r 16k -b 16 -e signed-integer -q sound.raw",
        shell=True,
    )
    # print(speaker.communicate())


def recognizer(
    stt: SpeechToText, audio_buffers: Deque[np.ndarray], output_queue: Deque[str]
):
    """Recognize text in audio buffers queue, save results in the text queue.

    Use ``stt`` to recognize speech in audio buffers from the
    ``audio_buffers_queue``, push recognized text to the ``text_queue``.

    Args:
        stt (SpeechToText): SpeechToText to use for voice recognition
        audio_buffers (Deque[np.ndarray]): Queue with audio buffers to recognize
        output_queue (Deque[str]): Queue to push recognized text to
    """

    print("Recognizer started")
    while True:
        stt.stt_from_queue_to_queue(
            audio_buffers,
            output_queue,
            greedy=True,
            pop=True,
        )

        time.sleep(0.5)


def run_continuous_asynchronously():
    audio_files_queue: Deque[Path] = deque()
    audio_buffers_queue: Deque[np.ndarray] = deque()
    audio_buckets_queue: Deque[np.ndarray] = deque()
    text_queue: Deque[str] = deque()

    stt = SpeechToText(MODEL, SCORER)
    sd = SoundDetector(-48, 250)

    file_loader_process = Process(
        target=file_loader, args=(load_raw, audio_files_queue, audio_buffers_queue)
    )
    file_loader_process.daemon = True

    audio_bucketer_process = Process(
        target=audio_bucketer,
        args=(sd, audio_buffers_queue, audio_buckets_queue),
    )
    audio_bucketer_process.daemon = True

    recognizer_process = Process(
        target=recognizer, args=(stt, audio_buckets_queue, text_queue)
    )
    recognizer_process.daemon = True

    with tempfile.TemporaryDirectory() as audio_dir:
        print(f"Voice io started!")
        sr = start_sox(Path(audio_dir), 3, audio_files_queue)
        file_loader_process.start()
        audio_bucketer_process.start()
        recognizer_process.start()

        try:
            while True:
                # print queue states
                # print(f"Files: {list(audio_files_queue)!r}")
                # print(f"Audio: {list(audio_buffers_queue)!r}")
                # print(f"Buckets: {list(audio_buckets_queue)!r}")
                # print(f"Text: {list(text_queue)!r}")

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
    audio = sr.record_once()
    speak_buffer(audio)
    print(stt.stt(audio))


if __name__ == "__main__":
    run_continuous_asynchronously()
