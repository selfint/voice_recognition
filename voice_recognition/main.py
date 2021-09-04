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


def load_wav(wav_file: Path, dtype: np.dtype) -> np.ndarray:
    """Load .wav file into a numpy array

    Args:
        wav_file (Path): .wav file to load into numpy array
        dtype: Dtype to use in numpy array

    Returns:
        np.array: .wav file content as a numpy array
    """

    # load .wav file data
    try:
        fin = wave.open(str(wav_file.resolve()), "rb")
    except EOFError:
        return np.zeros(16000)
    audio = np.frombuffer(fin.readframes(fin.getnframes()), dtype=dtype)
    fin.close()

    return audio


def load_raw(raw_file: Path, dtype: np.dtype) -> np.ndarray:
    """Load raw audio file into a numpy array

    Args:
        raw_file: Path to raw audio file
        dtype: Dtype to use in numpy array

    Returns:
        np.ndarray: Raw audio content as a numpy array
    """

    with raw_file.open("rb") as audio_file:
        audio_buffer = np.frombuffer(audio_file.read(), dtype=dtype)

    return audio_buffer


def file_loader(
    load_func: Callable[[Path, np.dtype], np.ndarray],
    dtype: np.dtype,
    audio_files: Deque[Path],
    output_queue: Deque[np.ndarray],
    keep_file: bool = False,
):
    """
    Load WAV files in audio files queue into audio buffers queue.
    Args:
        load_func: Function to call to load the file into an audio buffer
        dtype: Dtype to use in numpy array
        audio_files_queue: Queue containing paths of audio files to load
        audio_buffers_queue: Queue to upload audio buffers to
        keep_file: Whether to not delete the file after loading it
    """

    print("File loader started")
    while True:
        if audio_files:
            audio_file = audio_files.popleft()
            audio_buffer = load_func(audio_file, dtype)
            if not keep_file:
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
    max_buffer_size = SoundDetector.ms_to_byte_index(5000, 16000, 16)

    while True:
        if audio_buffers:
            # append next buffer to the audio buffer
            next_buffer = audio_buffers.popleft()
            audio_buffer = np.concatenate((audio_buffer, next_buffer))

            sounds = sd.detect_sound(audio_buffer, keep_silence=True)

            new_start = 0
            for start, end in sounds:
                sound = np.frombuffer(audio_buffer[start:end].tobytes(), dtype=np.int16)
                new_start = end

                # do not push the last sound if it is not finished yet
                if end < len(audio_buffer):
                    output_queue.append(sound)
                # unless it is longer than the maximum buffer size
                elif end - start >= max_buffer_size:
                    output_queue.append(sound)
                # if it is not longer then keep it
                else:
                    new_start = start

            # remove all parts of the buffer that have been processed
            audio_buffer = audio_buffer[new_start:].copy()

            time.sleep(0.5)


def file_writer(
    location: Path, audio_buffers: Deque[np.ndarray], output_queue: Deque[Path]
):
    """Write audio buffers to files and push their filename to ``output_queue``

    Take the raw bytes of each numpy array in the ``audio_buffers`` queue,
    write them to a temporary file, then use sox to convert the raw audio
    to a .wav file. The .wav files will be save at the given ``location``.

    Args:
        location: Path to save all .wav files in
        audio_buffers: Queue with buffers to write to .wav files
        output_queue: Queue to push .wav filenames to
    """

    print("File writer started")

    count = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        while True:
            if audio_buffers:
                buffer = audio_buffers.popleft()
                temp_filename = f"{temp_dir}/sound{count:03}.raw"
                buffer.tofile(temp_filename)
                count += 1

                wav_filename = Path(temp_filename).with_suffix(".wav").name
                wav_path = location / wav_filename
                convert_cmd = (
                    f"sox -r 16k -b 16 -e signed-integer {temp_filename}"
                    f" {wav_path.as_posix()}"
                )
                subprocess.Popen(convert_cmd, shell=True).wait()
                Path(temp_filename).unlink()

                speak_wav(wav_path)
                output_queue.append(wav_path)

            time.sleep(0.5)


def speak_buffer(audio_buffer: np.ndarray):
    audio_buffer.tofile("sound.raw")
    print("## speaking raw ##")
    subprocess.Popen(
        "play -r 16k -b 16 -e signed-integer -q sound.raw",
        shell=True,
    )


def speak_wav(wav_path: Path):
    print("## speaking .wav ##")
    subprocess.Popen(
        f"play -q {wav_path.as_posix()}",
        shell=True,
    )


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
        buffer = []
        while audio_buffers:
            buffer.append(audio_buffers.popleft())

        if len(buffer) == 0:
            continue

        # using a single buffer is faster than using a list of buffers
        # TODO: is it though?
        if len(buffer) == 1:
            buffer = buffer[0]

        text = stt.stt(buffer)
        output_queue.append(text)

        time.sleep(0.5)


def run_continuous_asynchronously():
    audio_files_queue: Deque[Path] = deque()
    audio_buffers_queue: Deque[np.ndarray] = deque()
    audio_buckets_queue: Deque[np.ndarray] = deque()
    wav_files_queue: Deque[Path] = deque()
    wav_audio_queue: Deque[np.ndarray] = deque()
    text_queue: Deque[str] = deque()

    stt = SpeechToText(MODEL, SCORER)
    sd = SoundDetector(-48, 250)
    transcript = Path("transcript.txt")

    file_loader_process = Process(
        target=file_loader,
        args=(load_raw, np.float64, audio_files_queue, audio_buffers_queue),
    )
    file_loader_process.daemon = True

    audio_bucketer_process = Process(
        target=audio_bucketer,
        args=(sd, audio_buffers_queue, audio_buckets_queue),
    )
    audio_bucketer_process.daemon = True

    file_writer_process = Process(
        target=file_writer,
        args=(Path("sounds"), audio_buckets_queue, wav_files_queue),
    )
    file_writer_process.daemon = True

    wav_loader_process = Process(
        target=file_loader,
        args=(load_wav, np.int16, wav_files_queue, wav_audio_queue),
        kwargs={"keep_file": True},
    )
    wav_loader_process.daemon = True

    recognizer_process = Process(
        target=recognizer, args=(stt, wav_audio_queue, text_queue)
    )
    recognizer_process.daemon = True

    with tempfile.TemporaryDirectory() as audio_dir:
        print(f"Voice io started!")
        sr = start_sox(Path(audio_dir), 1, audio_files_queue)
        file_loader_process.start()
        audio_bucketer_process.start()
        file_writer_process.start()
        wav_loader_process.start()
        recognizer_process.start()

        try:
            with transcript.open("a") as text_file:
                while True:
                    # print queue states
                    # print(f"Files: {list(audio_files_queue)!r}")
                    # print(f"Audio: {list(audio_buffers_queue)!r}")
                    # print(f"Buckets: {list(audio_buckets_queue)!r}")
                    # print(f"Text: {list(text_queue)!r}")

                    while text_queue:
                        text = text_queue.popleft()
                        print(f"{text!r}")
                        text_file.write(text + "\n")

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
