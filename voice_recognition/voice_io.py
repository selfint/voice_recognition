import subprocess
import tempfile
import time
import wave
from collections import deque
from multiprocessing.dummy import Process
from pathlib import Path
from typing import Callable, Deque, Optional

import numpy as np

from voice_recognition.sound_detector import SoundDetector
from voice_recognition.sox_recorder import SoxRecorder
from voice_recognition.speech_to_text import SpeechToText


class VoiceIO:
    def __init__(
        self,
        model: Path,
        scorer: Path,
        silence_threshold: int,
        min_silence_duration: int,
    ) -> None:
        """Allow talking with a user through voice.

        The user is recorded continuously, and all audio files are saved in the
        given ``audio_dir``. The audio files are then processed to extract
        non-silences, which try to be the shortest possible length. The best
        case scenario is word by word. These non-silences are then run through
        a speech to text model (powered by Mozilla deepspeech), and the text
        recognized in the audio is pushed to a given ``output_queue`` (only
        after ``start`` is called).


        Args:
            model: Path to .pbmm model to use
            scorer: Path to .scorer to user
            silence_threshold: The upper bound for how quiet is silent in dFBS
            min_silence_duration: Minimum duration (in milliseconds) of a silence
        """
        self._stt = SpeechToText(model, scorer)
        self._sd = SoundDetector(silence_threshold, min_silence_duration)

        self._sox_recorder: Optional[SoxRecorder] = None

        self._stop = False

    def start(
        self,
        audio_dir: Path,
        audio_length: float,
        output_queue: Deque[str],
    ):
        """Start recording the user and push recognized text to the ``output_queue``

        Starts a pipeline with these stages:
            1. Record user using SoX and save audio files in a temporary directory.
            2. Load the raw audio files into numpy array buffers.
            3. Extract non-silence periods from buffers, and group continuous
               non-silences.
            4. Write grouped non-silences to .wav files in the given ``audio_dir``.
            5. Load .wav files into numpy array buffers.
            6. Run speech to text on the buffers and push recognized text to
               the ``output_queue``.

        Stages 4-6 are redundant, and might be removed in the future.

        Args:
            audio_dir: Directory to save recorded .wav files in
            audio_length: Length of each file SoX records
            output_queue: Queue to push recognized text to
        """

        _raw_audio_file_paths: Deque[Path] = deque()
        _raw_audio_buffers: Deque[np.ndarray] = deque()
        _group_non_silences: Deque[np.ndarray] = deque()
        _wav_file_paths: Deque[Path] = deque()
        _wav_audio_buffers: Deque[np.ndarray] = deque()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._sox_recorder = SoxRecorder(audio_dir, output_queue)
            # TODO: make a class for the File loaders
            # TODO: add the rest of the processes as attributes
            # TODO: start all components in separate processes (or threads)

    @staticmethod
    def load_raw_audio_file(path: Path, dtype: np.dtype) -> np.ndarray:
        """Load raw audio file at ``path`` into a numpy array with dtype ``dtype``.

        Args:
            path: Path to raw audio file
            dtype: Dtype to use in numpy array

        Returns:
            np.ndarray: Raw audio content as a numpy array
        """

        with path.open("rb") as audio_file:
            audio_buffer = np.frombuffer(audio_file.read(), dtype=dtype)

        return audio_buffer


    @staticmethod
    def load_wav(path: Path, dtype: np.dtype) -> np.ndarray:
        """Load .wav file at ``path`` into a numpy array with dtype ``dtype``.

        Args:
            path: .wav file to load into numpy array
            dtype: Dtype to use in numpy array

        Returns:
            np.array: .wav file content as a numpy array
        """

        try:
            fin = wave.open(str(path.resolve()), "rb")
        except EOFError:
            return np.array([])

        audio = np.frombuffer(fin.readframes(fin.getnframes()), dtype=dtype)
        fin.close()

        return audio
