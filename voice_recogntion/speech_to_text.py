import wave
from pathlib import Path

import numpy as np
from deepspeech import Model


class SpeechToText:
    def __init__(self, model_path: Path, scorer_path: Path) -> None:
        print(f"Loading model from {str(model_path)!r}")
        self.model = Model(model_path=str(model_path.resolve()))

        print(f"Loading scorer from {str(scorer_path)!r}")
        self.model.enableExternalScorer(scorer_path=str(scorer_path.resolve()))

    def load_wav(self, wav_file: Path) -> np.ndarray:
        """Load .wav file into a numpy array

        Args:
            wav_file (Path): .wav file to load into numpy array

        Returns:
            np.array: .wav file content as a numpy array
        """

        # load .wav file data
        fin = wave.open(str(wav_file.resolve()), "rb")
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        fin.close()

        return audio

    def stt(self, audio_buffer: np.ndarray) -> str:
        """Run model on audio buffer

        Args:
            audio (np.ndarray): Audio buffer to run STT on

        Returns:
            str: Text recognized in audio buffer
        """

        return self.model.stt(audio_buffer)


