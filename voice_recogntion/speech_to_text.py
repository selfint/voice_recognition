from typing import Deque
from pathlib import Path

import numpy as np
from deepspeech import Model


class SpeechToText:
    def __init__(self, model_path: Path, scorer_path: Path) -> None:
        print(f"Loading model from {str(model_path)!r}")
        self.model = Model(model_path=str(model_path.resolve()))

        print(f"Loading scorer from {str(scorer_path)!r}")
        self.model.enableExternalScorer(scorer_path=str(scorer_path.resolve()))

    def stt(self, audio_buffer: np.ndarray) -> str:
        """Run model on audio buffer

        Args:
            audio (np.ndarray): Audio buffer to run STT on

        Returns:
            str: Text recognized in audio buffer
        """

        return self.model.stt(audio_buffer)

    def stt_to_queue(self, audio_buffer: np.ndarray, output_queue: Deque[str]):
        """Run model on audio buffer and push text to output queue

        Args:
            audio (np.ndarray): Audio buffer to run STT on
            output_queue (Deque[str]): Queue to push text to
        """

        output_queue.append(self.stt(audio_buffer))



