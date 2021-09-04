from pathlib import Path
from typing import Deque, List, Union

import numpy as np
from deepspeech import Model


class SpeechToText:
    def __init__(self, model_path: Path, scorer_path: Path) -> None:
        print(f"Loading model from {str(model_path)!r}")
        self.model = Model(model_path=str(model_path.resolve()))

        print(f"Loading scorer from {str(scorer_path)!r}")
        self.model.enableExternalScorer(scorer_path=str(scorer_path.resolve()))

    def stt(self, audio_buffer: Union[np.ndarray, List[np.ndarray]]) -> str:
        """Recognize text in audio buffer.

        Args:
            audio_buffer (np.ndarray): Audio buffer to run STT on

        Returns:
            str: Text recognized in audio buffer
        """

        if isinstance(audio_buffer, np.ndarray):
            return self.model.stt(audio_buffer)

        elif isinstance(audio_buffer, list):

            # merge all audio buffers into a stream
            stream = self.model.createStream()
            for buffer in audio_buffer:
                stream.feedAudioContent(buffer)

            return stream.finishStream()

    def stt_to_queue(
        self,
        audio_buffer: Union[np.ndarray, List[np.ndarray]],
        output_queue: Deque[str],
    ):
        """Run model on audio buffer and push text to output queue

        Args:
            audio_buffer (np.ndarray): Audio buffer to run STT on
            output_queue (Deque[str]): Queue to push text to
        """

        output_queue.append(self.stt(audio_buffer))

    def stt_from_queue_to_queue(
        self,
        input_queue: Deque[np.ndarray],
        output_queue: Deque[str],
        greedy: bool = False,
        pop: bool = True,
    ):
        """Pipe recognized text from audio buffer in the input queue to the output queue

        Pop an audio buffer from the `input_queue`. Then, perform STT on the
        popped audio buffer, push the recognized text to the `output_queue`.

        If `input_queue` is empty, nothing is done.

        If `greedy` is set then the entire input queue is consumed,
        if not then only the first element in the queue is consumed.

        If 'pop` is set then values are removed from the input queue.

        Args:
            input_queue (Deque[numpy.ndarray]): Queue to pop audio buffer from
            output_queue (Deque[str]): Queue to push text to
            greedy (bool): Whether to consume entire queue or only the first element
            pop (bool): Whether to remove elements from the input queue
        """

        # make sure input_queue is not empty
        if input_queue:
            audio_buffer: Union[np.ndarray, List[np.ndarray]]

            if greedy and pop:
                audio_buffer = []
                while input_queue:
                    audio_buffer.append(input_queue.popleft())
            elif greedy and not pop:
                audio_buffer = list(input_queue)
            elif not greedy and pop:
                audio_buffer = input_queue.popleft()
            else:
                audio_buffer = input_queue[0]

            if isinstance(audio_buffer, list) and len(audio_buffer) == 1:
                audio_buffer = audio_buffer[0]

            self.stt_to_queue(audio_buffer, output_queue)
