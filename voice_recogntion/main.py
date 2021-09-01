import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Deque

from voice_recogntion.sox_recorder import SoxRecorder


def start_sox(audio_dir: Path, audio_queue: Deque[Path]):
    """Start the SoxRecorder.

    Args:
        audio_dir (Path): Directory to generate audio files in
        audio_queue (Deque[Path]): Queue to push generated audio files into
    """
    sr = SoxRecorder(Path(audio_dir), audio_queue)
    sr.start()

    return sr


def main():
    audio_queue = deque()

    with tempfile.TemporaryDirectory() as audio_dir:
        sr = start_sox(Path(audio_dir), audio_queue)
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            sr.stop()


if __name__ == "__main__":
    main()
