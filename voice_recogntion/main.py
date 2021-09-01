import tempfile
import time
from collections import deque
from pathlib import Path

from voice_recogntion.sox_recorder import SoxRecorder


def main():
    audio_queue = deque()

    with tempfile.TemporaryDirectory() as audio_dir:
        sr = SoxRecorder(Path(audio_dir), audio_queue)

        sr.start_watchdog()
        sr.start_sox_subprocess()

        time.sleep(10)
        sr.stop_sox_subprocess()
        sr.stop_watchdog()

    print(list(audio_queue))


if __name__ == "__main__":
    main()
