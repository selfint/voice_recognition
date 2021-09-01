import tempfile
import time
from collections import deque
from pathlib import Path

from voice_recogntion.sox_recorder import SoxRecorder


def main():
    audio_queue = deque()

    with tempfile.TemporaryDirectory() as audio_dir:
        audio_path = Path(audio_dir)

        sr = SoxRecorder(audio_path, audio_queue)

        sr.start_watchdog()
        sr.start_sox_subprocess()

        time.sleep(10)
        sr.stop_sox_subprocess()
        sr.stop_watchdog()

    print(list(audio_queue))


if __name__ == "__main__":
    main()
