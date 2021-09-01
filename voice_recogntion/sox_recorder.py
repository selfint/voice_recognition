import subprocess
from pathlib import Path
from typing import Deque, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

SOX_RECORD_CMD = (
    "rec -r 16000 -c 1 -p | sox -p -b 16 {}/{} trim 0 {} : newfile : restart"
)


def _create_sox_record_cmd(
    record_dir: Path,
    audio_filename_template: str,
    duration: float,
) -> str:
    """Create SOX record command.

    Args:
        record_dir (Path): Path to directory for audio files
        audio_filename_template (str): Base template for file names, numbers will be
        inserted before the filetype suffix
        duration (float): Length (in seconds) of each audio file

    Returns:
        str: Sox command to continuosly generate audio files in a directory
    """

    return SOX_RECORD_CMD.format(record_dir, audio_filename_template, duration)


class SoxRecorder:
    def __init__(self, data_dir: Path, output_queue: Optional[Deque] = None) -> None:
        self._data_dir = data_dir
        self._output_queue = output_queue

        self._record_process: Optional[subprocess.Popen] = None
        self._watchdog: Optional[Observer] = None

    def start_sox_subprocess(self):
        """Start generating audio files in the data directory"""

        # make sure data dir exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

        cmd = _create_sox_record_cmd(
            record_dir=self._data_dir, audio_filename_template="audio.wav", duration=2
        )

        self._record_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop_sox_subprocess(self):
        """Stop the sox recording subprocess"""

        self._record_process.terminate()

    def start_watchdog(self):
        """
        Start the watchdog observer on the data dir, push created files to output queue
        """

        self._watchdog = Observer()
        self._watchdog.schedule(OnCreateHandler(self._output_queue), self._data_dir)
        self._watchdog.start()

    def stop_watchdog(self):
        """Stop watchdog execution"""

        self._watchdog.stop()
        self._watchdog.join()


class OnCreateHandler(FileSystemEventHandler):
    def __init__(self, output_queue: Deque) -> None:
        super().__init__()
        self._output_queue = output_queue

    def on_created(self, event: FileSystemEvent):
        """Add created file (ignore created directories) to the output queue

        Args:
            event (FileSystemEvent): On created event with new file
        """

        # only add files to the output queue
        if not event.is_directory:
            self._output_queue.append(Path(event.src_path))
