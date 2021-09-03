import subprocess
from pathlib import Path
from typing import Deque, Optional
import shlex
import numpy as np

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

SOX_RECORD_CMD = (
    "rec -r 16000 -c 1 -e signed-integer "
    "--endian little --compression 0.0 --no-dither -p "
    "| sox -p -b 16 {}/{} trim 0 {} : newfile : restart"
)

SOX_RECORD_ONCE_CMD = (
    "rec --type raw -r 16000 -c 1 -e signed-integer --endian little "
    "--compression 0.0 --no-dither - trim 0 {}"
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
        str: Sox command to continuously generate audio files in a directory
    """

    return SOX_RECORD_CMD.format(record_dir, audio_filename_template, duration)


def _create_sox_record_once_cmd(duration: float) -> str:
    """Create SOX record once command.

    Records the given duration and return the raw audio.

    Args:
        duration: Duration (in seconds) to record

    Returns:
        str: SOX command to record for the given duration and get raw audio data
    """

    return SOX_RECORD_ONCE_CMD.format(duration)



class SoxRecorder:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_queue: Optional[Deque] = None
    ) -> None:
        self._data_dir = data_dir
        self._output_queue = output_queue

        self._record_process: Optional[subprocess.Popen] = None
        self._watchdog: Optional[Observer] = None

    def start_sox_subprocess(self):
        """Start generating audio files in the data directory"""

        if self._data_dir is None:
            raise ValueError("Can't record since no data dir was given")

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

        if self._record_process is not None:
            self._record_process.terminate()

    def start_watchdog(self):
        """
        Start the watchdog observer on the data dir, push created files to output queue

        NOTE: will not start watchdog if ``_output_queue`` is None.
        """

        if self._output_queue is not None:
            self._watchdog = Observer()
            self._watchdog.schedule(OnCreateHandler(self._output_queue), self._data_dir)
            self._watchdog.start()

    def stop_watchdog(self):
        """Stop watchdog execution"""

        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog.join()

    def start(self):
        """Start all SoxRecorder activity

        Start generating audio files in the ``data_dir``, start a watchdog on
        that directory and push Path of created files to the ``output_queue``.
        """

        self.start_sox_subprocess()
        self.start_watchdog()

    def stop(self):
        """
        Stop all SoxRecorder activity

        Stop generating audio files in the ``data_dir`` and stop the watchdog
        on that directory.
        """

        self.stop_sox_subprocess()
        self.stop_watchdog()

    def record_once(self) -> np.ndarray:
        """Record audio once and return the audio data.

        Start a ``sox`` command in a subprocess, capture the raw audio result
        from standard out, and convert it to an ``np.ndarray`` object.

        Returns:
            np.ndarray: Audio buffer of recording
        """

        cmd = _create_sox_record_once_cmd(duration=2)

        try:
            output = subprocess.check_output(
                shlex.split(cmd),
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))

        return np.frombuffer(output, np.int16)



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
