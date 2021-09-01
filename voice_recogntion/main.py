from pathlib import Path
import subprocess
from typing import Optional

SOX_RECORD_CMD = (
    "rec -r 16000 -c 1 -p | sox -p -b 16 {}/{} trim 0 {} : newfile : restart"
)
DATA_DIR = Path("data")


def _create_sox_record_cmd(
    record_dir: Path,
    audio_filename_template: str,
    duration: int,
) -> str:
    """Create SOX record command.

    Args:
        record_dir (Path): Path to directory for audio files
        audio_filename_template (str): Base template for file names, numbers will be
                                       inserted before the filetype suffix
        duration (int): Length (in seconds) of each audio file

    Returns:
        str: Sox command to continuosly generate audio files in a directory
    """


class SoxRecorder:
    def __init__(self) -> None:
        self._record_process: Optional[subprocess.Popen] = None

    def start_sox_subprocess(self):
        """Start generating audio files in the DATA directory"""

        # make sure DATA dir exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        cmd = _create_sox_record_cmd(
            record_dir=DATA_DIR, audio_filename_template="audio.wav", duration=2
        )

        self._record_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
