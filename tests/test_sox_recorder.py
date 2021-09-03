import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

from voice_recognition.sox_recorder import SoxRecorder


@patch("subprocess.Popen")
def test_start_sox_subprocess(mock_popen: MagicMock):
    SoxRecorder(Path("tests/temp")).start_sox_subprocess()
    mock_popen.assert_called_once()


@patch("subprocess.Popen")
def test_stop_sox_subprocess(mock_popen: MagicMock):
    mock_process = MagicMock(subprocess.Popen)
    mock_terminate = MagicMock()
    mock_process.terminate = mock_terminate
    mock_popen.return_value = mock_process

    sr = SoxRecorder(Path("tests/temp"))
    sr.start_sox_subprocess()
    sr.stop_sox_subprocess()
    mock_terminate.assert_called_once()


def test_sox_recorder_watchdog():
    with tempfile.TemporaryDirectory() as tempdir:
        data_dir = Path(tempdir) / "data"
        data_dir.mkdir()
        output_queue = deque()

        sr = SoxRecorder(data_dir, output_queue)
        sr.start_watchdog()

        newfile = data_dir / "file1.wav"
        newfile.touch()
        time.sleep(0.01)
        assert list(output_queue) == [newfile]

        newfile2 = data_dir / "file2.wav"
        newfile2.touch()
        time.sleep(0.01)
        assert list(output_queue) == [newfile, newfile2]
