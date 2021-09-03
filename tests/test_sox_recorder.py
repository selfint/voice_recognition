import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

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
        assert list(output_queue) == []

        newfile2 = data_dir / "file2.wav"
        newfile2.touch()
        time.sleep(0.01)
        assert list(output_queue) == [newfile]

        newfile3 = data_dir / "file3.wav"
        newfile3.touch()
        time.sleep(0.01)
        assert list(output_queue) == [newfile, newfile2]


@patch("subprocess.check_output")
def test_record_once(mock_check_output: MagicMock):
    # return one second of silence
    zeroes = np.zeros(16000, dtype=np.int16)
    mock_check_output.return_value = zeroes.tobytes()

    sr = SoxRecorder()

    audio_buffer = sr.record_once()

    np.testing.assert_array_equal(audio_buffer, zeroes)
