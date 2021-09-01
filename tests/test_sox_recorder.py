from pathlib import Path
import subprocess
from voice_recogntion.sox_recorder import SoxRecorder
from unittest.mock import patch, MagicMock


@patch("subprocess.Popen")
def test_sox_recording_subprocess(mock_popen: MagicMock):
    SoxRecorder(Path("tests/temp")).start_sox_subprocess()
    mock_popen.assert_called_once()


@patch("subprocess.Popen")
def test_stop_recording(mock_popen: MagicMock):
    mock_process = MagicMock(subprocess.Popen)
    mock_terminate = MagicMock()
    mock_process.terminate = mock_terminate
    mock_popen.return_value = mock_process

    sr = SoxRecorder(Path("tests/temp"))
    sr.start_sox_subprocess()
    sr.stop_sox_subprocess()
    mock_terminate.assert_called_once()
