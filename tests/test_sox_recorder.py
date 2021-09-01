from pathlib import Path
from voice_recogntion.sox_recorder import SoxRecorder
from unittest.mock import patch, MagicMock


@patch("subprocess.Popen")
def test_sox_recording_subprocess(mock_popen: MagicMock):
    SoxRecorder(Path("tests/temp")).start_sox_subprocess()
    mock_popen.assert_called_once()
