from voice_recogntion import __version__
from voice_recogntion.main import SoxRecorder
from unittest.mock import patch, MagicMock


def test_version():
    assert __version__ == "0.1.0"


@patch("subprocess.Popen")
def test_sox_recording_subprocess(mock_popen: MagicMock):
    SoxRecorder().start_sox_subprocess()
    mock_popen.assert_called_once()
