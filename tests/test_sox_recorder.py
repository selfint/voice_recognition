from pathlib import Path
import subprocess
from voice_recogntion.sox_recorder import SoxRecorder
from unittest.mock import patch, MagicMock
from pyfakefs.fake_filesystem import FakeFilesystem


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


def _mock_recording(data_dir: Path):
    """Generate empty files with a .wav extension every 2 seconds
    in the given ``data_dir``

    Args:
        data_dir (Path): Path to directory to generate files in
    """

    counter = 0
    while True:
        filename = f"mock_data{counter:05}.wav"
        (data_dir / filename).touch()


def test_fake_filesystem(fs: FakeFilesystem):
    assert fs is not None
