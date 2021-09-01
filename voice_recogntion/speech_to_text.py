from pathlib import Path
from deepspeech import Model


class SpeechToText:
    def __init__(self, model_path: Path) -> None:
        model_path_str = model_path.resolve().as_posix()
        print(f"Loading model from {model_path_str!r}")
        self.model = Model(model_path=model_path_str)
