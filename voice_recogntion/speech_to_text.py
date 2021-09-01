from pathlib import Path
from deepspeech import Model


class SpeechToText:
    def __init__(self, model_path: Path, scorer_path: Path) -> None:
        print(f"Loading model from {str(model_path)!r}")
        self.model = Model(model_path=str(model_path.resolve()))

        print(f"Loading scorer from {str(scorer_path)!r}")
        self.model.enableExternalScorer(scorer_path=str(scorer_path.resolve()))
