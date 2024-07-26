# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from TTS.api import TTS


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def predict(self, text: str = Input(description="Text to prefix with 'hello '")) -> Path:
        """Run a single prediction on the model"""
        wav = self.tts.tts(text=text, speaker_wav="audio.mp3", language="ru")
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        return wav
