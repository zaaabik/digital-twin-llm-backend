import os

import kfserving

from language_model.LLama import LLama
from language_model.model import LanguageModel

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]
USE_8_BIT = os.getenv("USE_8_BIT", "false") == "true"
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "false") == "true"
ADAPTER = os.getenv("ADAPTER", "")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "")
print(f"USE_8_BIT: {USE_8_BIT}")
print(f"ADAPTER: {ADAPTER}")
print(f"USE_FLASH_ATTENTION: {USE_FLASH_ATTENTION}")
print(f"TOKENIZER_NAME: {TOKENIZER_NAME}")


class KFServingExplainModel(kfserving.KFModel):
    r"""Class for handling request to generate text with llm."""

    def __init__(self, name: str):
        r"""
        Args:
            name: server name
        """
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = True

    def load(self):
        r"""Load model instance."""
        self.model: LanguageModel = LLama(
            hf_token=HF_TOKEN,
            model_name=MODEL_NAME,
            use_8_bit=USE_8_BIT,
            use_flash_attention_2=USE_FLASH_ATTENTION,
            adapter=ADAPTER,
            tokenizer_name=TOKENIZER_NAME,
        )
        self.ready = True

    def predict(self, request: dict) -> dict:
        r"""
        Return generated text using LLM model
        Args:
            request: data for prediction
        """
        # pylint: disable=invalid-overridden-method
        text = request["instances"][0]["text"]
        if not self.model:
            return {"predictions": ""}
        model_response = self.model.generate(text)
        return {"predictions": model_response}


if __name__ == "__main__":
    default_name = "kfserving-default"

    model = KFServingExplainModel(default_name)
    model.load()
    kfserving.KFServer(workers=1).start([model])
