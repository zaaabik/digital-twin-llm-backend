"""Root file of service running REST api."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

from language_model.LLama import LLama
from language_model.model import LanguageModel
from utils.logger import get_pylogger

log = get_pylogger(__name__)
logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]
USE_8_BIT = os.getenv("USE_8_BIT", "false") == "true"
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "false") == "true"
ADAPTER = os.getenv("ADAPTER", "")
print(f"USE_8_BIT: {USE_8_BIT}")
print(f"ADAPTER: {ADAPTER}")
print(f"USE_FLASH_ATTENTION: {USE_FLASH_ATTENTION}")

log.info("Building LM model")
lm: LanguageModel = LLama(
    hf_token=HF_TOKEN,
    model_name=MODEL_NAME,
    use_8_bit=USE_8_BIT,
    use_flash_attention_2=USE_FLASH_ATTENTION,
    adapter=ADAPTER,
)

app = FastAPI()


@app.get("/ping")
def ping():
    r"""Health check route."""
    return {"ping": "pong"}


class GenerateRequest(BaseModel):
    text: str = ""


@app.post("/generate")
def generate(generate_request: GenerateRequest):
    r"""
    Update state of user, run language model and return response
    Args:
        generate_request: text using for generation
    """
    model_response = lm.generate(generate_request.text)
    return {"generated_text": model_response}
