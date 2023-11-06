import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from language_model.model import LanguageModel
from utils.logger import get_pylogger

log = get_pylogger(__name__)


class LLama(LanguageModel):
    r"""Class for generation answers for bot using LLM from huggingface repos."""

    def __init__(
        self,
        hf_token: str,
        model_name: str,
        use_8_bit: bool = False,
        use_4_bit: bool = False,
        use_flash_attention_2: bool = False,
        adapter: str = "",
        tokenizer_name: str = "",
    ):
        r"""
        Init class for generation answers for bot
        Args:
            hf_token: hugging face token using for download model
            model_name: LLM model name
            use_8_bit: load model in quantization mode
            use_flash_attention_2: memory efficient attention
            adapter: add adapter for base model
        """
        login(hf_token)

        tokenizer_name = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        kwargs = {}

        if use_flash_attention_2:
            kwargs["use_flash_attention_2"] = True

        if use_8_bit:
            kwargs["load_in_8bit"] = True

        if use_4_bit:
            kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", **kwargs
        )

        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)

        self.generation_config = GenerationConfig.from_pretrained(model_name)

    def get_tokens_as_tuple(self, word: str):
        r"""
        Return token ids
        Args:
            word: telegram id
        """
        return tuple(self.tokenizer([word], add_special_tokens=False).input_ids[0])

    def generate(self, context: str) -> str:
        r"""
        Return token ids
        Args:
            context: whole dialog using for generation answer
        """

        with torch.no_grad():
            data = self.tokenizer(context, return_tensors="pt")
            data = {k: v.to(self.model.device) for k, v in data.items()}
            output_ids = self.model.generate(**data, generation_config=self.generation_config)
            output_ids = [
                self.tokenizer.decode(
                    o[len(data["input_ids"][0]) :], skip_special_tokens=True
                ).strip()
                for o in output_ids
            ]
            outputs = output_ids
            log.debug("Outputs %s", outputs)
            return outputs[0].replace("</s>", "").replace("<s>", "")
