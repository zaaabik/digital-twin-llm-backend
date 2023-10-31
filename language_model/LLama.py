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
        tokenizer_name: str,
        use_8_bit: bool = False,
        use_flash_attention_2: bool = False,
        adapter: str = "",
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

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_flash_attention_2=use_flash_attention_2,
            load_in_8bit=use_8_bit,
        )

        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)

        self.generation_config = GenerationConfig(
            bos_token_id=1,
            eos_token_id=2,
            max_new_tokens=256,
            no_repeat_ngram_size=25,
            num_beams=3,
            num_return_sequences=1,
            pad_token_id=0,
            repetition_penalty=1.2,
            transformers_version="4.34.0",
            min_new_tokens=2,
        )

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
