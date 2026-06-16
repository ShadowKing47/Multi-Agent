import os
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalLLMClient:
    """
    Wraps a HuggingFace causal LM for local inference on Apple Silicon (MPS).
    Optionally merges a LoRA adapter (e.g. poet-sft-lora) before serving.

    float32 is forced — bf16/fp16 produce NaN losses and unstable logits on MPS.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        device: str = "mps",
    ):
        logger.info(f"Loading '{model_id}' on {device} — this takes ~1-2 min for 8B...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.to(device)
        self.model.eval()

        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            logger.info(f"LoRA adapter merged from '{adapter_path}'")

        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        logger.info("Local model ready.")

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int = 1024,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        result = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # pipeline appends the assistant turn as the last message dict
        return result[0]["generated_text"][-1]["content"].strip()
