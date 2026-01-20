import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3nForConditionalGeneration,
)

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    prompt: str
    model_size: str = "default"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_cache: bool = False
    layers_to_skip: Optional[List[int]] = None


class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    model_name: str
    processing_time: float


class GemmaE2B:
    def __init__(self, model_name: str = "google/gemma-3n-e4b-it"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model: {model_name} on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = (
            torch.bfloat16
            if (
                torch.cuda.is_available()
                and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            else torch.float16
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        try:
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs
            )
            logger.info(f"Using Gemma3nForConditionalGeneration for {model_name}")
        except Exception as e:
            logger.warning(f"Could not load with Gemma3nForConditionalGeneration: {e}")
            logger.info("Falling back to AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        cfg = getattr(self.model.config, "text_config", self.model.config)
        num_layers = getattr(cfg, "num_hidden_layers", None)
        num_kv_shared = getattr(cfg, "num_kv_shared_layers", 0)
        if num_layers is not None:
            num_kv_comp_layers = num_layers - num_kv_shared
            reserved = [num_kv_comp_layers - 2, num_kv_comp_layers - 1]
            logger.info(
                f"Model loaded. num_hidden_layers={num_layers}, num_kv_shared_layers={num_kv_shared}, reserved_skip_layers={reserved}"
            )

        logger.info(f"Model {model_name} loaded successfully on {self.device}")

    class _SkipDecoderLayer(nn.Module):
        def __init__(self, attention_type: str, layer_idx: int):
            super().__init__()
            self.attention_type = attention_type
            self.layer_idx = layer_idx

        def forward(self, *args, **kwargs):
            if not args:
                raise RuntimeError(
                    "_SkipDecoderLayer expects at least one positional arg"
                )
            hidden_like = args[0]
            output_attentions = kwargs.get("output_attentions", False)
            outputs = (hidden_like,)
            if output_attentions:
                outputs += (None,)
            return outputs

    @contextmanager
    def _skip_layers_context(self, layers_to_skip: List[int]):
        if (
            not isinstance(self.model, Gemma3nForConditionalGeneration)
            or not layers_to_skip
        ):
            yield
            return
        original_layers = {}
        try:
            lm = self.model.language_model
            for idx in layers_to_skip:
                orig = lm.layers[idx]
                original_layers[idx] = orig
                lm.layers[idx] = self._SkipDecoderLayer(
                    attention_type=getattr(orig, "attention_type", "full_attention"),
                    layer_idx=getattr(orig, "layer_idx", idx),
                )
            yield
        finally:
            lm = self.model.language_model
            for idx, module in original_layers.items():
                lm.layers[idx] = module

    def _sanitize_layers_to_skip(
        self, layers_to_skip: Optional[List[int]]
    ) -> List[int]:
        if not layers_to_skip:
            return []

        cfg = getattr(self.model.config, "text_config", self.model.config)
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        num_kv_shared = int(getattr(cfg, "num_kv_shared_layers", 0))
        num_kv_comp_layers = num_layers - num_kv_shared
        local_kv_sharing_layer_idx = num_kv_comp_layers - 2
        global_kv_sharing_layer_idx = num_kv_comp_layers - 1
        reserved = {local_kv_sharing_layer_idx, global_kv_sharing_layer_idx}

        sanitized = []
        for i in set(layers_to_skip):
            if i in reserved:
                logger.warning(
                    f"Katman {i} KV sharing için ayrılmıştır ve atlanamaz. Listeden çıkarılıyor."
                )
                continue
            if i < 0 or i >= num_layers:
                logger.warning(
                    f"Katman {i} geçersiz aralıkta (0..{num_layers - 1}). Listeden çıkarılıyor."
                )
                continue
            sanitized.append(i)

        sanitized = sorted(sanitized)
        if sanitized != sorted(set(layers_to_skip)):
            logger.info(f"Temizlenmiş atlanacak katmanlar: {sanitized}")
        return sanitized

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_cache: bool = False,
        layers_to_skip: Optional[List[int]] = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif self.device == "cpu":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

        if layers_to_skip and isinstance(self.model, Gemma3nForConditionalGeneration):
            layers_to_skip = self._sanitize_layers_to_skip(layers_to_skip)
            logger.info(f"Generating with pre-swap skipped layers: {layers_to_skip}")

            with self._skip_layers_context(layers_to_skip):
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        use_cache=use_cache,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
        else:
            if layers_to_skip:
                logger.warning(
                    "Layer skipping is only implemented for Gemma3nForConditionalGeneration."
                )
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


@serve.deployment(
    ray_actor_options={"num_cpus": 8, "num_gpus": 1},
    graceful_shutdown_timeout_s=300,
    health_check_period_s=60,
    health_check_timeout_s=30,
)
class GemmaServeDeployment:
    def __init__(self, model_name: str = "google/gemma-3n-e4b-it"):
        self.model = GemmaE2B(model_name)
        self.model_name = model_name
        logger.info(f"GemmaServeDeployment initialized with {model_name}")

    def _get_size_layers(self, model_size: str) -> List[int]:
        if model_size in config.MODEL_CONFIG:
            return config.MODEL_CONFIG[model_size].get("exclude_layers", [])
        else:
            logger.warning(
                f"Model size {model_size} not found in config, using default."
            )
            return config.MODEL_CONFIG["default"].get("exclude_layers", [])

    async def __call__(self, request) -> GenerationResponse:
        start_time = time.time()

        try:
            if hasattr(request, "json"):
                request_data = await request.json()
            else:
                request_data = request

            generation_request = GenerationRequest(**request_data)

            generated_text = self.model.generate_text(
                prompt=generation_request.prompt,
                max_length=generation_request.max_length,
                temperature=generation_request.temperature,
                top_p=generation_request.top_p,
                top_k=generation_request.top_k,
                do_sample=generation_request.do_sample,
                use_cache=generation_request.use_cache,
                layers_to_skip=generation_request.layers_to_skip
                or self._get_size_layers(generation_request.model_size),
            )

            processing_time = time.time() - start_time

            return GenerationResponse(
                generated_text=generated_text,
                prompt=generation_request.prompt,
                model_name=self.model_name,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise e

    def health_check(self) -> Dict[str, str]:
        return {
            "status": "healthy",
            "model": self.model_name,
            "timestamp": str(time.time()),
        }
