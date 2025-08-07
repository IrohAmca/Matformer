import logging
import time
from typing import Dict, List, Optional

import torch
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
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        try:
            self.model = Gemma3nForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
            logger.info(f"Using Gemma3nForConditionalGeneration for {model_name}")
        except Exception as e:
            logger.warning(f"Could not load with Gemma3nForConditionalGeneration: {e}")
            logger.info("Falling back to AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        logger.info(f"Model {model_name} loaded successfully on {self.device}")

    def generate_text(self, 
                 prompt: str, 
                 max_length: int = 100, 
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 do_sample: bool = True,
                 layers_to_skip: Optional[List[int]] = None) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif self.device == "cpu":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        if layers_to_skip and isinstance(self.model, Gemma3nForConditionalGeneration):
            logger.info(f"Generating with skipped layers using hooks: {layers_to_skip}")
            
            hook_handles = []
            
            def skip_layer_hook(module, args, output):
                logger.info(f"Skipping layer {module.__class__.__name__} via hook.")
                if isinstance(output, tuple):
                    return (args[0],) + (None,) * (len(output) - 1)
                return args[0]

            try:
                for layer_idx in layers_to_skip:
                    if 0 <= layer_idx < len(self.model.language_model.layers):
                        layer = self.model.language_model.layers[layer_idx]
                        handle = layer.register_forward_hook(skip_layer_hook)
                        hook_handles.append(handle)
                    else:
                        logger.warning(f"Layer index {layer_idx} is out of bounds. Skipping.")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            finally:
                for handle in hook_handles:
                    handle.remove()
                logger.info("Removed all layer skipping hooks.")

        else:
            if layers_to_skip:
                logger.warning("Layer skipping is only implemented for Gemma3nForConditionalGeneration.")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
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
            logger.warning(f"Model size {model_size} not found in config, using default.")
            return config.MODEL_CONFIG["default"].get("exclude_layers", [])
    
    async def __call__(self, request) -> GenerationResponse:
        start_time = time.time()

        try:
            if hasattr(request, 'json'):
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
                layers_to_skip=generation_request.layers_to_skip or self._get_size_layers(generation_request.model_size)
            )
            
            processing_time = time.time() - start_time
            
            return GenerationResponse(
                generated_text=generated_text,
                prompt=generation_request.prompt,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise e
    
    def health_check(self) -> Dict[str, str]:
        return {
            "status": "healthy",
            "model": self.model_name,
            "timestamp": str(time.time())
        }
