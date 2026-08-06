"""Quantization and PEFT helpers for Falcon QLoRA."""

from __future__ import annotations

from typing import Any

from sql_model.config import FalconLoraConfig, FalconQuantizationConfig


def resolve_torch_dtype(dtype_name: str):
    """Resolve a torch dtype from configuration text."""
    import torch

    dtype_by_name = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_by_name[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def build_bitsandbytes_config(config: FalconQuantizationConfig):
    """Build a Transformers BitsAndBytesConfig for 4-bit QLoRA."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.bnb_4bit_compute_dtype),
    )


def build_lora_config(config: FalconLoraConfig):
    """Build a PEFT LoRA config targeting Falcon projection modules."""
    from peft import LoraConfig

    return LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=config.task_type,
    )


def prepare_model_for_qlora(model: Any):
    """Prepare a loaded causal LM for k-bit PEFT training."""
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(model)


def attach_lora_adapter(model: Any, config: FalconLoraConfig):
    """Attach trainable LoRA adapters to a Falcon model."""
    from peft import get_peft_model

    return get_peft_model(model, build_lora_config(config))
