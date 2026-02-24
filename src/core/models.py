"""
Model loading and management for HAI-DEF models.

Centralises model lifecycle so each model is loaded exactly once
and shared across agents.
"""

from __future__ import annotations

import logging
import torch
from typing import Any, Optional

log = logging.getLogger(__name__)


class ModelManager:
    """Singleton-ish registry that lazily loads and caches HAI-DEF models."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._processors: dict[str, Any] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"ModelManager initialised  |  device={self._device}")

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # MedGemma (multimodal or text-only)
    # ------------------------------------------------------------------

    def load_medgemma(
        self,
        model_id: str = "google/medgemma-4b-it",
        quantize: bool = False,
    ) -> tuple[Any, Any]:
        """Load a MedGemma model + processor.  Returns (model, processor)."""
        if model_id in self._models:
            return self._models[model_id], self._processors[model_id]

        from transformers import AutoProcessor, AutoModelForImageTextToText

        log.info(f"Loading {model_id}  (quantize={quantize}) ...")

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        if quantize:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                log.info("  -> 4-bit quantisation enabled")
            except ImportError:
                log.warning("bitsandbytes not available -- loading without quantisation")

        # Try multimodal first, fall back to text-only CausalLM
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        processor = AutoProcessor.from_pretrained(model_id)

        self._models[model_id] = model
        self._processors[model_id] = processor
        log.info(f"  -> {model_id} loaded successfully")
        return model, processor

    # ------------------------------------------------------------------
    # MedASR
    # ------------------------------------------------------------------

    def load_medasr(self, model_id: str = "google/medasr") -> Any:
        """Load the MedASR ASR pipeline.  Returns the HF pipeline object."""
        if model_id in self._models:
            return self._models[model_id]

        log.info(f"Loading {model_id} ...")
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline("automatic-speech-recognition", model=model_id)
            self._models[model_id] = pipe
            log.info(f"  -> {model_id} loaded successfully")
            return pipe
        except Exception as exc:
            log.error(f"Failed to load MedASR: {exc}")
            log.info("MedASR requires transformers >= 5.0.0; "
                     "falling back to mock transcription for demo.")
            self._models[model_id] = None
            return None

    # ------------------------------------------------------------------
    # MedSigLIP
    # ------------------------------------------------------------------

    def load_medsiglip(self, model_id: str = "google/medsiglip-448") -> tuple[Any, Any]:
        """Load MedSigLIP for image embedding/classification."""
        if model_id in self._models:
            return self._models[model_id], self._processors[model_id]

        log.info(f"Loading {model_id} ...")
        try:
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
            processor = AutoProcessor.from_pretrained(model_id)
            model = model.to(self._device)
            self._models[model_id] = model
            self._processors[model_id] = processor
            log.info(f"  -> {model_id} loaded successfully")
            return model, processor
        except Exception as exc:
            log.error(f"Failed to load MedSigLIP: {exc}")
            self._models[model_id] = None
            self._processors[model_id] = None
            return None, None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._models and self._models[model_id] is not None

    def unload(self, model_id: str) -> None:
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._processors:
            del self._processors[model_id]
        torch.cuda.empty_cache()
        log.info(f"Unloaded {model_id}")

    def unload_all(self) -> None:
        model_ids = list(self._models.keys())
        for mid in model_ids:
            self.unload(mid)


# Global singleton
model_manager = ModelManager()
