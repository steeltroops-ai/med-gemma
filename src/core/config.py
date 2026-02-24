"""MedScribe AI - Core configuration module."""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a single HAI-DEF model."""

    model_id: str
    display_name: str
    model_type: str  # "multimodal", "text", "asr", "embedding"
    quantize: bool = False
    max_new_tokens: int = 1024
    device_map: str = "auto"


@dataclass
class AppConfig:
    """Application-wide configuration."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    temp_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "tmp")

    # Model configurations
    medgemma_4b: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="google/medgemma-4b-it",
        display_name="MedGemma 4B IT",
        model_type="multimodal",
        max_new_tokens=1024,
    ))

    medgemma_27b_text: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="google/medgemma-27b-text-it",
        display_name="MedGemma 27B Text",
        model_type="text",
        quantize=True,  # 27B needs quantization on most hardware
        max_new_tokens=2048,
    ))

    medasr: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="google/medasr",
        display_name="MedASR",
        model_type="asr",
    ))

    medsiglip: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="google/medsiglip-448",
        display_name="MedSigLIP",
        model_type="embedding",
    ))

    # Server config
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # HF Token (from environment)
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    def __post_init__(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)


# Global config singleton
config = AppConfig()
