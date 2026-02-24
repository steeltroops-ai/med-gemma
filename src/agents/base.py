"""
Base agent abstraction for the MedScribe AI agentic pipeline.

Every HAI-DEF model is wrapped as an independent agent that
conforms to this interface, enabling orchestration, timing,
logging, and graceful degradation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.schemas import AgentResult


class BaseAgent(ABC):
    """Abstract base class for all clinical agents."""

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.logger = logging.getLogger(f"agent.{name}")
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(self) -> None:
        """Load the underlying ML model.  Called once at startup."""

    def initialize(self) -> None:
        """Public entry-point for model loading."""
        self.logger.info(f"Initialising agent '{self.name}' with model {self.model_id} ...")
        try:
            self._load_model()
            self._ready = True
            self.logger.info(f"Agent '{self.name}' ready.")
        except Exception as exc:
            self.logger.error(f"Agent '{self.name}' failed to initialise: {exc}")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def _process(self, input_data: Any) -> Any:
        """Core processing logic -- override in subclass."""

    async def execute(self, input_data: Any) -> AgentResult:
        """Run the agent with timing and error handling.

        If the model is not loaded, _process() is still called so that
        agents can return demo/fallback data instead of hard-failing.
        """
        start = time.perf_counter()
        try:
            if not self._ready:
                self.logger.warning(
                    f"Agent '{self.name}' not initialised -- running in fallback/demo mode."
                )
            result = self._process(input_data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result,
                processing_time_ms=round(elapsed_ms, 1),
                model_used=self.model_id,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.logger.error(f"Agent '{self.name}' execution error: {exc}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=str(exc),
                processing_time_ms=round(elapsed_ms, 1),
                model_used=self.model_id,
            )
