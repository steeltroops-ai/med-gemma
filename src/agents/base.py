"""
Base agent abstraction for the MedScribe AI agentic pipeline.

Every HAI-DEF model is wrapped as an independent agent that conforms
to this interface. The BaseAgent provides:

  - Standardised lifecycle management (initialize, execute)
  - Automatic execution timing and structured telemetry
  - Error boundary isolation (agent failure never cascades)
  - Execution counters for observability (runs, failures, total time)
  - Graceful degradation (fallback/demo mode when model unavailable)

Architecture reference: docs/ARCHITECTURE.md Section 4
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.schemas import AgentResult


class BaseAgent(ABC):
    """Abstract base class for all clinical agents.

    Every agent in the pipeline extends this class. The orchestrator
    interacts exclusively through the public interface:

        agent.initialize()       -> prepare the model / resources
        agent.execute(input)     -> AgentResult with timing + error info
        agent.is_ready           -> whether the agent initialized OK
        agent.telemetry          -> cumulative execution statistics

    Subclasses implement:
        _load_model()  -> one-time initialization (model loading, config)
        _process(data) -> core inference / processing logic
    """

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.logger = logging.getLogger(f"agent.{name}")
        self._ready = False

        # --- Telemetry counters ---
        self._execution_count: int = 0
        self._failure_count: int = 0
        self._total_time_ms: float = 0.0
        self._last_execution_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(self) -> None:
        """Load the underlying ML model.  Called once at startup."""

    def initialize(self) -> None:
        """Public entry-point for model loading."""
        self.logger.info(
            f"Initialising agent '{self.name}' with model {self.model_id} ..."
        )
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
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> dict:
        """Cumulative execution statistics for observability."""
        return {
            "agent_name": self.name,
            "model_id": self.model_id,
            "is_ready": self._ready,
            "execution_count": self._execution_count,
            "failure_count": self._failure_count,
            "success_rate": (
                round(1 - self._failure_count / self._execution_count, 4)
                if self._execution_count > 0
                else None
            ),
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_time_ms": (
                round(self._total_time_ms / self._execution_count, 1)
                if self._execution_count > 0
                else None
            ),
            "last_execution_ms": round(self._last_execution_ms, 1),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def _process(self, input_data: Any) -> Any:
        """Core processing logic -- override in subclass."""

    async def execute(self, input_data: Any) -> AgentResult:
        """Run the agent with timing, telemetry, and error isolation.

        If the model is not loaded, _process() is still called so that
        agents can return demo/fallback data instead of hard-failing.

        Returns an AgentResult envelope containing:
          - success status
          - output data (or None on failure)
          - error message (if failed)
          - processing time in milliseconds
          - model identifier
        """
        self._execution_count += 1
        start = time.perf_counter()

        try:
            if not self._ready:
                self.logger.warning(
                    f"Agent '{self.name}' not initialised "
                    "-- running in fallback/demo mode."
                )
            result = await asyncio.to_thread(self._process, input_data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._total_time_ms += elapsed_ms
            self._last_execution_ms = elapsed_ms
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result,
                processing_time_ms=round(elapsed_ms, 1),
                model_used=self.model_id,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._total_time_ms += elapsed_ms
            self._last_execution_ms = elapsed_ms
            self._failure_count += 1
            self.logger.error(f"Agent '{self.name}' execution error: {exc}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=str(exc),
                processing_time_ms=round(elapsed_ms, 1),
                model_used=self.model_id,
            )
