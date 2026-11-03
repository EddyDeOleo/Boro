
"""Contracts for creating execution-scoped perception providers."""

from __future__ import annotations

from typing import Mapping, Protocol

from src.models.perception_config import PerceptionConfig
from src.perception.perception_provider import PerceptionProvider


TrackingBoxes = Mapping[str, Mapping[str, int]]


class PerceptionProviderFactory(Protocol):
    """Create a perception provider for one verification execution."""

    def create(
        self,
        config: PerceptionConfig,
        tracking_boxes: TrackingBoxes,
    ) -> PerceptionProvider:
        """Create a provider using execution-specific tracking boxes."""
        ...
