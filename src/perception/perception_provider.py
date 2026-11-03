

"""Provider-independent interface for visual perception."""

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from src.models.observation_context import ObservationContext
from src.models.perception_config import PerceptionConfig
from src.models.perception_result import PerceptionResult
from src.models.region import RegionDefinition


@runtime_checkable
class PerceptionProvider(Protocol):
    """Contract for visual perception providers."""

    def perceive(
        self,
        video_path: Path,
        context: ObservationContext,
        config: PerceptionConfig,
        regions: Sequence[RegionDefinition] = (),
    ) -> PerceptionResult:
        """Analyze a video and return normalized perception data."""
        ...
