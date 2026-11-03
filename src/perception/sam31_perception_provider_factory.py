
"""Factory for execution-scoped SAM 3.1 perception providers."""

from __future__ import annotations

from collections.abc import Mapping

from src.models.perception_config import PerceptionConfig
from src.perception.meta_sam31_client import MetaSAM31Client
from src.perception.meta_sam31_inference import (
    MetaSAM31Inference,
    SAM31BoxPrompt,
)
from src.perception.perception_provider import PerceptionProvider
from src.perception.perception_provider_factory import (
    TrackingBoxes,
)
from src.perception.sam31_perception_provider import (
    Sam31PerceptionProvider,
)


class SAM31PerceptionProviderFactory:
    """Create an independent SAM 3.1 provider for each execution."""

    def __init__(self, client: MetaSAM31Client) -> None:
        if not isinstance(client, MetaSAM31Client):
            raise TypeError(
                "client must be a MetaSAM31Client."
            )

        self.client = client

    def create(
        self,
        config: PerceptionConfig,
        tracking_boxes: TrackingBoxes,
    ) -> PerceptionProvider:
        """Build a provider using the supplied entity bounding boxes."""

        if not isinstance(config, PerceptionConfig):
            raise TypeError(
                "config must be a PerceptionConfig."
            )

        if config.provider != "meta":
            raise ValueError(
                "SAM 3.1 requires the 'meta' perception provider."
            )

        if config.model != "sam-3.1":
            raise ValueError(
                "SAM 3.1 factory requires model 'sam-3.1'."
            )

        if not isinstance(tracking_boxes, Mapping):
            raise TypeError(
                "tracking_boxes must be a mapping."
            )

        configured_ids = set(config.entity_id_map.keys())
        tracking_ids = set(tracking_boxes.keys())

        if tracking_ids != configured_ids:
            raise ValueError(
                "Tracking box IDs must match the configured provider "
                "entity IDs."
            )

        initial_boxes: dict[str, SAM31BoxPrompt] = {}

        for provider_id, coordinates in tracking_boxes.items():
            if not isinstance(provider_id, str):
                raise TypeError(
                    "Tracking box provider IDs must be strings."
                )

            if not isinstance(coordinates, Mapping):
                raise TypeError(
                    f"Tracking box {provider_id!r} must be a mapping."
                )

            expected_keys = {"x1", "y1", "x2", "y2"}

            if set(coordinates.keys()) != expected_keys:
                raise ValueError(
                    f"Tracking box {provider_id!r} must contain "
                    "exactly x1, y1, x2, and y2."
                )

            initial_boxes[provider_id] = SAM31BoxPrompt(
                x1=coordinates["x1"],
                y1=coordinates["y1"],
                x2=coordinates["x2"],
                y2=coordinates["y2"],
            )

        inference = MetaSAM31Inference(
            client=self.client,
            initial_boxes=initial_boxes,
        )

        return Sam31PerceptionProvider(
            inference=inference,
        )
