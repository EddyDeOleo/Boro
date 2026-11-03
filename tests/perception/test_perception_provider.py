
from pathlib import Path

from src.perception.perception_provider import PerceptionProvider


class FakePerceptionProvider:
    def perceive(
        self,
        video_path,
        context,
        config,
        regions=(),
    ):
        raise NotImplementedError


def test_accepts_structural_implementation():
    provider = FakePerceptionProvider()

    assert isinstance(provider, PerceptionProvider)


def test_rejects_object_without_perceive_method():
    assert not isinstance(object(), PerceptionProvider)


def test_protocol_exposes_perceive_method():
    assert hasattr(PerceptionProvider, "perceive")


def test_protocol_is_not_tied_to_a_concrete_provider():
    assert PerceptionProvider.__module__ == (
        "src.perception.perception_provider"
    )
