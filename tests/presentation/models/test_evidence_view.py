

import pytest

from src.presentation.models.evidence_view import EvidenceView


@pytest.fixture
def valid_evidence_view() -> EvidenceView:
    return EvidenceView(
        evidence_id="evidence_001",
        evidence_type="STABLE_IN_REGION",
        description="The object remained stable inside the target region.",
        frame_start=10,
        frame_end=30,
        timestamp_start_seconds=0.333333,
        timestamp_end_seconds=1.0,
        observation_ids=("observation_001", "observation_002"),
        entity_ids=("entity_001",),
        metrics={
            "mean_speed": 0.12,
            "observation_count": 21,
            "region_overlap": 0.95,
            "is_stable": True,
            "note": "Stable placement detected",
            "optional_value": None,
        },
        limitations=("Visual evidence only.",),
    )


class TestEvidenceView:
    def test_creates_valid_evidence_view(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        assert valid_evidence_view.evidence_id == "evidence_001"
        assert valid_evidence_view.evidence_type == "STABLE_IN_REGION"
        assert valid_evidence_view.frame_start == 10
        assert valid_evidence_view.frame_end == 30
        assert valid_evidence_view.timestamp_start_seconds == 0.333333
        assert valid_evidence_view.timestamp_end_seconds == 1.0

    def test_allows_empty_observation_and_entity_references(self) -> None:
        evidence = EvidenceView(
            evidence_id="evidence_002",
            evidence_type="VISUAL_OBSERVATION",
            description="Evidence without linked identifiers.",
            frame_start=0,
            frame_end=0,
            timestamp_start_seconds=0.0,
            timestamp_end_seconds=0.0,
        )

        assert evidence.observation_ids == ()
        assert evidence.entity_ids == ()

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("evidence_id", ""),
            ("evidence_id", "   "),
            ("evidence_type", ""),
            ("evidence_type", "   "),
            ("description", ""),
            ("description", "   "),
        ],
    )
    def test_rejects_empty_required_strings(
        self,
        valid_evidence_view: EvidenceView,
        field_name: str,
        value: str,
    ) -> None:
        with pytest.raises(ValueError):
            self._replace(valid_evidence_view, **{field_name: value})

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("frame_start", -1),
            ("frame_end", -1),
            ("frame_start", True),
            ("frame_end", False),
            ("frame_start", 1.5),
            ("frame_end", "10"),
        ],
    )
    def test_rejects_invalid_frame_indices(
        self,
        valid_evidence_view: EvidenceView,
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError):
            self._replace(valid_evidence_view, **{field_name: value})

    def test_rejects_frame_start_after_frame_end(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(ValueError, match="frame_start"):
            self._replace(
                valid_evidence_view,
                frame_start=31,
                frame_end=30,
            )

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("timestamp_start_seconds", -1.0),
            ("timestamp_end_seconds", -1.0),
            ("timestamp_start_seconds", float("inf")),
            ("timestamp_end_seconds", float("-inf")),
            ("timestamp_start_seconds", float("nan")),
            ("timestamp_end_seconds", float("nan")),
            ("timestamp_start_seconds", True),
            ("timestamp_end_seconds", False),
            ("timestamp_start_seconds", "0.5"),
            ("timestamp_end_seconds", "1.0"),
        ],
    )
    def test_rejects_invalid_timestamps(
        self,
        valid_evidence_view: EvidenceView,
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError):
            self._replace(valid_evidence_view, **{field_name: value})

    def test_rejects_timestamp_start_after_timestamp_end(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(ValueError, match="timestamp_start_seconds"):
            self._replace(
                valid_evidence_view,
                timestamp_start_seconds=2.0,
                timestamp_end_seconds=1.0,
            )

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("observation_ids", ["observation_001"]),
            ("observation_ids", ("",)),
            ("observation_ids", ("   ",)),
            ("observation_ids", ("observation_001", "observation_001")),
            ("entity_ids", ["entity_001"]),
            ("entity_ids", ("",)),
            ("entity_ids", ("   ",)),
            ("entity_ids", ("entity_001", "entity_001")),
        ],
    )
    def test_rejects_invalid_identifier_tuples(
        self,
        valid_evidence_view: EvidenceView,
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError):
            self._replace(valid_evidence_view, **{field_name: value})

    def test_rejects_non_tuple_limitations(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(ValueError, match="limitations must be a tuple"):
            self._replace(
                valid_evidence_view,
                limitations=["Visual evidence only."],
            )

    @pytest.mark.parametrize(
        "limitations",
        [
            ("",),
            ("   ",),
            ("Visual evidence only.", "Visual evidence only."),
        ],
    )
    def test_rejects_invalid_limitations(
        self,
        valid_evidence_view: EvidenceView,
        limitations: tuple[str, ...],
    ) -> None:
        with pytest.raises(ValueError):
            self._replace(
                valid_evidence_view,
                limitations=limitations,
            )

    def test_rejects_non_mapping_metrics(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(ValueError, match="metrics must be a mapping"):
            self._replace(
                valid_evidence_view,
                metrics=[("speed", 0.1)],
            )

    def test_rejects_empty_metric_keys(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(ValueError, match="metric key"):
            self._replace(
                valid_evidence_view,
                metrics={"": 0.1},
            )

    @pytest.mark.parametrize(
        "value",
        [
            [],
            {},
            object(),
            (1, 2),
        ],
    )
    def test_rejects_unsupported_metric_values(
        self,
        valid_evidence_view: EvidenceView,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported metric value"):
            self._replace(
                valid_evidence_view,
                metrics={"unsupported": value},
            )

    @pytest.mark.parametrize(
        "value",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
    )
    def test_rejects_non_finite_metric_values(
        self,
        valid_evidence_view: EvidenceView,
        value: float,
    ) -> None:
        with pytest.raises(ValueError, match="finite float"):
            self._replace(
                valid_evidence_view,
                metrics={"invalid_metric": value},
            )

    def test_freezes_metrics_mapping(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        with pytest.raises(TypeError):
            valid_evidence_view.metrics["new_metric"] = 1.0

    def test_copies_metrics_before_freezing(self) -> None:
        source_metrics = {"speed": 0.15}

        evidence = EvidenceView(
            evidence_id="evidence_003",
            evidence_type="MOTION",
            description="Object movement evidence.",
            frame_start=0,
            frame_end=10,
            timestamp_start_seconds=0.0,
            timestamp_end_seconds=0.333333,
            metrics=source_metrics,
        )

        source_metrics["speed"] = 0.99

        assert evidence.metrics["speed"] == 0.15

    def test_preserves_metric_value_types(
        self,
        valid_evidence_view: EvidenceView,
    ) -> None:
        assert valid_evidence_view.metrics["mean_speed"] == 0.12
        assert valid_evidence_view.metrics["observation_count"] == 21
        assert valid_evidence_view.metrics["is_stable"] is True
        assert valid_evidence_view.metrics["note"] == "Stable placement detected"
        assert valid_evidence_view.metrics["optional_value"] is None

    @staticmethod
    def _replace(
        evidence: EvidenceView,
        **changes: object,
    ) -> EvidenceView:
        values = {
            "evidence_id": evidence.evidence_id,
            "evidence_type": evidence.evidence_type,
            "description": evidence.description,
            "frame_start": evidence.frame_start,
            "frame_end": evidence.frame_end,
            "timestamp_start_seconds": evidence.timestamp_start_seconds,
            "timestamp_end_seconds": evidence.timestamp_end_seconds,
            "observation_ids": evidence.observation_ids,
            "entity_ids": evidence.entity_ids,
            "metrics": dict(evidence.metrics),
            "limitations": evidence.limitations,
        }
        values.update(changes)

        return EvidenceView(**values)
