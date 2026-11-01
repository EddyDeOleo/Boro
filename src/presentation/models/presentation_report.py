
"""Complete presentation model for robotic execution reports."""

from dataclasses import dataclass

from src.presentation.models.report_overview import ReportOverview
from src.presentation.models.criterion_view import CriterionView
from src.presentation.models.timeline_event_view import TimelineEventView
from src.presentation.models.evidence_view import EvidenceView


@dataclass(frozen=True)
class PresentationReport:
    """Presentation-ready representation of an execution report."""

    overview: ReportOverview

    criteria: tuple[CriterionView, ...]
    timeline: tuple[TimelineEventView, ...]
    evidence: tuple[EvidenceView, ...]

    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report structure and cross-references."""

        if not isinstance(self.overview, ReportOverview):
            raise ValueError("overview must be a ReportOverview.")

        self._validate_model_tuple(
            self.criteria,
            CriterionView,
            "criteria",
        )
        self._validate_model_tuple(
            self.timeline,
            TimelineEventView,
            "timeline",
        )
        self._validate_model_tuple(
            self.evidence,
            EvidenceView,
            "evidence",
        )

        if not isinstance(self.limitations, tuple):
            raise ValueError("limitations must be a tuple.")

        for limitation in self.limitations:
            if not isinstance(limitation, str) or not limitation.strip():
                raise ValueError(
                    "Every limitation must be a non-empty string."
                )

        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations cannot contain duplicates.")

        self._validate_unique_ids()
        self._validate_references()
        self._validate_timeline_order()
        self._validate_evidence_order()

    @staticmethod
    def _validate_model_tuple(
        items: tuple,
        expected_type: type,
        field_name: str,
    ) -> None:
        """Validate a tuple containing instances of an expected model."""

        if not isinstance(items, tuple):
            raise ValueError(f"{field_name} must be a tuple.")

        if not all(isinstance(item, expected_type) for item in items):
            raise ValueError(
                f"Every item in {field_name} must be a "
                f"{expected_type.__name__} instance."
            )

    def _validate_unique_ids(self) -> None:
        """Ensure identifiers are unique within each collection."""

        collections = (
            (
                "criteria",
                [item.criterion_id for item in self.criteria],
            ),
            (
                "timeline",
                [item.event_id for item in self.timeline],
            ),
            (
                "evidence",
                [item.evidence_id for item in self.evidence],
            ),
        )

        for collection_name, identifiers in collections:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(
                    f"{collection_name} must contain unique IDs."
                )

    def _validate_references(self) -> None:
        """Ensure criterion and event references exist."""

        event_ids = {
            event.event_id for event in self.timeline
        }
        evidence_ids = {
            item.evidence_id for item in self.evidence
        }

        event_evidence = {
            event.event_id: set(event.evidence_ids)
            for event in self.timeline
        }

        for criterion in self.criteria:
            missing_events = set(criterion.event_ids) - event_ids

            if missing_events:
                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references unknown events."
                )

            missing_evidence = (
                set(criterion.evidence_ids) - evidence_ids
            )

            if missing_evidence:
                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references unknown evidence."
                )

            associated_evidence = {
                evidence_id
                for event_id in criterion.event_ids
                for evidence_id in event_evidence[event_id]
            }

            if not set(criterion.evidence_ids).issubset(
                associated_evidence
            ):
                raise ValueError(
                    f"Criterion {criterion.criterion_id!r} "
                    f"references evidence not associated with its events."
                )

        for event in self.timeline:
            missing_evidence = (
                set(event.evidence_ids) - evidence_ids
            )

            if missing_evidence:
                raise ValueError(
                    f"Event {event.event_id!r} "
                    f"references unknown evidence."
                )

    def _validate_timeline_order(self) -> None:
        """Ensure events are ordered chronologically."""

        for previous, current in zip(
            self.timeline,
            self.timeline[1:],
        ):
            previous_key = (
                previous.timestamp_start,
                previous.timestamp_end,
                previous.event_id,
            )
            current_key = (
                current.timestamp_start,
                current.timestamp_end,
                current.event_id,
            )

            if previous_key > current_key:
                raise ValueError(
                    "Timeline events must be chronologically ordered."
                )

    def _validate_evidence_order(self) -> None:
        """Ensure evidence is ordered by frame interval."""

        for previous, current in zip(
            self.evidence,
            self.evidence[1:],
        ):
            previous_key = (
                previous.frame_start,
                previous.frame_end,
                previous.evidence_id,
            )
            current_key = (
                current.frame_start,
                current.frame_end,
                current.evidence_id,
            )

            if previous_key > current_key:
                raise ValueError(
                    "Evidence must be ordered by frame interval."
                )
