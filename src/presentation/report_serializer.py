
"""JSON serialization for Boro presentation reports."""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.presentation.models import (
    CriterionView,
    EvidenceView,
    PresentationReport,
    ReportOverview,
    TimelineEventView,
)


class ReportSerializationError(ValueError):
    """Raised when a presentation report cannot be serialized or parsed."""


class ReportSerializer:
    """Serialize and deserialize PresentationReport instances."""

    SCHEMA_VERSION = "1.0"

    _REPORT_KEYS = {
        "overview",
        "criteria",
        "timeline",
        "evidence",
        "limitations",
    }

    _OVERVIEW_KEYS = {
        "verdict",
        "summary",
        "coverage_scope",
        "covers_entire_context",
        "start_time_seconds",
        "end_time_seconds",
        "duration_seconds",
        "fps",
        "context_frame_count",
        "observed_frame_count",
    }

    _CRITERION_KEYS = {
        "criterion_id",
        "requirement",
        "status",
        "is_mandatory",
        "explanation",
        "event_ids",
        "evidence_ids",
    }

    _TIMELINE_KEYS = {
        "event_id",
        "event_type",
        "status",
        "frame_start",
        "frame_end",
        "timestamp_start",
        "timestamp_end",
        "entity_ids",
        "confidence",
        "evidence_ids",
        "attributes",
    }

    _EVIDENCE_KEYS = {
        "evidence_id",
        "evidence_type",
        "description",
        "frame_start",
        "frame_end",
        "timestamp_start_seconds",
        "timestamp_end_seconds",
        "observation_ids",
        "entity_ids",
        "metrics",
        "limitations",
    }

    @classmethod
    def to_dict(cls, report: PresentationReport) -> dict[str, Any]:
        """Convert a presentation report into a JSON-compatible dictionary."""
        if not isinstance(report, PresentationReport):
            raise ReportSerializationError(
                "report must be a PresentationReport instance."
            )

        return {
            "schema_version": cls.SCHEMA_VERSION,
            "report": {
                "overview": cls._overview_to_dict(report.overview),
                "criteria": [
                    cls._criterion_to_dict(item)
                    for item in report.criteria
                ],
                "timeline": [
                    cls._timeline_event_to_dict(item)
                    for item in report.timeline
                ],
                "evidence": [
                    cls._evidence_to_dict(item)
                    for item in report.evidence
                ],
                "limitations": list(report.limitations),
            },
        }

    @classmethod
    def to_json(
        cls,
        report: PresentationReport,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize a presentation report as a JSON string."""
        data = cls.to_dict(report)

        try:
            return json.dumps(
                data,
                indent=indent,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                "The presentation report contains values that cannot be "
                "serialized as valid JSON."
            ) from exc

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PresentationReport:
        """Reconstruct and validate a presentation report from a dictionary."""
        if not isinstance(data, Mapping):
            raise ReportSerializationError(
                "Serialized report must be a mapping."
            )

        cls._validate_keys(
            data,
            {"schema_version", "report"},
            "serialized report",
        )

        schema_version = data["schema_version"]
        if schema_version != cls.SCHEMA_VERSION:
            raise ReportSerializationError(
                f"Unsupported schema version: {schema_version!r}. "
                f"Expected {cls.SCHEMA_VERSION!r}."
            )

        report_data = data["report"]
        cls._require_mapping(report_data, "report")
        cls._validate_keys(
            report_data,
            cls._REPORT_KEYS,
            "report",
        )

        overview = cls._overview_from_dict(report_data["overview"])

        criteria_data = cls._require_list(
            report_data["criteria"],
            "report.criteria",
        )
        criteria = tuple(
            cls._criterion_from_dict(item, index)
            for index, item in enumerate(criteria_data)
        )

        timeline_data = cls._require_list(
            report_data["timeline"],
            "report.timeline",
        )
        timeline = tuple(
            cls._timeline_event_from_dict(item, index)
            for index, item in enumerate(timeline_data)
        )

        evidence_data = cls._require_list(
            report_data["evidence"],
            "report.evidence",
        )
        evidence = tuple(
            cls._evidence_from_dict(item, index)
            for index, item in enumerate(evidence_data)
        )

        limitations = cls._require_string_list(
            report_data["limitations"],
            "report.limitations",
        )

        try:
            return PresentationReport(
                overview=overview,
                criteria=criteria,
                timeline=timeline,
                evidence=evidence,
                limitations=limitations,
            )
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                "The serialized report violates PresentationReport "
                "validation rules."
            ) from exc

    @classmethod
    def from_json(cls, json_string: str) -> PresentationReport:
        """Parse a JSON string and reconstruct its presentation report."""
        if not isinstance(json_string, str):
            raise ReportSerializationError(
                "json_string must be a string."
            )

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as exc:
            raise ReportSerializationError(
                "Input is not valid JSON."
            ) from exc

        return cls.from_dict(data)

    @classmethod
    def _overview_to_dict(
        cls,
        overview: ReportOverview,
    ) -> dict[str, Any]:
        return {
            "verdict": overview.verdict,
            "summary": overview.summary,
            "coverage_scope": overview.coverage_scope,
            "covers_entire_context": overview.covers_entire_context,
            "start_time_seconds": overview.start_time_seconds,
            "end_time_seconds": overview.end_time_seconds,
            "duration_seconds": overview.duration_seconds,
            "fps": overview.fps,
            "context_frame_count": overview.context_frame_count,
            "observed_frame_count": overview.observed_frame_count,
        }

    @classmethod
    def _criterion_to_dict(
        cls,
        criterion: CriterionView,
    ) -> dict[str, Any]:
        return {
            "criterion_id": criterion.criterion_id,
            "requirement": criterion.requirement,
            "status": criterion.status,
            "is_mandatory": criterion.is_mandatory,
            "explanation": criterion.explanation,
            "event_ids": list(criterion.event_ids),
            "evidence_ids": list(criterion.evidence_ids),
        }

    @classmethod
    def _timeline_event_to_dict(
        cls,
        event: TimelineEventView,
    ) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "status": event.status,
            "frame_start": event.frame_start,
            "frame_end": event.frame_end,
            "timestamp_start": event.timestamp_start,
            "timestamp_end": event.timestamp_end,
            "entity_ids": list(event.entity_ids),
            "confidence": event.confidence,
            "evidence_ids": list(event.evidence_ids),
            "attributes": dict(event.attributes),
        }

    @classmethod
    def _evidence_to_dict(
        cls,
        evidence: EvidenceView,
    ) -> dict[str, Any]:
        return {
            "evidence_id": evidence.evidence_id,
            "evidence_type": evidence.evidence_type,
            "description": evidence.description,
            "frame_start": evidence.frame_start,
            "frame_end": evidence.frame_end,
            "timestamp_start_seconds": evidence.timestamp_start_seconds,
            "timestamp_end_seconds": evidence.timestamp_end_seconds,
            "observation_ids": list(evidence.observation_ids),
            "entity_ids": list(evidence.entity_ids),
            "metrics": dict(evidence.metrics),
            "limitations": list(evidence.limitations),
        }

    @classmethod
    def _overview_from_dict(
        cls,
        data: Any,
    ) -> ReportOverview:
        cls._require_mapping(data, "report.overview")
        cls._validate_keys(data, cls._OVERVIEW_KEYS, "report.overview")

        try:
            return ReportOverview(**dict(data))
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                "Invalid report.overview data."
            ) from exc

    @classmethod
    def _criterion_from_dict(
        cls,
        data: Any,
        index: int,
    ) -> CriterionView:
        location = f"report.criteria[{index}]"
        cls._require_mapping(data, location)
        cls._validate_keys(data, cls._CRITERION_KEYS, location)

        values = dict(data)
        values["event_ids"] = cls._require_string_list(
            values["event_ids"],
            f"{location}.event_ids",
        )
        values["evidence_ids"] = cls._require_string_list(
            values["evidence_ids"],
            f"{location}.evidence_ids",
        )

        try:
            return CriterionView(**values)
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                f"Invalid {location} data."
            ) from exc

    @classmethod
    def _timeline_event_from_dict(
        cls,
        data: Any,
        index: int,
    ) -> TimelineEventView:
        location = f"report.timeline[{index}]"
        cls._require_mapping(data, location)
        cls._validate_keys(data, cls._TIMELINE_KEYS, location)

        values = dict(data)
        values["entity_ids"] = cls._require_string_list(
            values["entity_ids"],
            f"{location}.entity_ids",
        )
        values["evidence_ids"] = cls._require_string_list(
            values["evidence_ids"],
            f"{location}.evidence_ids",
        )
        values["attributes"] = cls._require_mapping(
            values["attributes"],
            f"{location}.attributes",
        )

        try:
            return TimelineEventView(**values)
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                f"Invalid {location} data."
            ) from exc

    @classmethod
    def _evidence_from_dict(
        cls,
        data: Any,
        index: int,
    ) -> EvidenceView:
        location = f"report.evidence[{index}]"
        cls._require_mapping(data, location)
        cls._validate_keys(data, cls._EVIDENCE_KEYS, location)

        values = dict(data)
        values["observation_ids"] = cls._require_string_list(
            values["observation_ids"],
            f"{location}.observation_ids",
        )
        values["entity_ids"] = cls._require_string_list(
            values["entity_ids"],
            f"{location}.entity_ids",
        )
        values["limitations"] = cls._require_string_list(
            values["limitations"],
            f"{location}.limitations",
        )
        values["metrics"] = cls._require_mapping(
            values["metrics"],
            f"{location}.metrics",
        )

        try:
            return EvidenceView(**values)
        except (TypeError, ValueError) as exc:
            raise ReportSerializationError(
                f"Invalid {location} data."
            ) from exc

    @staticmethod
    def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ReportSerializationError(
                f"{location} must be an object."
            )
        if not all(isinstance(key, str) for key in value):
            raise ReportSerializationError(
                f"{location} must contain string keys only."
            )
        return value

    @staticmethod
    def _require_list(value: Any, location: str) -> list[Any]:
        if not isinstance(value, list):
            raise ReportSerializationError(
                f"{location} must be an array."
            )
        return value

    @classmethod
    def _require_string_list(
        cls,
        value: Any,
        location: str,
    ) -> tuple[str, ...]:
        items = cls._require_list(value, location)

        if not all(isinstance(item, str) for item in items):
            raise ReportSerializationError(
                f"{location} must contain strings only."
            )

        return tuple(items)

    @staticmethod
    def _validate_keys(
        data: Mapping[str, Any],
        expected_keys: set[str],
        location: str,
    ) -> None:
        actual_keys = set(data.keys())
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys

        if missing or unexpected:
            details = []

            if missing:
                details.append(f"missing keys: {sorted(missing)}")

            if unexpected:
                details.append(f"unexpected keys: {sorted(unexpected)}")

            raise ReportSerializationError(
                f"Invalid {location}: {'; '.join(details)}."
            )
