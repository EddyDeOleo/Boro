
"""Spatial region definitions and entity-region observations."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TypeAlias


Point2D: TypeAlias = tuple[float, float]

_EPSILON = 1e-9


class RegionType(str, Enum):
    """Supported semantic types of spatial regions."""

    TARGET_ZONE = "TARGET_ZONE"
    FORBIDDEN_ZONE = "FORBIDDEN_ZONE"
    WORK_AREA = "WORK_AREA"


class RegionRelation(str, Enum):
    """Observed spatial relationship between an entity and a region."""

    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    PARTIALLY_INTERSECTING = "PARTIALLY_INTERSECTING"
    UNKNOWN = "UNKNOWN"


def _is_valid_integer(value: object) -> bool:
    """Return whether a value is an integer but not a boolean."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_number(value: object) -> bool:
    """Return whether a value is a finite integer or float, excluding booleans."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _orientation(
    a: Point2D,
    b: Point2D,
    c: Point2D,
) -> float:
    """Calculate the cross product of vectors AB and AC."""

    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _on_segment(
    a: Point2D,
    b: Point2D,
    point: Point2D,
) -> bool:
    """Return whether a point lies on a line segment."""

    return (
        abs(_orientation(a, b, point)) <= _EPSILON
        and min(a[0], b[0]) - _EPSILON
        <= point[0]
        <= max(a[0], b[0]) + _EPSILON
        and min(a[1], b[1]) - _EPSILON
        <= point[1]
        <= max(a[1], b[1]) + _EPSILON
    )


def _segments_intersect(
    a: Point2D,
    b: Point2D,
    c: Point2D,
    d: Point2D,
) -> bool:
    """Return whether two line segments intersect."""

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    first_straddles = (
        (o1 > _EPSILON and o2 < -_EPSILON)
        or (o1 < -_EPSILON and o2 > _EPSILON)
    )

    second_straddles = (
        (o3 > _EPSILON and o4 < -_EPSILON)
        or (o3 < -_EPSILON and o4 > _EPSILON)
    )

    if first_straddles and second_straddles:
        return True

    return (
        (abs(o1) <= _EPSILON and _on_segment(a, b, c))
        or (abs(o2) <= _EPSILON and _on_segment(a, b, d))
        or (abs(o3) <= _EPSILON and _on_segment(c, d, a))
        or (abs(o4) <= _EPSILON and _on_segment(c, d, b))
    )


def _validate_polygon(polygon: tuple[Point2D, ...]) -> None:
    """Validate polygon geometry and reject degenerate or invalid polygons."""

    if len(polygon) < 3:
        raise ValueError("A polygon must contain at least three points.")

    for point in polygon:
        if len(point) != 2:
            raise ValueError(
                "Every polygon point must contain two coordinates."
            )

        for coordinate in point:
            if not _is_valid_number(coordinate):
                raise ValueError(
                    "Polygon coordinates must be finite numeric values."
                )

    if len(set(polygon)) != len(polygon):
        raise ValueError("Polygon cannot contain duplicate vertices.")

    area_twice = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )

    if abs(area_twice) <= _EPSILON:
        raise ValueError("Polygon must have a non-zero area.")

    vertex_count = len(polygon)

    for first_index in range(vertex_count):
        first_start = polygon[first_index]
        first_end = polygon[(first_index + 1) % vertex_count]

        for second_index in range(first_index + 1, vertex_count):
            # Consecutive edges share a valid vertex.
            if second_index == first_index + 1:
                continue

            # The first and last edges also share a valid vertex.
            if first_index == 0 and second_index == vertex_count - 1:
                continue

            second_start = polygon[second_index]
            second_end = polygon[(second_index + 1) % vertex_count]

            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                raise ValueError("Polygon cannot self-intersect.")


@dataclass(frozen=True)
class RegionDefinition:
    """A polygonal region defined in image coordinates."""

    region_id: str
    name: str
    region_type: RegionType

    polygon: tuple[Point2D, ...]

    frame_width: int
    frame_height: int
    reference_frame: int = 0

    coordinate_system: str = "pixel"

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be a non-empty string.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")

        if not isinstance(self.region_type, RegionType):
            raise ValueError("region_type must be a RegionType value.")

        if (
            not _is_valid_integer(self.frame_width)
            or self.frame_width <= 0
        ):
            raise ValueError("frame_width must be a positive integer.")

        if (
            not _is_valid_integer(self.frame_height)
            or self.frame_height <= 0
        ):
            raise ValueError("frame_height must be a positive integer.")

        if (
            not _is_valid_integer(self.reference_frame)
            or self.reference_frame < 0
        ):
            raise ValueError(
                "reference_frame must be a non-negative integer."
            )

        if self.coordinate_system != "pixel":
            raise ValueError("Only pixel coordinates are currently supported.")

        # Normalize JSON-style lists into immutable tuples before validation.
        try:
            normalized_polygon = tuple(
                tuple(point) for point in self.polygon
            )
        except (TypeError, ValueError):
            raise ValueError(
                "polygon must contain valid coordinate pairs."
            ) from None

        object.__setattr__(self, "polygon", normalized_polygon)

        _validate_polygon(self.polygon)

        for x, y in self.polygon:
            if not 0 <= x < self.frame_width:
                raise ValueError("Polygon point exceeds frame width.")

            if not 0 <= y < self.frame_height:
                raise ValueError("Polygon point exceeds frame height.")

    def contains_point(self, point: Point2D) -> bool:
        """Return whether a point is inside the polygon.

        Points on the polygon boundary are treated as inside.
        """

        try:
            if len(point) != 2:
                raise ValueError(
                    "point must contain two coordinates."
                )
        except TypeError:
            raise ValueError(
                "point must contain two coordinates."
            ) from None

        x, y = point

        if not _is_valid_number(x) or not _is_valid_number(y):
            raise ValueError(
                "Point coordinates must be finite numeric values."
            )

        vertices = self.polygon
        inside = False

        for index in range(len(vertices)):
            x1, y1 = vertices[index]
            x2, y2 = vertices[(index + 1) % len(vertices)]

            if _on_segment((x1, y1), (x2, y2), (x, y)):
                return True

            if (y1 > y) != (y2 > y):
                intersection_x = (
                    x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                )

                if x < intersection_x:
                    inside = not inside

        return inside


@dataclass(frozen=True)
class RegionObservation:
    """Observed relationship between an entity and a spatial region."""

    entity_id: str
    region_id: str
    frame_index: int

    relation: RegionRelation

    centroid_inside: bool | None = None
    intersection_area_pixels: float | None = None
    overlap_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("entity_id must be a non-empty string.")

        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be a non-empty string.")

        if (
            not _is_valid_integer(self.frame_index)
            or self.frame_index < 0
        ):
            raise ValueError(
                "frame_index must be a non-negative integer."
            )

        if not isinstance(self.relation, RegionRelation):
            raise ValueError("relation must be a RegionRelation value.")

        if (
            self.centroid_inside is not None
            and not isinstance(self.centroid_inside, bool)
        ):
            raise ValueError("centroid_inside must be a boolean or None.")

        if self.intersection_area_pixels is not None:
            if (
                not _is_valid_number(self.intersection_area_pixels)
                or self.intersection_area_pixels < 0
            ):
                raise ValueError(
                    "intersection_area_pixels must be finite "
                    "and non-negative."
                )

        if self.overlap_ratio is not None:
            if (
                not _is_valid_number(self.overlap_ratio)
                or not 0.0 <= self.overlap_ratio <= 1.0
            ):
                raise ValueError(
                    "overlap_ratio must be finite and between 0 and 1."
                )
