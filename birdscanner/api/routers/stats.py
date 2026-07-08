"""Aggregate statistics endpoints powering the Statistics dashboard.

Every endpoint runs a read-only SQL ``GROUP BY`` aggregation over the detections
table, so the response stays small no matter how large the history grows (only
the aggregated buckets cross the wire, never the raw rows). Time bucketing uses
SQLite ``strftime`` on the stored timestamps; detection timestamps are written
with naive local ``datetime.now()``, so all bins are in local wall-clock time.

The temporal endpoints accept an optional ``from``/``to`` range (see
:class:`StatsRange`); the frontend derives the ``from`` cutoff from its 7d/30d/90d/all
selector. ``first-sightings`` is intentionally all-time (the new-species curve is
cumulative from the very first detection).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import ColumnElement, Integer, case, cast
from sqlmodel import Session, col, func, select

from birdscanner.db.models import DetectionRecord
from birdscanner.api.dependencies import get_session

router = APIRouter(prefix="/api/stats", tags=["stats"])

_DEFAULT_BIN_MINUTES = 15
_DEFAULT_TOP_SPECIES = 8
_OTHER_BUCKET = "Other"


# ---------------------------------------------------------------------------
# Shared query helpers
# ---------------------------------------------------------------------------


class StatsRange:
    """Optional ``from``/``to`` timestamp window shared by the temporal endpoints.

    Grouping the two range parameters into one dependency (mirroring
    ``DetectionFilters`` in :mod:`birdscanner.api.routers.detections`) keeps every
    endpoint signature short and gives the range-to-SQL translation a single home.
    """

    def __init__(
        self,
        from_: Optional[datetime] = Query(
            default=None, alias="from", description="Earliest timestamp (inclusive)"
        ),
        to: Optional[datetime] = Query(
            default=None, description="Latest timestamp (inclusive)"
        ),
    ) -> None:
        """Capture the request's range query parameters.

        Args:
            from_: Optional inclusive earliest timestamp (``from`` query alias).
            to: Optional inclusive latest timestamp.
        """
        self.from_ = from_
        self.to = to

    def apply(self, query):
        """Return ``query`` narrowed to the supplied range.

        Args:
            query: The base ``select(...)`` statement to filter.

        Returns:
            The statement with a ``where`` clause per supplied bound.
        """
        if self.from_ is not None:
            query = query.where(DetectionRecord.timestamp >= self.from_)
        if self.to is not None:
            query = query.where(DetectionRecord.timestamp <= self.to)
        return query


def _timestamp() -> Any:
    """Return the detection-timestamp column expression."""
    return col(DetectionRecord.timestamp)


def _count() -> ColumnElement[int]:
    """Return a ``COUNT(detections.id)`` expression."""
    # pylint: disable=not-callable  # sqlalchemy's func.count is a dynamic callable
    return func.count(col(DetectionRecord.id))


def _minute_of_day() -> ColumnElement[Any]:
    """Return minutes-since-local-midnight (0–1439) for the detection timestamp."""
    return cast(func.strftime("%H", _timestamp()), Integer) * 60 + cast(
        func.strftime("%M", _timestamp()), Integer
    )


def _period_column(interval: str) -> ColumnElement[Any]:
    """Return the text period-bucket expression for a ``day`` or ``week`` interval.

    Both formats sort lexicographically in chronological order, so ordering the
    query by this expression yields chronologically ordered buckets.

    Args:
        interval: Either ``"day"`` or ``"week"``.

    Returns:
        A ``strftime`` expression bucketing the timestamp into that interval.
    """
    if interval == "week":
        return func.strftime("%Y-%W", _timestamp())
    return func.strftime("%Y-%m-%d", _timestamp())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class StatsSummary(BaseModel):
    """Top-line totals for the statistics page header.

    Attributes:
        total: Total number of detections in the range.
        distinct_species: Number of distinct species seen in the range.
        first_detection: Earliest detection timestamp in the range (``None`` if empty).
        last_detection: Latest detection timestamp in the range (``None`` if empty).
        corrected_count: Number of user-corrected detections in the range.
    """

    total: int
    distinct_species: int
    first_detection: Optional[datetime]
    last_detection: Optional[datetime]
    corrected_count: int


@router.get("/summary", response_model=StatsSummary)
def get_summary(
    stats_range: StatsRange = Depends(),
    session: Session = Depends(get_session),
) -> StatsSummary:
    """Return top-line detection totals for the (optionally ranged) history.

    Args:
        stats_range: Injected ``from``/``to`` window.
        session: Injected database session.

    Returns:
        A :class:`StatsSummary` with totals, distinct-species count, first/last
        timestamps, and the corrected-detection count.
    """
    # pylint: disable=not-callable  # sqlalchemy aggregate funcs are dynamic callables
    # The mixed aggregate funcs below defeat select's typed overloads (call-overload).
    stmt = stats_range.apply(
        select(  # type: ignore[call-overload]
            _count(),
            func.count(func.distinct(col(DetectionRecord.species))),
            func.min(_timestamp()),
            func.max(_timestamp()),
            func.sum(case((col(DetectionRecord.corrected).is_(True), 1), else_=0)),
        )
    )
    total, distinct_species, first_seen, last_seen, corrected = session.exec(stmt).one()
    return StatsSummary(
        total=total or 0,
        distinct_species=distinct_species or 0,
        first_detection=first_seen,
        last_detection=last_seen,
        corrected_count=int(corrected or 0),
    )


# ---------------------------------------------------------------------------
# Time-of-day density
# ---------------------------------------------------------------------------


class TimeBin(BaseModel):
    """A single time-of-day histogram bucket.

    Attributes:
        minute: Start of the bucket in minutes since midnight (0–1439).
        count: Number of detections in the bucket.
    """

    minute: int
    count: int


class TimeOfDayResponse(BaseModel):
    """Time-of-day histogram used to draw the (client-smoothed) density curve.

    Attributes:
        bin_minutes: Width of each bucket in minutes.
        bins: The populated buckets, ordered by ``minute`` ascending.
    """

    bin_minutes: int
    bins: List[TimeBin]


@router.get("/time-of-day", response_model=TimeOfDayResponse)
def get_time_of_day(
    stats_range: StatsRange = Depends(),
    species: Optional[str] = Query(default=None, description="Filter to one species"),
    bin_minutes: int = Query(
        default=_DEFAULT_BIN_MINUTES, ge=1, le=120, description="Bucket width (minutes)"
    ),
    session: Session = Depends(get_session),
) -> TimeOfDayResponse:
    """Return a fine time-of-day histogram of detections.

    The client smooths these buckets (circular Gaussian) into a density curve, so
    only the compact histogram crosses the wire rather than every row.

    Args:
        stats_range: Injected ``from``/``to`` window.
        species: Optional single-species filter.
        bin_minutes: Bucket width in minutes (1–120).
        session: Injected database session.

    Returns:
        A :class:`TimeOfDayResponse` of populated buckets.
    """
    # Floor division (``//``): SQLAlchemy's ``/`` performs Python-style *true*
    # division (float), which would split one clock-minute bucket into several
    # spurious float groups. ``//`` floors, keeping one integer bucket per slot.
    bucket = _minute_of_day() // bin_minutes
    stmt = stats_range.apply(select(bucket.label("bucket"), _count().label("count")))
    if species is not None:
        stmt = stmt.where(DetectionRecord.species == species)
    stmt = stmt.group_by(bucket).order_by(bucket)
    rows = session.exec(stmt).all()
    bins = [TimeBin(minute=int(row[0]) * bin_minutes, count=row[1]) for row in rows]
    return TimeOfDayResponse(bin_minutes=bin_minutes, bins=bins)


# ---------------------------------------------------------------------------
# Hour x day-of-week activity heatmap
# ---------------------------------------------------------------------------


class ActivityCell(BaseModel):
    """One cell of the hour x day-of-week activity heatmap.

    Attributes:
        dow: Day of week normalised to Monday=0 ... Sunday=6.
        hour: Hour of day (0–23).
        count: Number of detections in that (day, hour) slot.
    """

    dow: int
    hour: int
    count: int


class ActivityResponse(BaseModel):
    """Sparse activity heatmap (only populated cells are returned).

    Attributes:
        cells: The populated ``(dow, hour)`` cells; the client fills the 7x24 grid.
    """

    cells: List[ActivityCell]


@router.get("/activity", response_model=ActivityResponse)
def get_activity(
    stats_range: StatsRange = Depends(),
    session: Session = Depends(get_session),
) -> ActivityResponse:
    """Return detection counts per (day-of-week, hour) slot.

    SQLite's ``%w`` numbers Sunday as 0; cells are re-based to Monday=0 so the
    heatmap reads Monday-first.

    Args:
        stats_range: Injected ``from``/``to`` window.
        session: Injected database session.

    Returns:
        An :class:`ActivityResponse` of populated cells.
    """
    dow = cast(func.strftime("%w", _timestamp()), Integer)
    hour = cast(func.strftime("%H", _timestamp()), Integer)
    stmt = stats_range.apply(
        select(dow.label("dow"), hour.label("hour"), _count().label("count"))
    )
    stmt = stmt.group_by(dow, hour).order_by(dow, hour)
    rows = session.exec(stmt).all()
    cells = [
        ActivityCell(dow=(int(row[0]) + 6) % 7, hour=int(row[1]), count=row[2])
        for row in rows
    ]
    return ActivityResponse(cells=cells)


# ---------------------------------------------------------------------------
# Sightings-over-time + diversity timeline
# ---------------------------------------------------------------------------


class TimelinePoint(BaseModel):
    """One interval of the sightings-over-time timeline.

    Attributes:
        date: The period bucket label (``YYYY-MM-DD`` for days, ``YYYY-WW`` weeks).
        total: Total detections in the period.
        distinct_species: Number of distinct species seen in the period.
        counts: Per-species detection counts, with non-top-N species folded into
            an ``"Other"`` bucket so the stacked chart stays legible.
    """

    date: str
    total: int
    distinct_species: int
    counts: Dict[str, int]


class TimelineResponse(BaseModel):
    """Timeline powering both the stacked sightings chart and the diversity line.

    Attributes:
        interval: The bucketing interval (``"day"`` or ``"week"``).
        species: The top-N species names (stack series order), most-common first.
        points: The per-interval points, ordered chronologically.
    """

    interval: str
    species: List[str]
    points: List[TimelinePoint]


def _build_timeline_points(
    rows: Sequence[Tuple[str, str, int]], top_n: int
) -> Tuple[List[str], List[TimelinePoint]]:
    """Fold ``(period, species, count)`` rows into ordered timeline points.

    The rows must be ordered by period so the assembled points come out
    chronologically. Species outside the overall top-``top_n`` (by total count)
    are aggregated into the ``"Other"`` bucket.

    Args:
        rows: ``(period, species, count)`` tuples grouped by period then species.
        top_n: How many species to keep as their own stack series.

    Returns:
        A ``(species_order, points)`` pair: the top-N species names (most-common
        first) and the per-period :class:`TimelinePoint` list.
    """
    totals: Dict[str, int] = {}
    for _, species, count in rows:
        totals[species] = totals.get(species, 0) + count
    top = {
        sp for sp, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    }

    ordered: List[str] = []
    acc: Dict[str, Dict[str, Any]] = {}
    for period, species, count in rows:
        if period not in acc:
            acc[period] = {"total": 0, "distinct": 0, "counts": {}}
            ordered.append(period)
        entry = acc[period]
        entry["total"] += count
        entry["distinct"] += 1
        key = species if species in top else _OTHER_BUCKET
        entry["counts"][key] = entry["counts"].get(key, 0) + count

    points = [
        TimelinePoint(
            date=period,
            total=acc[period]["total"],
            distinct_species=acc[period]["distinct"],
            counts=acc[period]["counts"],
        )
        for period in ordered
    ]
    species_order = sorted(top, key=lambda sp: (-totals[sp], sp))
    return species_order, points


@router.get("/timeline", response_model=TimelineResponse)
def get_timeline(
    stats_range: StatsRange = Depends(),
    interval: str = Query(default="day", pattern="^(day|week)$"),
    top: int = Query(
        default=_DEFAULT_TOP_SPECIES, ge=1, le=20, description="Species kept unstacked"
    ),
    session: Session = Depends(get_session),
) -> TimelineResponse:
    """Return per-interval detection counts, split by top-N species + ``Other``.

    Serves two charts from one query: the stacked sightings-over-time area (via
    ``counts``) and the species-diversity line (via ``distinct_species``).

    Args:
        stats_range: Injected ``from``/``to`` window.
        interval: Bucketing interval, ``"day"`` (default) or ``"week"``.
        top: Number of species to keep as their own series (1–20).
        session: Injected database session.

    Returns:
        A :class:`TimelineResponse` with the species stack order and points.
    """
    period = _period_column(interval)
    stmt = stats_range.apply(
        select(
            period.label("period"),
            col(DetectionRecord.species),
            _count().label("count"),
        )
    )
    stmt = stmt.group_by(period, col(DetectionRecord.species)).order_by(period)
    rows = session.exec(stmt).all()
    species_order, points = _build_timeline_points(rows, top)
    return TimelineResponse(interval=interval, species=species_order, points=points)


# ---------------------------------------------------------------------------
# Daily activity window (first / last seen per day)
# ---------------------------------------------------------------------------


class DayWindow(BaseModel):
    """The activity window for a single day.

    Attributes:
        date: The day (``YYYY-MM-DD``).
        first_minute: Minutes-since-midnight of the earliest detection that day.
        last_minute: Minutes-since-midnight of the latest detection that day.
        count: Number of detections that day.
    """

    date: str
    first_minute: int
    last_minute: int
    count: int


class DailyWindowResponse(BaseModel):
    """Per-day earliest/latest detection window.

    Attributes:
        days: One :class:`DayWindow` per populated day, ordered chronologically.
    """

    days: List[DayWindow]


@router.get("/daily-window", response_model=DailyWindowResponse)
def get_daily_window(
    stats_range: StatsRange = Depends(),
    session: Session = Depends(get_session),
) -> DailyWindowResponse:
    """Return the earliest and latest detection time for each day.

    Args:
        stats_range: Injected ``from``/``to`` window.
        session: Injected database session.

    Returns:
        A :class:`DailyWindowResponse` describing each day's activity band.
    """
    # pylint: disable=not-callable  # sqlalchemy aggregate funcs are dynamic callables
    day = func.strftime("%Y-%m-%d", _timestamp())
    minute = _minute_of_day()
    stmt = stats_range.apply(
        select(
            day.label("day"),
            func.min(minute).label("first_minute"),
            func.max(minute).label("last_minute"),
            _count().label("count"),
        )
    )
    stmt = stmt.group_by(day).order_by(day)
    rows = session.exec(stmt).all()
    days = [
        DayWindow(
            date=row[0],
            first_minute=int(row[1]),
            last_minute=int(row[2]),
            count=row[3],
        )
        for row in rows
    ]
    return DailyWindowResponse(days=days)


# ---------------------------------------------------------------------------
# First sightings (new-species cumulative curve)
# ---------------------------------------------------------------------------


class FirstSighting(BaseModel):
    """The first time a species was ever seen.

    Attributes:
        species: The species name.
        first_seen: Timestamp of that species' earliest detection.
    """

    species: str
    first_seen: datetime


@router.get("/first-sightings", response_model=List[FirstSighting])
def get_first_sightings(
    session: Session = Depends(get_session),
) -> List[FirstSighting]:
    """Return each species' earliest detection, ordered oldest-first (all-time).

    This is intentionally not range-filtered: the new-species curve is cumulative
    from the very first detection, so it always reflects the full history.

    Args:
        session: Injected database session.

    Returns:
        A list of :class:`FirstSighting`, ordered by ``first_seen`` ascending.
    """
    # pylint: disable=not-callable,assignment-from-no-return  # sqlalchemy func.min
    first_seen = func.min(_timestamp())
    stmt = (
        select(col(DetectionRecord.species), first_seen.label("first_seen"))
        .group_by(col(DetectionRecord.species))
        .order_by(first_seen)
    )
    rows = session.exec(stmt).all()
    return [FirstSighting(species=row[0], first_seen=row[1]) for row in rows]
