"""Latency benchmarks for the Postgres data-access layer.

Design
------
The benchmark core is **generic and pure** with respect to the thing being
measured: :func:`benchmark` takes a zero-argument callable, times it
``iterations`` times with :func:`time.perf_counter`, and returns a
:class:`LatencyStats` of percentiles. It does not know about Postgres — so the
remaining ``postgres_client`` read helpers can be benchmarked by handing
:func:`benchmark` a different closure (this is the intended extension point).

What this measures
------------------
We time the ``db.*`` helper end to end — *acquire a pooled connection → execute →
fetch*. That is exactly what the dynamic-query tool pays per call, so the numbers
reflect the bot's real DB cost, not a bare ``EXECUTE`` against an already-held
connection. LLM latency is deliberately out of scope.

Methodology
-----------
* **Warmup.** The pool is lazy (``min_size=2``) and Postgres fills its page
  cache on first touch, so the first calls pay connect + cold-cache costs that
  have nothing to do with steady-state query latency. We run ``warmup`` calls
  and discard them before measuring.
* **Percentiles, not mean.** A mean hides the tail. For a chatbot the p95/p99 is
  what a user feels on a bad turn, so that is what we report (plus min/max as the
  hot floor and worst single sample).
* **Representative inputs.** Latency depends on the data shape, so each function
  is benchmarked over named *scenarios* rather than one convenient input. For
  :func:`benchmark_is_space_available` the two scenarios are a *free* window (no
  overlapping reservation) and a *conflict* window (an active reservation
  overlaps) — they exercise the same ``idx_reservations_space_time`` index but
  the conflict path short-circuits on the ``has_conflict`` EXISTS.

The inputs in ``__main__`` assume the shipped seed (see ``postgres_seed.sql``):
space 1 has no reservations; space 86 has two active reservations on 2026-05-09.
Re-seeding with different data only changes the inputs, not the harness.

Run::

    uv run python -m evaluation.performance_eval
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone

from data.sql_store import postgres_client as db


# ---------------------------------------------------------------------------
# Pure stats helpers
# ---------------------------------------------------------------------------
def _percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile of an **ascending-sorted** sample (q in [0, 100]).

    Nearest-rank (rather than interpolation) keeps every reported number an
    actually-observed latency, which is the honest thing to quote for a tail.
    """
    if not sorted_ms:
        raise ValueError("cannot take a percentile of an empty sample")
    rank = math.ceil(q / 100 * len(sorted_ms))
    index = min(max(rank, 1), len(sorted_ms)) - 1
    return sorted_ms[index]


@dataclass
class LatencyStats:
    """Latency distribution for one benchmarked callable, in milliseconds."""

    label: str
    n: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float

    @classmethod
    def from_samples(cls, label: str, samples_ms: list[float]) -> LatencyStats:
        ordered = sorted(samples_ms)
        return cls(
            label=label,
            n=len(ordered),
            min_ms=ordered[0],
            p50_ms=_percentile(ordered, 50),
            p95_ms=_percentile(ordered, 95),
            p99_ms=_percentile(ordered, 99),
            max_ms=ordered[-1],
            mean_ms=sum(ordered) / len(ordered),
        )

    def __str__(self) -> str:
        return (
            f"{self.label:<28} n={self.n:<5} "
            f"min={self.min_ms:6.2f}  p50={self.p50_ms:6.2f}  "
            f"p95={self.p95_ms:6.2f}  p99={self.p99_ms:6.2f}  "
            f"max={self.max_ms:7.2f}  mean={self.mean_ms:6.2f}  (ms)"
        )


# ---------------------------------------------------------------------------
# Generic benchmark runner (the reuse point for the other DB helpers)
# ---------------------------------------------------------------------------
def benchmark(
    call: Callable[[], object],
    *,
    label: str,
    iterations: int = 1000,
    warmup: int = 50,
) -> LatencyStats:
    """Time a zero-arg callable ``iterations`` times and return its latency.

    ``warmup`` calls run first and are discarded so pool fill-up and cold cache
    don't pollute the tail. ``call`` should wrap the function under test together
    with its arguments, e.g. ``lambda: db.is_space_available(1, start, end)``.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    for _ in range(warmup):
        call()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        samples_ms.append((time.perf_counter() - start) * 1000)

    return LatencyStats.from_samples(label, samples_ms)


# ---------------------------------------------------------------------------
# is_space_available — the worked example
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AvailabilityScenario:
    """One input case for :func:`benchmark_is_space_available`."""

    label: str
    space_id: int
    start_time: datetime
    end_time: datetime


def benchmark_is_space_available(
    scenarios: list[AvailabilityScenario],
    *,
    iterations: int = 1000,
    warmup: int = 50,
) -> list[LatencyStats]:
    """Benchmark :func:`postgres_client.is_space_available` per scenario.

    Each scenario is timed independently because the free and conflict paths
    have different query costs; returning one :class:`LatencyStats` per scenario
    keeps those distributions separate.
    """
    results: list[LatencyStats] = []
    for sc in scenarios:
        stats = benchmark(
            lambda sc=sc: db.is_space_available(
                sc.space_id, sc.start_time, sc.end_time
            ),
            label=f"is_space_available[{sc.label}]",
            iterations=iterations,
            warmup=warmup,
        )
        results.append(stats)
    return results


def _default_scenarios() -> list[AvailabilityScenario]:
    """Scenarios tied to the shipped seed (``postgres_seed.sql``)."""
    return [
        # Space 1 has no reservations → the conflict EXISTS finds nothing.
        AvailabilityScenario(
            label="free",
            space_id=1,
            start_time=datetime(2026, 5, 9, 7, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc),
        ),
        # Space 86 has an active (confirmed) booking 06:30–09:30 on this date.
        AvailabilityScenario(
            label="conflict",
            space_id=86,
            start_time=datetime(2026, 5, 9, 7, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc),
        ),
    ]


# ---------------------------------------------------------------------------
# get_available_spaces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpacesScenario:
    """One input case for :func:`benchmark_get_available_spaces`.

    The two axes that move latency here are ``floor`` (the ``%(floor)s IS NULL
    OR floor = %(floor)s`` predicate — passing a floor lets the planner use
    ``idx_spaces_floor``; ``None`` cannot, so it leans on the partial
    ``idx_spaces_available``) and ``limit`` (how many rows are actually fetched
    and marshalled back).
    """

    label: str
    floor: int | None
    limit: int


def benchmark_get_available_spaces(
    scenarios: list[SpacesScenario],
    *,
    iterations: int = 1000,
    warmup: int = 50,
) -> list[LatencyStats]:
    """Benchmark :func:`postgres_client.get_available_spaces` per scenario.

    Unlike ``is_space_available`` (which returns a single boolean), this helper
    returns rows, so ``limit`` directly drives fetch/marshal cost — hence a
    small-``limit`` and a large-``limit`` scenario are timed separately.
    """
    results: list[LatencyStats] = []
    for sc in scenarios:
        stats = benchmark(
            lambda sc=sc: db.get_available_spaces(floor=sc.floor, limit=sc.limit),
            label=f"get_available_spaces[{sc.label}]",
            iterations=iterations,
            warmup=warmup,
        )
        results.append(stats)
    return results


def _default_spaces_scenarios() -> list[SpacesScenario]:
    """Scenarios tied to the shipped seed (180 spaces, ~half available)."""
    return [
        # All floors, the helper's default page size — the common "what's open?".
        SpacesScenario(label="all-floors,limit20", floor=None, limit=20),
        # One floor: adds the floor predicate (idx_spaces_floor eligible).
        SpacesScenario(label="floor1,limit20", floor=1, limit=20),
        # All floors, limit past the row count: no LIMIT cutoff, max marshalling.
        SpacesScenario(label="all-floors,limit500", floor=None, limit=500),
    ]


# ---------------------------------------------------------------------------
# get_operating_hours
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatingHoursScenario:
    """One input case for :func:`benchmark_get_operating_hours`.

    The latency-relevant axis is which arm of the query's ``UNION ALL`` does the
    work. An *override* date matches the first arm directly on
    ``uq_operating_hours_date``. An ordinary *weekday* leaves the first arm empty
    and pays for the second arm's correlated ``NOT EXISTS`` override-probe plus
    the ``day_of_week`` lookup.
    """

    label: str
    target_date: date


def benchmark_get_operating_hours(
    scenarios: list[OperatingHoursScenario],
    *,
    iterations: int = 1000,
    warmup: int = 50,
) -> list[LatencyStats]:
    """Benchmark :func:`postgres_client.get_operating_hours` per scenario."""
    results: list[LatencyStats] = []
    for sc in scenarios:
        stats = benchmark(
            lambda sc=sc: db.get_operating_hours(sc.target_date),
            label=f"get_operating_hours[{sc.label}]",
            iterations=iterations,
            warmup=warmup,
        )
        results.append(stats)
    return results


def _default_hours_scenarios() -> list[OperatingHoursScenario]:
    """Scenarios tied to the shipped seed (all 7 weekdays + two date overrides).

    Note: every weekday has a recurring row, so the helper never returns ``None``
    under this seed — there is no reachable "no rule" case to benchmark.
    """
    return [
        # 2026-01-01 has a specific_date override (New Year — closed).
        OperatingHoursScenario(label="override", target_date=date(2026, 1, 1)),
        # 2026-06-15 is an ordinary Monday with no override → weekly arm + probe.
        OperatingHoursScenario(label="weekday", target_date=date(2026, 6, 15)),
    ]


if __name__ == "__main__":
    # Own the pool for the run so it closes when the benchmark finishes, not at
    # interpreter shutdown — otherwise the closing log lands after the results.
    with db.db_lifespan():
        print("is_space_available latency (warm cache, sequential)\n")
        for stats in benchmark_is_space_available(_default_scenarios()):
            print(stats)

        print("\nget_available_spaces latency (warm cache, sequential)\n")
        for stats in benchmark_get_available_spaces(_default_spaces_scenarios()):
            print(stats)

        print("\nget_operating_hours latency (warm cache, sequential)\n")
        for stats in benchmark_get_operating_hours(_default_hours_scenarios()):
            print(stats)