"""Postgres data-access layer for the parking RAG chatbot.

Read helpers (`get_available_spaces`, `count_available_spaces`,
`is_space_available`, `get_operating_hours`, `is_open_at`) back the dynamic-query
agent's tools. Reservation helpers (`save_reservation`, `get_reservation`,
`get_reservations_by_session`, `update_reservation_status`) back the reservation
agent. All queries target the ``parking`` schema (see sql_schema_script.sql).
"""

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Literal

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from config.configuration import get_settings

ReservationStatus = Literal["pending", "confirmed", "cancelled", "expired"]

# Reservations in these states occupy the space for their time window.
_ACTIVE_STATUSES = ("pending", "confirmed")

_pool: ConnectionPool | None = None
# Guards lazy creation/teardown of the process-wide pool so concurrent callers
# (LangGraph nodes, eval workers) can never create two pools and leak one.
_pool_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    """Return the process-wide connection pool, opening it on first use.

    Thread-safe via double-checked locking. The pool is created closed and
    opened explicitly (``open=False`` + :meth:`ConnectionPool.open`) so a dead
    database fails fast here instead of on the first query, and to avoid the
    deprecated constructor-side ``open=True``. Ownership of the pool's lifetime
    belongs to :func:`db_lifespan`, which closes it deterministically.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                settings = get_settings().postgres
                pool = ConnectionPool(
                    conninfo=settings.url,
                    min_size=settings.pool_min_size,
                    max_size=settings.pool_max_size,
                    timeout=settings.pool_timeout,
                    # Pin search_path at connect time (libpq option) so all
                    # queries resolve against the parking schema without
                    # per-call qualification.
                    kwargs={
                        "row_factory": dict_row,
                        "options": f"-c search_path={settings.schema_name}",
                    },
                    open=False,
                )
                pool.open(wait=True, timeout=settings.pool_timeout)
                _pool = pool
    return _pool


def close_pool() -> None:
    """Close the pool and stop its worker threads. Idempotent."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def db_lifespan() -> Iterator[None]:
    """Own the connection pool for the lifetime of a process or run.

    Open the pool on enter and close it deterministically on exit. Wrap the
    application entry point (CLI/bot ``main``), an eval run, or use it as a
    pytest fixture so the pool's workers are stopped when the work finishes
    instead of at interpreter shutdown.
    """
    _get_pool()
    try:
        yield
    finally:
        close_pool()


# =============================================================================
# spaces — live availability (read-only / dynamic-query tools)
# =============================================================================
def get_available_spaces(floor: int | None = None, limit: int = 20) -> list[dict]:
    """Return spaces currently free per the live ``is_available`` flag.

    Answers "what spots are open right now?". For booking a future window use
    :func:`is_space_available` instead, which checks reservation overlap.

    Args:
        floor: Restrict to a single floor, or ``None`` for all floors.
        limit: Maximum number of rows to return.

    Returns:
        Rows of ``{id, floor, space_number, last_updated}`` ordered by location.
    """
    sql = """
        SELECT id, floor, space_number, last_updated
        FROM spaces
        WHERE is_available = TRUE
          AND (%(floor)s::int IS NULL OR floor = %(floor)s::int)
        ORDER BY floor, space_number
        LIMIT %(limit)s
    """
    with _get_pool().connection() as conn:
        cur = conn.execute(sql, {"floor": floor, "limit": limit})
        return cur.fetchall()


def count_available_spaces(floor: int | None = None) -> list[dict]:
    """Count currently-free spaces grouped by floor.

    Answers "how many free spots on floor 3?" (or the whole garage).

    Args:
        floor: Restrict to a single floor, or ``None`` for all floors.

    Returns:
        Rows of ``{floor, available}`` ordered by floor.
    """
    sql = """
        SELECT floor, COUNT(*) AS available
        FROM spaces
        WHERE is_available = TRUE
          AND (%(floor)s::int IS NULL OR floor = %(floor)s::int)
        GROUP BY floor
        ORDER BY floor
    """
    with _get_pool().connection() as conn:
        cur = conn.execute(sql, {"floor": floor})
        return cur.fetchall()


def is_space_available(
    space_id: int, start_time: datetime, end_time: datetime
) -> bool:
    """Whether a space can be booked for the half-open window ``[start, end)``.

    A space is bookable when it exists and has no overlapping reservation in an
    active state (``pending`` / ``confirmed``). Two windows overlap when
    ``existing.start < new.end AND existing.end > new.start``.

    Args:
        space_id: Target space.
        start_time: Requested start (timezone-aware).
        end_time: Requested end (timezone-aware).

    Returns:
        ``True`` if the space exists and the window is free.

    Raises:
        ValueError: If ``end_time`` is not after ``start_time``.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be after start_time")

    sql = """
        SELECT
            EXISTS (SELECT 1 FROM spaces WHERE id = %(space_id)s) AS space_exists,
            EXISTS (
                SELECT 1 FROM reservations
                WHERE space_id = %(space_id)s
                  AND status = ANY(%(statuses)s)
                  AND start_time < %(end_time)s
                  AND end_time > %(start_time)s
            ) AS has_conflict
    """
    params = {
        "space_id": space_id,
        "statuses": list(_ACTIVE_STATUSES),
        "start_time": start_time,
        "end_time": end_time,
    }
    with _get_pool().connection() as conn:
        row = conn.execute(sql, params).fetchone()
    return bool(row["space_exists"]) and not row["has_conflict"]


# =============================================================================
# operating_hours — open/close times with holiday overrides (read-only)
# =============================================================================
def get_operating_hours(target_date: date) -> dict | None:
    """Resolve operating hours for a calendar date.

    Precedence: a ``specific_date`` override (holiday / special hours) wins over
    the recurring weekly row. ``day_of_week`` uses 0 = Sunday … 6 = Saturday.

    Args:
        target_date: The date to look up.

    Returns:
        Row of ``{open_time, close_time, is_holiday, is_closed, note}``, or
        ``None`` if no rule is defined for that date.
    """
    # date.isoweekday(): Mon=1..Sun=7  ->  % 7 gives Sun=0..Sat=6
    day_of_week = target_date.isoweekday() % 7

    sql = """
        SELECT open_time, close_time, is_holiday, is_closed, note
        FROM operating_hours
        WHERE specific_date = %(target_date)s

        UNION ALL

        SELECT open_time, close_time, is_holiday, is_closed, note
        FROM operating_hours
        WHERE specific_date IS NULL
          AND is_holiday = FALSE
          AND day_of_week = %(day_of_week)s
          AND NOT EXISTS (
              SELECT 1 FROM operating_hours WHERE specific_date = %(target_date)s
          )
        LIMIT 1
    """
    params = {"target_date": target_date, "day_of_week": day_of_week}
    with _get_pool().connection() as conn:
        return conn.execute(sql, params).fetchone()


def get_weekly_operating_hours() -> list[dict]:
    """Return the recurring weekly schedule (the regular working days/hours).

    Answers general questions like "what are your working days/hours?" that are
    not tied to a specific calendar date. Date-specific holiday overrides are
    excluded — use :func:`get_operating_hours` for a particular date.

    ``day_of_week`` uses 0 = Sunday … 6 = Saturday.

    Returns:
        Rows of ``{day_of_week, open_time, close_time, is_closed, note}`` for
        each defined weekday, ordered Sunday → Saturday.
    """
    sql = """
        SELECT day_of_week, open_time, close_time, is_closed, note
        FROM operating_hours
        WHERE specific_date IS NULL
          AND is_holiday = FALSE
        ORDER BY day_of_week
    """
    with _get_pool().connection() as conn:
        return conn.execute(sql).fetchall()


def is_open_at(when: datetime) -> bool:
    """Whether the facility is open at a specific moment.

    Args:
        when: The instant to check; its date selects the rule and its
            wall-clock time is compared against open/close.

    Returns:
        ``True`` if a rule exists for the date, it is not marked closed, and the
        time falls within ``[open_time, close_time)``.
    """
    hours = get_operating_hours(when.date())
    if hours is None or hours["is_closed"]:
        return False
    return hours["open_time"] <= when.time() < hours["close_time"]


# =============================================================================
# reservations — bookings collected by the reservation agent (writes)
# =============================================================================
def save_reservation(
    user_session_id: str,
    space_id: int,
    start_time: datetime,
    end_time: datetime,
    vehicle_plate: str,
    contact_email: str,
    status: ReservationStatus = "pending",
) -> str:
    """Insert a reservation and return its generated UUID.

    ``id`` is generated here (``uuid.uuid4``) because the schema has no DB
    default. The caller is responsible for confirming availability first via
    :func:`is_space_available`.

    The booked space's live ``is_available`` flag is set to ``FALSE`` in the
    same transaction as the insert, so the two never diverge: either both land
    or neither does.

    Returns:
        The new reservation's id as a string.
    """
    reservation_id = str(uuid.uuid4())
    insert_sql = """
        INSERT INTO reservations (
            id, user_session_id, space_id, start_time, end_time,
            vehicle_plate, contact_email, status
        )
        VALUES (
            %(id)s, %(user_session_id)s, %(space_id)s, %(start_time)s,
            %(end_time)s, %(vehicle_plate)s, %(contact_email)s, %(status)s
        )
        RETURNING id
    """
    mark_unavailable_sql = """
        UPDATE spaces
        SET is_available = FALSE, last_updated = NOW()
        WHERE id = %(space_id)s
    """
    params = {
        "id": reservation_id,
        "user_session_id": user_session_id,
        "space_id": space_id,
        "start_time": start_time,
        "end_time": end_time,
        "vehicle_plate": vehicle_plate,
        "contact_email": contact_email,
        "status": status,
    }
    with _get_pool().connection() as conn:
        row = conn.execute(insert_sql, params).fetchone()
        conn.execute(mark_unavailable_sql, {"space_id": space_id})
    return str(row["id"])


def get_reservation(reservation_id: str) -> dict | None:
    """Fetch a single reservation by its UUID, or ``None`` if not found."""
    sql = "SELECT * FROM reservations WHERE id = %(id)s"
    with _get_pool().connection() as conn:
        return conn.execute(sql, {"id": reservation_id}).fetchone()


def get_reservations_by_session(session_id: str) -> list[dict]:
    """Return all reservations for a chat session, newest first.

    Backs "show my bookings"; uses ``idx_reservations_session``.
    """
    sql = """
        SELECT * FROM reservations
        WHERE user_session_id = %(session_id)s
        ORDER BY created_at DESC
    """
    with _get_pool().connection() as conn:
        return conn.execute(sql, {"session_id": session_id}).fetchall()


def update_reservation_status(
    reservation_id: str, status: ReservationStatus
) -> bool:
    """Transition a reservation to a new status and bump ``updated_at``.

    When the new status releases the space (``cancelled`` / ``expired``), the
    space's live ``is_available`` flag is restored to ``TRUE`` — but only if no
    *other* reservation still occupies it in an active state. This mirrors the
    flag flip done in :func:`save_reservation`; all writes share one transaction.

    Returns:
        ``True`` if a row was updated, ``False`` if the id was not found.
    """
    update_sql = """
        UPDATE reservations
        SET status = %(status)s, updated_at = NOW()
        WHERE id = %(id)s
        RETURNING space_id
    """
    # Re-open the space only when nothing else active is holding it.
    free_space_sql = """
        UPDATE spaces
        SET is_available = TRUE, last_updated = NOW()
        WHERE id = %(space_id)s
          AND NOT EXISTS (
              SELECT 1 FROM reservations
              WHERE space_id = %(space_id)s
                AND status = ANY(%(statuses)s)
          )
    """
    with _get_pool().connection() as conn:
        row = conn.execute(
            update_sql, {"status": status, "id": reservation_id}
        ).fetchone()
        if row is None:
            return False
        if status in ("cancelled", "expired"):
            conn.execute(
                free_space_sql,
                {"space_id": row["space_id"], "statuses": list(_ACTIVE_STATUSES)},
            )
        return True
