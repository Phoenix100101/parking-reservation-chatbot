"""
RAG Chatbot — Seed Data Generator
==================================

Generates seed data for the two backing stores described in
`Rag_architecture.txt` and `schema.sql`:

  • Weaviate  → static facility & parking descriptions (FacilityInfo, ParkingDetails)
  • Postgres  → spaces, operating_hours, reservations

Two modes (auto-detected, or forced via --mode):

  1. live   — connect to Weaviate + Postgres and INSERT directly.
  2. dump   — write JSON (Weaviate) and SQL (Postgres) files to ./seed_output/
              so you can load them later or inspect them.

Usage
-----
    python seed_data.py                      # auto: live if reachable, else dump
    python seed_data.py --mode dump          # always write files
    python seed_data.py --mode live          # require live connections
    python seed_data.py --scale 0.25 --reservations 10   # smaller dev dataset

Environment variables (only used in --mode live)
------------------------------------------------
    WEAVIATE_URL          default: http://localhost:8080
    WEAVIATE_API_KEY      default: (none)
    POSTGRES_DSN          default: postgresql://postgres:postgres@localhost:5432/chatbot

Dependencies
------------
    pip install weaviate-client psycopg[binary]    # only needed for --mode live
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# Reproducibility
# =============================================================================
RNG_SEED = 42
random.seed(RNG_SEED)


# =============================================================================
# Static content — FacilityInfo (vector-store source documents)
# =============================================================================
# Each entry corresponds to one logical document. The generator chunks each
# `description` into ~300-token pieces with ~50-token overlap (architecture spec)
# and stores each chunk as a separate Weaviate object so retrieval is granular.
# =============================================================================
FACILITY_DOCS: list[dict[str, Any]] = [
    {
        "category": "location",
        "source_doc": "facility_overview.md",
        "description": (
            "The parking facility is located at 12 Mangilik El Avenue in Astana, "
            "directly adjacent to the central business district and a five-minute "
            "walk from the Expo metro station. The main vehicle entrance is on the "
            "north side of the building, accessible from Turan Avenue. Pedestrian "
            "entrances are located on the east and west sides, both leading to the "
            "central elevator lobby. The facility has six floors of parking above "
            "ground (floors 1 through 6) plus two underground levels reserved for "
            "long-term contract holders. GPS coordinates: 51.0903 N, 71.4180 E. "
            "The nearest landmarks are the Khan Shatyr shopping mall to the south "
            "and the Palace of Independence to the east. Drivers approaching from "
            "the airport should take Kabanbay Batyr Avenue west and turn right onto "
            "Turan; signage is posted approximately 300 meters before the entrance."
        ),
    },
    {
        "category": "booking_process",
        "source_doc": "how_to_book.md",
        "description": (
            "Reservations can be made through the chatbot, the mobile app, or the "
            "facility website. To book through the chatbot, simply state that you "
            "want to reserve a parking space and provide the requested details: "
            "the date you need parking, the time range (start and end), your "
            "preferred zone or floor if you have one, the vehicle license plate, "
            "and a contact email address where the confirmation will be sent. "
            "Reservations can be made up to thirty days in advance and require "
            "confirmation within ten minutes of being placed, otherwise the hold "
            "expires and the space is released back to the available pool. "
            "Cancellations made more than two hours before the start time are "
            "refunded in full. Cancellations made within two hours of start time "
            "incur a fifty percent fee. No-shows are charged the full reserved "
            "amount and may affect future booking eligibility."
        ),
    },
    {
        "category": "parking",
        "source_doc": "parking_zones.md",
        "description": (
            "The facility is divided into three zones based on access type and "
            "pricing. Zone A occupies floors 1 and 2 and is reserved for short-term "
            "visitors and hourly bookings; spaces here are wider than standard and "
            "located closest to the main lobby and elevators. Zone B covers floors "
            "3 and 4 and is the standard zone for daily and multi-day reservations. "
            "Zone C is on floors 5 and 6 and is the budget tier with reduced rates; "
            "it is also where electric vehicle charging stations are concentrated, "
            "with twelve fast-charging bays distributed across the two floors. "
            "Each zone has its own dedicated elevator bank to avoid congestion at "
            "peak times. Zone signage uses color coding: Zone A is blue, Zone B is "
            "green, and Zone C is orange."
        ),
    },
    {
        "category": "general",
        "source_doc": "amenities.md",
        "description": (
            "The facility offers a range of amenities for visitors. All floors have "
            "twenty-four-hour CCTV coverage and on-site security personnel patrol "
            "the premises every two hours. Restrooms are available on floors 1, 3, "
            "and 5. A small convenience store and coffee kiosk are located in the "
            "main lobby on the ground floor and are open from six in the morning "
            "until ten in the evening. Free Wi-Fi is available throughout the "
            "building; the network name is ParkConnect and no password is required. "
            "Vending machines are placed near the elevator banks on every floor. "
            "An air pump and windshield washer station is provided on the ground "
            "floor near the exit. Lost and found inquiries can be directed to the "
            "security desk in the main lobby."
        ),
    },
    {
        "category": "general",
        "source_doc": "accessibility.md",
        "description": (
            "Accessible parking spaces are provided on every floor and located "
            "closest to the elevators. Each floor has a minimum of four accessible "
            "spaces, marked with the international accessibility symbol and painted "
            "blue. All elevators are wheelchair accessible and have braille floor "
            "indicators. The main entrance and all pedestrian entrances are step-"
            "free with automatic sliding doors. Accessible restrooms are available "
            "on floors 1, 3, and 5 alongside the standard restrooms. Service animals "
            "are welcome anywhere on the premises. Visitors who require additional "
            "assistance can request help at the security desk in the main lobby or "
            "by pressing the assistance button at any of the elevator panels, which "
            "connects directly to on-duty staff."
        ),
    },
    {
        "category": "booking_process",
        "source_doc": "payment_and_rates.md",
        "description": (
            "Payment is processed at the time of reservation confirmation. Accepted "
            "payment methods include all major credit and debit cards as well as "
            "Kaspi QR for local users. Hourly rates apply in Zone A and start at "
            "five hundred tenge per hour. Daily rates apply in Zone B and start at "
            "three thousand tenge per day. Zone C is the most economical option at "
            "two thousand tenge per day. Multi-day discounts are automatically "
            "applied for reservations of three or more consecutive days. Monthly "
            "passes are available at the facility office for long-term users. "
            "Receipts are emailed automatically upon successful payment. Refunds "
            "for eligible cancellations are processed back to the original payment "
            "method within five business days."
        ),
    },
]


# =============================================================================
# Static content — ParkingDetails (per-zone facts)
# =============================================================================
PARKING_DETAILS: list[dict[str, Any]] = [
    {
        "zone_name": "Zone A",
        "floor": 1,
        "capacity_total": 60,
        "amenities": ["wider spaces", "closest to lobby", "hourly billing", "accessible spaces"],
        "description": (
            "Zone A on floor 1 is the premium short-term zone, designed for "
            "visitors who need quick access to the main lobby. Spaces here are "
            "fifteen percent wider than the standard parking dimensions, making "
            "them suitable for larger vehicles and easier loading. The zone is "
            "billed hourly and is recommended for visits of less than six hours. "
            "Four accessible spaces are reserved at the end of the row closest "
            "to the elevator bank."
        ),
    },
    {
        "zone_name": "Zone A",
        "floor": 2,
        "capacity_total": 60,
        "amenities": ["wider spaces", "hourly billing", "covered", "accessible spaces"],
        "description": (
            "Zone A on floor 2 mirrors floor 1 in pricing and dimensions, "
            "providing additional capacity for short-term visitors during peak "
            "hours. Floor 2 is fully covered and slightly quieter than floor 1, "
            "making it a good choice during inclement weather. Accessible spaces "
            "are available adjacent to the elevators."
        ),
    },
    {
        "zone_name": "Zone B",
        "floor": 3,
        "capacity_total": 80,
        "amenities": ["daily billing", "covered", "standard size", "accessible spaces"],
        "description": (
            "Zone B on floor 3 is the standard zone for daily reservations and "
            "is the most common choice for office workers and day visitors. "
            "Spaces are standard size and the floor is fully covered. The "
            "elevator bank for Zone B is independent of Zone A to reduce "
            "congestion during morning and evening peak hours."
        ),
    },
    {
        "zone_name": "Zone B",
        "floor": 4,
        "capacity_total": 80,
        "amenities": ["daily billing", "covered", "multi-day discounts", "accessible spaces"],
        "description": (
            "Zone B on floor 4 is identical in layout to floor 3 and is "
            "preferred by users with multi-day reservations because of the "
            "automatic discount applied for three or more consecutive days. "
            "Lighting on this floor was upgraded to LED in 2024."
        ),
    },
    {
        "zone_name": "Zone C",
        "floor": 5,
        "capacity_total": 100,
        "amenities": ["budget rate", "EV charging", "daily billing", "accessible spaces"],
        "description": (
            "Zone C on floor 5 is the economy zone with the lowest daily rates "
            "in the facility. Six fast-charging stations for electric vehicles "
            "are located along the south wall; charging is billed separately "
            "from parking. The zone is slightly further from the main lobby but "
            "is well-connected by a dedicated elevator bank."
        ),
    },
    {
        "zone_name": "Zone C",
        "floor": 6,
        "capacity_total": 100,
        "amenities": ["budget rate", "EV charging", "open-air", "accessible spaces"],
        "description": (
            "Zone C on floor 6 is the rooftop level and offers another six "
            "EV fast-charging stations. The level is open-air, which makes it "
            "preferable for tall vehicles such as vans and SUVs that exceed "
            "the height limit of covered floors. Daily rates match floor 5."
        ),
    },
]


# =============================================================================
# Chunking helper — approximate token-count chunking with overlap
# =============================================================================
def chunk_text(text: str, max_tokens: int = 300, overlap_tokens: int = 50) -> list[str]:
    """
    Approximate token chunking by whitespace. Good enough for seed data;
    swap in a real tokenizer (tiktoken, transformers) in the production
    ingestion pipeline.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max_tokens - overlap_tokens
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


# =============================================================================
# Postgres seed builders
# =============================================================================
@dataclass
class Space:
    floor: int
    space_number: str
    is_available: bool
    last_updated: datetime
    # `id` is assigned by Postgres SERIAL — left unset until insert.
    id: int | None = None


@dataclass
class OperatingHours:
    day_of_week: int | None
    specific_date: date | None
    open_time: time
    close_time: time
    is_holiday: bool
    is_closed: bool
    note: str | None


@dataclass
class Reservation:
    id: str  # uuid string
    user_session_id: str
    space_id: int   # filled in after spaces are inserted / numbered
    start_time: datetime
    end_time: datetime
    vehicle_plate: str
    contact_email: str
    status: str
    created_at: datetime
    updated_at: datetime


# Per-floor capacity is derived from PARKING_DETAILS so that the dynamic
# Postgres data (live availability) agrees with the static Weaviate data
# (zone descriptions). If you change capacity_total in PARKING_DETAILS,
# the spaces table will follow automatically.
#
# NOTE on schema vs architecture:
#   The architecture doc references a `zones` SQL table (zone_id, hourly_rate,
#   daily_rate). That table is NOT in schema.sql — zone metadata lives in
#   Weaviate's ParkingDetails class instead. The `spaces` table only stores
#   floor + space_number + availability, and the chatbot derives the zone
#   from the floor number (Zone A = floors 1-2, B = 3-4, C = 5-6).
FLOOR_CAPACITIES: dict[int, int] = {
    d["floor"]: d["capacity_total"] for d in PARKING_DETAILS
}


def build_spaces(scale: float = 1.0) -> list[Space]:
    """
    Build one Space row per parking spot, using FLOOR_CAPACITIES as the
    source of truth for how many spaces exist on each floor. About 30%
    are pre-marked unavailable to make the dataset realistic for the
    chatbot to query.

    `scale` lets you shrink the dataset for quick testing without losing
    the per-floor proportions (e.g. scale=0.25 → 15/15/20/20/25/25).
    """
    now = datetime.now(timezone.utc)
    spaces: list[Space] = []
    for floor in sorted(FLOOR_CAPACITIES):
        capacity = max(1, int(round(FLOOR_CAPACITIES[floor] * scale)))
        for n in range(1, capacity + 1):
            spaces.append(
                Space(
                    floor=floor,
                    space_number=f"F{floor}-{n:03d}",
                    is_available=random.random() > 0.30,
                    last_updated=now - timedelta(minutes=random.randint(0, 240)),
                )
            )
    return spaces


def build_operating_hours() -> list[OperatingHours]:
    """
    Weekly recurring rows (Sun=0 .. Sat=6) plus a couple of date-specific
    holiday overrides matching the partial unique indexes in schema.sql.
    """
    rows: list[OperatingHours] = []

    # Weekly: Mon–Fri 06:00–23:00, Sat 08:00–22:00, Sun 09:00–21:00
    weekly = {
        0: (time(9, 0),  time(21, 0)),   # Sunday
        1: (time(6, 0),  time(23, 0)),   # Monday
        2: (time(6, 0),  time(23, 0)),
        3: (time(6, 0),  time(23, 0)),
        4: (time(6, 0),  time(23, 0)),
        5: (time(6, 0),  time(23, 0)),
        6: (time(8, 0),  time(22, 0)),   # Saturday
    }
    for dow, (open_t, close_t) in weekly.items():
        rows.append(OperatingHours(
            day_of_week=dow,
            specific_date=None,
            open_time=open_t,
            close_time=close_t,
            is_holiday=False,
            is_closed=False,
            note="Standard weekly hours",
        ))

    # Date-specific holiday overrides
    today = date.today()
    overrides = [
        # Closed all day
        (date(today.year, 1, 1),  time(0, 0),  time(0, 0),  True,  True,  "New Year's Day — closed"),
        # Reduced hours
        (date(today.year, 3, 22), time(10, 0), time(18, 0), True,  False, "Nauryz — reduced hours"),
        (date(today.year, 5, 1),  time(10, 0), time(18, 0), True,  False, "Unity Day — reduced hours"),
        (date(today.year, 12, 31), time(8, 0), time(18, 0), True,  False, "New Year's Eve — early close"),
    ]
    for d, open_t, close_t, is_holiday, is_closed, note in overrides:
        rows.append(OperatingHours(
            day_of_week=None,
            specific_date=d,
            open_time=open_t,
            close_time=close_t,
            is_holiday=is_holiday,
            is_closed=is_closed,
            note=note,
        ))
    return rows


def random_plate() -> str:
    # Kazakhstan-style format: NNN LLL RR (3 digits, space, 3 letters, space,
    # 2-digit region code). Matches the regex r"\d{3}\s[A-Z]{3}\s\d{2}",
    # which is what the output guardrail's plate regex would key on.
    # Stays well within VARCHAR(20).
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"   # no I/O/Q to avoid digit confusion
    digits = "0123456789"
    region_codes = ["01", "02", "03", "05", "10", "12", "16", "17"]
    return (
        "".join(random.choices(digits, k=3))
        + " "
        + "".join(random.choices(letters, k=3))
        + " "
        + random.choice(region_codes)
    )


def random_email() -> str:
    first = random.choice(["alex", "marina", "dmitri", "aigerim", "yerlan", "zhanna",
                           "olga", "kanat", "sasha", "nurlan", "aida", "timur"])
    last = random.choice(["smith", "ivanov", "kim", "abdullin", "petrov", "lee",
                          "tashkenov", "novak", "khan", "park", "yusupov"])
    domain = random.choice(["example.com", "mail.test", "gmail.example", "company.kz"])
    return f"{first}.{last}{random.randint(1, 99)}@{domain}"


def build_reservations(spaces: list[Space], n: int) -> list[Reservation]:
    """
    Build `n` plausible reservations referencing positions in `spaces`.
    `space_id` is assigned as the 1-based index because spaces are inserted
    in the same order and Postgres SERIAL starts at 1; in dump-mode we just
    rely on that ordering, in live-mode we use the IDs returned from INSERT.
    """
    reservations: list[Reservation] = []
    now = datetime.now(timezone.utc)
    statuses = ["pending", "confirmed", "confirmed", "confirmed", "cancelled", "expired"]

    for _ in range(n):
        space_idx = random.randrange(len(spaces))   # 0-based
        # We'll resolve to actual DB id later (live) or assume idx+1 (dump).
        space_id_placeholder = space_idx + 1

        # Random start time within ±10 days, duration 1–8 hours, on a half hour
        offset_hours = random.randint(-240, 240)
        start = (now + timedelta(hours=offset_hours)).replace(minute=0, second=0, microsecond=0)
        if random.random() < 0.5:
            start += timedelta(minutes=30)
        duration_hours = random.choice([1, 2, 3, 4, 6, 8])
        end = start + timedelta(hours=duration_hours)

        created = start - timedelta(hours=random.randint(1, 72))
        if created > now:
            created = now - timedelta(hours=random.randint(1, 24))

        reservations.append(Reservation(
            id=str(uuid.uuid4()),
            user_session_id=f"sess_{uuid.uuid4().hex[:16]}",
            space_id=space_id_placeholder,
            start_time=start,
            end_time=end,
            vehicle_plate=random_plate(),
            contact_email=random_email(),
            status=random.choice(statuses),
            created_at=created,
            updated_at=created + timedelta(minutes=random.randint(0, 30)),
        ))
    return reservations


# =============================================================================
# Weaviate object builder
# =============================================================================
def build_weaviate_objects() -> dict[str, list[dict[str, Any]]]:
    """Return objects keyed by Weaviate class name."""
    facility_objs: list[dict[str, Any]] = []
    for doc in FACILITY_DOCS:
        for i, chunk in enumerate(chunk_text(doc["description"])):
            facility_objs.append({
                "description": chunk,
                "category": doc["category"],
                "source_doc": doc["source_doc"],
                "chunk_index": i,
            })

    parking_objs: list[dict[str, Any]] = []
    for d in PARKING_DETAILS:
        # ParkingDetails entries are short and self-contained; no chunking needed.
        parking_objs.append({
            "zone_name": d["zone_name"],
            "floor": d["floor"],
            "capacity_total": d["capacity_total"],
            "amenities": d["amenities"],
            "description": d["description"],
        })

    return {"FacilityInfo": facility_objs, "ParkingDetails": parking_objs}


# =============================================================================
# DUMP MODE — write JSON + SQL to disk
# =============================================================================
def dump_mode(
    out_dir: Path,
    spaces: list[Space],
    hours: list[OperatingHours],
    reservations: list[Reservation],
    weaviate_objs: dict[str, list[dict[str, Any]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Weaviate JSON (one file per class) ----------------------------------
    for cls, objs in weaviate_objs.items():
        path = out_dir / f"weaviate_{cls}.json"
        path.write_text(json.dumps(objs, indent=2, ensure_ascii=False))
        print(f"  wrote {path}  ({len(objs)} objects)")

    # ---- Postgres SQL --------------------------------------------------------
    sql_path = out_dir / "postgres_seed.sql"
    lines: list[str] = []
    lines.append("-- Auto-generated seed data. Apply AFTER schema.sql.")
    lines.append("BEGIN;")
    lines.append("")

    # spaces
    lines.append("-- spaces ------------------------------------------------------")
    for s in spaces:
        lines.append(
            "INSERT INTO spaces (floor, space_number, is_available, last_updated) "
            f"VALUES ({s.floor}, {sql_str(s.space_number)}, {sql_bool(s.is_available)}, "
            f"{sql_ts(s.last_updated)});"
        )
    lines.append("")

    # operating_hours
    lines.append("-- operating_hours --------------------------------------------")
    for h in hours:
        lines.append(
            "INSERT INTO operating_hours (day_of_week, specific_date, open_time, close_time, "
            "is_holiday, is_closed, note) VALUES ("
            f"{sql_int_or_null(h.day_of_week)}, "
            f"{sql_date_or_null(h.specific_date)}, "
            f"{sql_time(h.open_time)}, "
            f"{sql_time(h.close_time)}, "
            f"{sql_bool(h.is_holiday)}, "
            f"{sql_bool(h.is_closed)}, "
            f"{sql_str_or_null(h.note)});"
        )
    lines.append("")

    # reservations — assumes spaces above were inserted in the same order so
    # SERIAL ids run 1..N. If spaces already exist in the DB this won't be true;
    # use --mode live for correctness in that case.
    lines.append("-- reservations ----------------------------------------------")
    lines.append("-- NOTE: space_id values assume spaces were just inserted with")
    lines.append("--       SERIAL starting at 1. If your `spaces` table already")
    lines.append("--       has rows, regenerate with --mode live for correct ids.")
    for r in reservations:
        lines.append(
            "INSERT INTO reservations (id, user_session_id, space_id, start_time, end_time, "
            "vehicle_plate, contact_email, status, created_at, updated_at) VALUES ("
            f"{sql_str(r.id)}::uuid, "
            f"{sql_str(r.user_session_id)}, "
            f"{r.space_id}, "
            f"{sql_ts(r.start_time)}, "
            f"{sql_ts(r.end_time)}, "
            f"{sql_str(r.vehicle_plate)}, "
            f"{sql_str(r.contact_email)}, "
            f"{sql_str(r.status)}::reservation_status, "
            f"{sql_ts(r.created_at)}, "
            f"{sql_ts(r.updated_at)});"
        )
    lines.append("")
    lines.append("COMMIT;")
    sql_path.write_text("\n".join(lines))
    print(
        f"  wrote {sql_path}  "
        f"({len(spaces)} spaces, {len(hours)} hours rows, "
        f"{len(reservations)} reservations)"
    )


# Tiny SQL-literal helpers (safe for seed data; do NOT reuse for user input)
def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def sql_str_or_null(s: str | None) -> str:
    return "NULL" if s is None else sql_str(s)

def sql_bool(b: bool) -> str:
    return "TRUE" if b else "FALSE"

def sql_int_or_null(i: int | None) -> str:
    return "NULL" if i is None else str(i)

def sql_date_or_null(d: date | None) -> str:
    return "NULL" if d is None else f"DATE '{d.isoformat()}'"

def sql_time(t: time) -> str:
    return f"TIME '{t.strftime('%H:%M:%S')}'"

def sql_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return f"TIMESTAMPTZ '{ts.isoformat()}'"


# =============================================================================
# LIVE MODE — connect & insert
# =============================================================================
def live_mode(
    spaces: list[Space],
    hours: list[OperatingHours],
    reservations: list[Reservation],
    weaviate_objs: dict[str, list[dict[str, Any]]],
) -> None:
    insert_into_postgres(spaces, hours, reservations)
    insert_into_weaviate(weaviate_objs)


def insert_into_postgres(
    spaces: list[Space],
    hours: list[OperatingHours],
    reservations: list[Reservation],
) -> None:
    try:
        import psycopg
    except ImportError:
        sys.exit("psycopg not installed. Run: pip install 'psycopg[binary]'")

    dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/chatbot",
    )
    print(f"  connecting to Postgres at {redact_dsn(dsn)}")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # spaces
            space_id_map: dict[tuple[int, str], int] = {}
            for s in spaces:
                cur.execute(
                    "INSERT INTO spaces (floor, space_number, is_available, last_updated) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (s.floor, s.space_number, s.is_available, s.last_updated),
                )
                row = cur.fetchone()
                assert row is not None
                space_id_map[(s.floor, s.space_number)] = row[0]
            print(f"    inserted {len(spaces)} spaces")

            # operating_hours
            for h in hours:
                cur.execute(
                    "INSERT INTO operating_hours "
                    "(day_of_week, specific_date, open_time, close_time, "
                    " is_holiday, is_closed, note) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (h.day_of_week, h.specific_date, h.open_time, h.close_time,
                     h.is_holiday, h.is_closed, h.note),
                )
            print(f"    inserted {len(hours)} operating_hours rows")

            # reservations — remap placeholder space_id -> actual id from RETURNING
            spaces_list = list(space_id_map.values())
            for r in reservations:
                # placeholder was idx+1 from build_reservations; convert back to idx
                idx = r.space_id - 1
                idx = max(0, min(idx, len(spaces_list) - 1))
                actual_space_id = spaces_list[idx]
                cur.execute(
                    "INSERT INTO reservations "
                    "(id, user_session_id, space_id, start_time, end_time, "
                    " vehicle_plate, contact_email, status, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (r.id, r.user_session_id, actual_space_id,
                     r.start_time, r.end_time, r.vehicle_plate, r.contact_email,
                     r.status, r.created_at, r.updated_at),
                )
            print(f"    inserted {len(reservations)} reservations")
        conn.commit()
    print("  Postgres seed complete.")


def insert_into_weaviate(weaviate_objs: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import weaviate  # type: ignore
        from weaviate.classes.config import Configure, Property, DataType  # type: ignore
    except ImportError:
        sys.exit("weaviate-client not installed. Run: pip install weaviate-client")

    url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.environ.get("WEAVIATE_API_KEY")
    print(f"  connecting to Weaviate at {url}")

    auth = None
    if api_key:
        from weaviate.auth import AuthApiKey  # type: ignore
        auth = AuthApiKey(api_key)

    # weaviate-client v4 connection
    client = weaviate.connect_to_custom(
        http_host=url.replace("https://", "").replace("http://", "").split(":")[0],
        http_port=int(url.rsplit(":", 1)[-1]) if ":" in url.replace("https://", "").replace("http://", "") else 80,
        http_secure=url.startswith("https"),
        grpc_host=url.replace("https://", "").replace("http://", "").split(":")[0],
        grpc_port=50051,
        grpc_secure=url.startswith("https"),
        auth_credentials=auth,
    )

    try:
        # Define schemas if missing
        existing = {c.name for c in client.collections.list_all().values()}

        if "FacilityInfo" not in existing:
            client.collections.create(
                name="FacilityInfo",
                properties=[
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="source_doc", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="chunk_index", data_type=DataType.INT, skip_vectorization=True),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_contextionary(),
            )
            print("    created class FacilityInfo")

        if "ParkingDetails" not in existing:
            client.collections.create(
                name="ParkingDetails",
                properties=[
                    Property(name="zone_name", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="floor", data_type=DataType.INT, skip_vectorization=True),
                    Property(name="capacity_total", data_type=DataType.INT, skip_vectorization=True),
                    Property(name="amenities", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                    Property(name="description", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_contextionary(),
            )
            print("    created class ParkingDetails")

        # Insert
        for cls_name, objs in weaviate_objs.items():
            coll = client.collections.get(cls_name)
            with coll.batch.dynamic() as batch:
                for o in objs:
                    batch.add_object(properties=o)
            print(f"    inserted {len(objs)} objects into {cls_name}")
    finally:
        client.close()
    print("  Weaviate seed complete.")


def redact_dsn(dsn: str) -> str:
    # postgresql://user:pass@host:port/db  →  postgresql://user:***@host:port/db
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host}"
    return dsn


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Seed Weaviate + Postgres for the RAG chatbot.")
    ap.add_argument("--mode", choices=["auto", "live", "dump"], default="auto",
                    help="auto (default): try live, fall back to dump. "
                         "live: require live connections. "
                         "dump: write files only.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor on per-floor capacity from PARKING_DETAILS "
                         "(default: 1.0 → 60+60+80+80+100+100 = 480 spaces). "
                         "Use e.g. 0.25 for a smaller dev dataset.")
    ap.add_argument("--reservations", type=int, default=25,
                    help="number of seed reservations (default: 25)")
    ap.add_argument("--out-dir", type=Path, default=Path("seed_output"),
                    help="output directory for dump mode (default: ./seed_output)")
    args = ap.parse_args()

    print("Generating seed data...")
    spaces = build_spaces(scale=args.scale)
    hours = build_operating_hours()
    reservations = build_reservations(spaces, args.reservations)
    weaviate_objs = build_weaviate_objects()

    print(f"  built {len(spaces)} spaces "
          f"({sum(1 for s in spaces if s.is_available)} available)")
    print(f"  built {len(hours)} operating_hours rows "
          f"({sum(1 for h in hours if h.specific_date)} date overrides)")
    print(f"  built {len(reservations)} reservations")
    print(f"  built {sum(len(v) for v in weaviate_objs.values())} Weaviate objects "
          f"across {len(weaviate_objs)} classes")

    if args.mode == "dump":
        print("\nDump mode: writing files...")
        dump_mode(args.out_dir, spaces, hours, reservations, weaviate_objs)
        return

    if args.mode == "live":
        print("\nLive mode: inserting into Postgres + Weaviate...")
        live_mode(spaces, hours, reservations, weaviate_objs)
        return

    # auto: try live, fall back to dump on connection error
    print("\nAuto mode: attempting live insert...")
    try:
        live_mode(spaces, hours, reservations, weaviate_objs)
    except Exception as e:
        print(f"  live mode failed ({type(e).__name__}: {e})")
        print("  falling back to dump mode.")
        dump_mode(args.out_dir, spaces, hours, reservations, weaviate_objs)


if __name__ == "__main__":
    main()