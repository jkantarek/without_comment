import datetime
import hashlib
import json
import logging
import sqlite3
import os
from typing import List, Dict, Optional, Tuple

import httpx

from field_mapper import FieldMapper, REQUIRED_FIELDS

logger = logging.getLogger(__name__)

# Default Dent Theatre configuration
DEFAULT_BASE_URL = "https://thedentheatre.com"
DEFAULT_API_TEMPLATE = "/api/open/GetItemsByMonth?month={month}&collectionId={collection_id}"
DEFAULT_COLLECTION_ID = "64bc3c406b6d3d1edd3c84db"
DEFAULT_REFERER_TEMPLATE = "https://thedentheatre.com/calendar?view=calendar&month={month}"

# Default request headers (from the curl example)
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:151.0) Gecko/20100101 Firefox/151.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-GPC": "1",
    "Priority": "u=0",
}

# Default lookahead in months
DEFAULT_LOOKAHEAD_MONTHS = 3


class CalendarFetcher:
    """
    Fetches event calendar data from a venue API, stores JSON locally with
    SHA-256 hashes for diff tracking, and provides parsed event data grouped
    by week.
    """

    def __init__(
        self,
        db_path: str,
        base_url: str = "",
        api_template: str = "",
        collection_id: str = "",
        request_headers: Optional[Dict[str, str]] = None,
        lookahead_months: int = DEFAULT_LOOKAHEAD_MONTHS,
    ):
        self.db_path = db_path
        self.base_url = base_url or DEFAULT_BASE_URL
        self.api_template = api_template or DEFAULT_API_TEMPLATE
        self.collection_id = collection_id or DEFAULT_COLLECTION_ID
        self.request_headers = request_headers or DEFAULT_REQUEST_HEADERS
        self.lookahead_months = lookahead_months
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            logger.warning(
                f"Could not enable WAL mode: {e}. Falling back to TRUNCATE."
            )
            conn.execute("PRAGMA journal_mode=TRUNCATE")
        return conn

    def _init_db(self):
        """Create calendar-related tables if they don't exist."""
        with self._get_conn() as conn:
            # Calendar sources configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    api_template TEXT NOT NULL,
                    collection_id TEXT NOT NULL,
                    referer_template TEXT,
                    request_headers_json TEXT,
                    lookahead_months INTEGER DEFAULT 3,
                    enabled INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Stored JSON per month per source, with hash for diff detection
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_months (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    month_key TEXT NOT NULL,  -- e.g., '06-2026'
                    raw_json TEXT,
                    json_hash TEXT NOT NULL,
                    fetch_url TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'success',  -- success, failed, pending
                    error_message TEXT,
                    FOREIGN KEY (source_id) REFERENCES calendar_sources(id),
                    UNIQUE(source_id, month_key)
                )
            """)

            # Individual parsed events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    month_id INTEGER NOT NULL,
                    event_id TEXT NOT NULL,  -- from API (unique event identifier)
                    title TEXT,
                    full_url TEXT,
                    start_date_ms INTEGER,
                    end_date_ms INTEGER,
                    start_date_iso TEXT,
                    end_date_iso TEXT,
                    week_key TEXT,  -- ISO week like '2026-W24'
                    week_guid TEXT, -- RSS GUID for the week this event belongs to
                    body_json TEXT,  -- full JSON for this event
                    FOREIGN KEY (source_id) REFERENCES calendar_sources(id),
                    FOREIGN KEY (month_id) REFERENCES calendar_months(id)
                )
            """)
            # Field mappings for normalizing different API JSON structures
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_field_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    internal_field TEXT NOT NULL,  -- event_id, title, full_url, start_date_ms, end_date_ms
                    json_path TEXT NOT NULL,  -- e.g., '.id', '.structuredContent.startDate'
                    transform TEXT DEFAULT 'identity',  -- identity, str, int, multiply_1000, iso_to_ms
                    FOREIGN KEY (source_id) REFERENCES calendar_sources(id),
                    UNIQUE(source_id, internal_field)
                )
            """)

            # Create indexes for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calendar_months_source_month
                ON calendar_months(source_id, month_key)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calendar_events_source_week
                ON calendar_events(source_id, week_key)
            """)

            conn.commit()

    def add_source(
        self,
        name: str,
        base_url: str = "",
        api_template: str = "",
        collection_id: str = "",
        referer_template: str = "",
        request_headers: Optional[Dict[str, str]] = None,
        lookahead_months: int = DEFAULT_LOOKAHEAD_MONTHS,
    ) -> int:
        """Add a calendar source and return its ID."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calendar_sources
                    (name, base_url, api_template, collection_id,
                     referer_template, request_headers_json, lookahead_months)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    base_url or DEFAULT_BASE_URL,
                    api_template or DEFAULT_API_TEMPLATE,
                    collection_id or DEFAULT_COLLECTION_ID,
                    referer_template or DEFAULT_REFERER_TEMPLATE,
                    json.dumps(request_headers or DEFAULT_REQUEST_HEADERS),
                    lookahead_months,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_sources(self, enabled_only: bool = True) -> List[Dict]:
        """Retrieve configured calendar sources."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM calendar_sources"
            if enabled_only:
                query += " WHERE enabled = 1"
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def enable_source(self, source_id: int) -> None:
        self._update_source_field(source_id, "enabled", 1)

    def disable_source(self, source_id: int) -> None:
        self._update_source_field(source_id, "enabled", 0)

    def delete_source(self, source_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM calendar_events WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM calendar_months WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM calendar_sources WHERE id = ?", (source_id,))
            conn.commit()

    def _update_source_field(self, source_id: int, field: str, value) -> None:
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE calendar_sources SET {field} = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (value, source_id),
            )
            conn.commit()

    def get_field_mappings(self, source_id: int) -> List[Dict]:
        """Get all field mappings for a calendar source."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM calendar_field_mappings WHERE source_id = ? ORDER BY internal_field",
                (source_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_field_mapping(
        self,
        source_id: int,
        internal_field: str,
        json_path: str,
        transform: str = "identity",
    ) -> None:
        """Add or update a field mapping for a calendar source."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO calendar_field_mappings
                    (source_id, internal_field, json_path, transform)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, internal_field) DO UPDATE SET
                    json_path = excluded.json_path,
                    transform = excluded.transform
                """,
                (source_id, internal_field, json_path, transform),
            )
            conn.commit()

    def delete_field_mapping(self, source_id: int, internal_field: str) -> None:
        """Delete a field mapping for a calendar source."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM calendar_field_mappings WHERE source_id = ? AND internal_field = ?",
                (source_id, internal_field),
            )
            conn.commit()

    def get_required_fields(self) -> List[str]:
        """Return the list of required internal fields."""
        return REQUIRED_FIELDS

    def get_available_transforms(self) -> Dict[str, str]:
        """Return available transforms with descriptions."""
        return {
            "identity": "No transformation (raw value)",
            "str": "Convert to string",
            "int": "Convert to integer",
            "float": "Convert to float",
            "trim": "Trim whitespace from string",
            "multiply_1000": "Multiply by 1000 (e.g., seconds to milliseconds)",
            "iso_to_ms": "Convert ISO date string to milliseconds",
        }

    def get_default_mappings(self) -> List[Dict]:
        """
        Return default field mappings for common calendar API patterns.
        These are presets for quick setup.
        """
        return [
            {
                "name": "Squarespace Calendar (Dent Theatre)",
                "mappings": [
                    {"internal_field": "event_id", "json_path": "id", "transform": "str"},
                    {"internal_field": "title", "json_path": "title", "transform": "identity"},
                    {"internal_field": "full_url", "json_path": "fullUrl", "transform": "identity"},
                    {"internal_field": "start_date_ms", "json_path": "startDate", "transform": "identity"},
                    {"internal_field": "end_date_ms", "json_path": "endDate", "transform": "identity"},
                ],
            },
            {
                "name": "Eventbrite-style",
                "mappings": [
                    {"internal_field": "event_id", "json_path": "id", "transform": "str"},
                    {"internal_field": "title", "json_path": "name", "transform": "identity"},
                    {"internal_field": "full_url", "json_path": "url", "transform": "identity"},
                    {"internal_field": "start_date_ms", "json_path": "start_date", "transform": "iso_to_ms"},
                    {"internal_field": "end_date_ms", "json_path": "end_date", "transform": "iso_to_ms"},
                ],
            },
            {
                "name": "Generic JSON API",
                "mappings": [
                    {"internal_field": "event_id", "json_path": "id", "transform": "str"},
                    {"internal_field": "title", "json_path": "title", "transform": "identity"},
                    {"internal_field": "full_url", "json_path": "url", "transform": "identity"},
                    {"internal_field": "start_date_ms", "json_path": "start_time", "transform": "multiply_1000"},
                    {"internal_field": "end_date_ms", "json_path": "end_time", "transform": "multiply_1000"},
                ],
            },
        ]
    async def fetch_month(
        self,
        source_id: int,
        month_key: str,
    ) -> Tuple[Optional[List[Dict]], bool]:
        """
        Fetch a single month's events from the API.
        Returns (events, was_changed) tuple.
        """
        source = self._get_source(source_id)
        if not source:
            logger.error(f"Source {source_id} not found")
            return [], False

        # Build request URL
        referer = (source.get("referer_template") or DEFAULT_REFERER_TEMPLATE).format(
            month=month_key
        )
        headers = self.request_headers.copy()
        if source.get("request_headers_json"):
            try:
                headers.update(json.loads(source["request_headers_json"]))
            except json.JSONDecodeError:
                logger.error(f"Invalid request_headers_json for source {source_id}")
        headers["Referer"] = referer

        api_url = source["api_template"].format(
            month=month_key, collection_id=source["collection_id"]
        )
        fetch_url = f"{source['base_url']}{api_url}"

        # Check if we already have data for this month
        existing_hash = self._get_month_hash(source_id, month_key)

        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True, headers=headers
            ) as client:
                logger.info(f"Fetching calendar month {month_key}: {fetch_url}")
                resp = await client.get(fetch_url)
                resp.raise_for_status()
                events = resp.json()

                # Compute hash for diff detection
                new_hash = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()

                # Store in DB
                with self._get_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO calendar_months
                            (source_id, month_key, raw_json, json_hash, fetch_url, fetched_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_id, month_key) DO UPDATE SET
                            raw_json = excluded.raw_json,
                            json_hash = excluded.json_hash,
                            fetch_url = excluded.fetch_url,
                            fetched_at = CURRENT_TIMESTAMP,
                            status = 'success',
                            error_message = NULL
                        """,
                        (source_id, month_key, json.dumps(events), new_hash, fetch_url, datetime.datetime.now().isoformat(), "success"),
                    )
                    conn.commit()

                # Parse events into calendar_events table
                parsed_count = self._parse_events(source_id, month_key, events, new_hash != existing_hash)

                was_changed = new_hash != existing_hash
                logger.info(
                    f"Month {month_key}: fetched {len(events)} events, "
                    f"{parsed_count} parsed, changed={was_changed}"
                )
                return events, was_changed

        except Exception as e:
            logger.error(f"Error fetching month {month_key} for source {source_id}: {e}")
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO calendar_months
                        (source_id, month_key, raw_json, json_hash, fetch_url, fetched_at, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, month_key) DO UPDATE SET
                        fetched_at = CURRENT_TIMESTAMP,
                        status = 'failed',
                        error_message = excluded.error_message
                    """,
                    (
                        source_id,
                        month_key,
                        None,
                        hashlib.sha256(month_key.encode()).hexdigest(),
                        fetch_url,
                        datetime.datetime.now().isoformat(),
                        "failed",
                        str(e),
                    ),
                )
                conn.commit()
            return [], False

    def _parse_events(
        self,
        source_id: int,
        month_key: str,
        events: List[Dict],
        force_parse: bool = False,
    ) -> int:
        """
        Parse fetched events into calendar_events table using field mappings.
        Returns count of parsed events.
        """
        if not events:
            return 0

        # Get field mappings for this source
        field_mappings = self.get_field_mappings(source_id)

        # Create mapper
        mapper = FieldMapper(field_mappings)
        validation_errors = mapper.validate()
        if validation_errors:
            logger.error(
                f"Field mapping validation errors for source {source_id}: "
                f"{', '.join(validation_errors)}"
            )
            return 0

        # Get source for base URL
        source = self._get_source(source_id)
        base_url = source.get("base_url", "") if source else ""

        with self._get_conn() as conn:
            month_row = conn.execute(
                "SELECT id FROM calendar_months WHERE source_id = ? AND month_key = ?",
                (source_id, month_key),
            ).fetchone()
            if not month_row:
                logger.error(f"No month record found for source {source_id}, month {month_key}")
                return 0
            month_id = month_row[0]

            # Clean up old events for this month
            conn.execute(
                "DELETE FROM calendar_events WHERE month_id = ?", (month_id,)
            )

            parsed = 0
            skipped = 0
            for event in events:
                # Map raw event to internal schema
                mapped = mapper.map_event(event, source_base_url=base_url)

                event_id = mapped.get("event_id")
                title = mapped.get("title", "Unknown")
                full_url = mapped.get("full_url", "")
                start_ms = mapped.get("start_date_ms")
                end_ms = mapped.get("end_date_ms")

                # Skip events without required fields
                if not event_id:
                    skipped += 1
                    continue

                # Convert to ISO for readability
                start_iso = ""
                end_iso = ""
                if start_ms:
                    start_iso = datetime.datetime.fromtimestamp(
                        start_ms / 1000.0, tz=datetime.timezone.utc
                    ).isoformat()
                if end_ms:
                    end_iso = datetime.datetime.fromtimestamp(
                        end_ms / 1000.0, tz=datetime.timezone.utc
                    ).isoformat()

                # Determine week key (ISO week)
                week_key = ""
                if start_ms:
                    dt = datetime.datetime.fromtimestamp(
                        start_ms / 1000.0, tz=datetime.timezone.utc
                    )
                    week_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                    week_iso = week_key
                else:
                    week_iso = week_key or ""

                conn.execute(
                    """
                    INSERT INTO calendar_events
                        (source_id, month_id, event_id, title, full_url,
                         start_date_ms, end_date_ms, start_date_iso, end_date_iso,
                         week_key, week_guid, body_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (source_id, month_id, event_id, title, full_url,
                        start_ms,
                        end_ms,
                        start_iso,
                        end_iso,
                        week_key,
                        week_iso,  # week_guid (same as week_key for grouping)
                        json.dumps(event),
                    ),
                )
                parsed += 1

            conn.commit()

        logger.info(
            f"Parsed {parsed} events for month {month_key}, skipped {skipped} due to missing fields"
        )
        return parsed

    def _get_source(self, source_id: int) -> Optional[Dict]:
        """Get a single calendar source by ID."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM calendar_sources WHERE id = ?", (source_id,)).fetchone()
            return dict(row) if row else None

    def _get_month_hash(self, source_id: int, month_key: str) -> Optional[str]:
        """Get the stored hash for a month, or None if not fetched."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT json_hash FROM calendar_months WHERE source_id = ? AND month_key = ? ORDER BY fetched_at DESC LIMIT 1",
                (source_id, month_key),
            ).fetchone()
            return row[0] if row else None

    def get_months_for_source(self, source_id: int) -> List[Dict]:
        """Get all fetched months for a source."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM calendar_months
                WHERE source_id = ? AND status = 'success'
                ORDER BY month_key
                """,
                (source_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_events_for_week(
        self, source_id: int, year: int, week_num: int
    ) -> List[Dict]:
        """Get all events for a specific ISO week from a source."""
        week_key = f"{year}-W{week_num:02d}"
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM calendar_events
                WHERE source_id = ? AND week_key = ?
                ORDER BY start_date_ms
                """,
                (source_id, week_key),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_future_weeks(self, source_id: int) -> List[Dict]:
        """
        Get all future weeks with events for a source.
        Returns list of dicts with week info and event counts.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    week_key,
                    COUNT(*) as event_count,
                    MIN(start_date_ms) as first_event_date,
                    MAX(start_date_ms) as last_event_date,
                    GROUP_CONCAT(event_id) as event_ids
                FROM calendar_events
                WHERE source_id = ?
                  AND start_date_ms > ?
                GROUP BY week_key
                ORDER BY first_event_date
                """,
                (source_id, int(now.timestamp() * 1000)),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_month_events(self, source_id: int, month_key: str) -> List[Dict]:
        """Get all parsed events for a specific month."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            month_row = conn.execute(
                "SELECT id FROM calendar_months WHERE source_id = ? AND month_key = ?",
                (source_id, month_key),
            ).fetchone()
            if not month_row:
                return []
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE month_id = ? ORDER BY start_date_ms",
                (month_row[0],),
            ).fetchall()
            return [dict(row) for row in rows]

    def compute_lookahead_months(self, current_date: Optional[datetime.datetime] = None) -> List[str]:
        """
        Compute the list of month keys (MM-YYYY) for the lookahead window.
        Returns months starting from next month for the given number of months.
        """
        if current_date is None:
            current_date = datetime.datetime.now(datetime.timezone.utc)

        months = []
        # Start from next month
        year = current_date.year
        month = current_date.month + 1

        for _ in range(self.lookahead_months):
            if month > 12:
                month = 1
                year += 1
            months.append(f"{month:02d}-{year}")
            month += 1

        return months

    async def fetch_all_lookahead(self, source_id: int) -> Dict[str, Tuple[List[Dict], bool]]:
        """
        Fetch all months in the lookahead window for a source.
        Returns dict of month_key -> (events, was_changed).
        """
        month_keys = self.compute_lookahead_months()
        results = {}

        for month_key in month_keys:
            # Rate limiting: sleep between requests to avoid hammering
            await asyncio.sleep(1.0)
            events, changed = await self.fetch_month(source_id, month_key)
            results[month_key] = (events, changed)

        return results


import asyncio
