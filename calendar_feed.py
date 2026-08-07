"""
Calendar RSS Feed Generator

Generates weekly RSS feed items from calendar event data, creating
pseudo-article items with CDATA descriptions containing clickable act links.
"""

import datetime
import logging
from typing import List, Dict, Optional

import rfeed

from calendar_fetcher import CalendarFetcher

logger = logging.getLogger(__name__)

# Default venue base URL for resolving relative links
DEFAULT_VENUE_URL = "https://thedentheatre.com"


class CalendarFeedGenerator:
    """
    Generates RSS feed items grouped by week from calendar event data.
    Each feed item represents one week with all shows aggregated.
    """

    def __init__(
        self,
        fetcher: CalendarFetcher,
        venue_base_url: str = DEFAULT_VENUE_URL,
        feed_title: str = "Event Calendar Digest",
        feed_description: str = "Weekly digest of upcoming shows and events",
    ):
        self.fetcher = fetcher
        self.venue_base_url = venue_base_url
        self.feed_title = feed_title
        self.feed_description = feed_description

    def get_weekly_rss_items(
        self,
        source_id: int,
        include_past: bool = False,
    ) -> List[rfeed.Item]:
        """
        Generate RSS items for all weeks with events.
        Each item is a weekly digest with aggregated shows.
        """
        weeks = self.fetcher.get_all_future_weeks(source_id)
        source = self.fetcher._get_source(source_id)

        if not source:
            logger.error(f"Source {source_id} not found")
            return []

        rss_items = []

        for week in weeks:
            # Skip if no events
            if not week.get("event_count") or week["event_count"] == 0:
                continue

            events = self.fetcher.get_events_for_week(
                source_id,
                int(week["week_key"].split("-")[0]),
                int(week["week_key"].split("-W")[1]),
            )

            if not events:
                continue

            # Build the weekly digest
            week_start_ms = week["first_event_date"]
            week_end_ms = week["last_event_date"]

            # Generate week title
            week_start_dt = datetime.datetime.fromtimestamp(
                week_start_ms / 1000.0, tz=datetime.timezone.utc
            )
            week_label = f"Week of {week_start_dt.strftime('%B %d')}"

            event_count = week["event_count"]
            title = f"{week_label} – {event_count} show{'s' if event_count != 1 else ''}"

            # Build CDATA description with clickable act links
            cdata_html = self._build_week_cdata(source, events, week_label)

            # Build the link (calendar page for the week)
            base_url = source.get("base_url", self.venue_base_url)
            link = f"{base_url}/calendar?view=calendar&month={week_start_dt.strftime('%m-%Y')}"

            # GUID for this week
            week_guid = f"calendar-week-{source_id}-{week_start_dt.strftime('%Y-%m-%d')}"

            # pubDate: Monday of the week
            monday = week_start_dt - datetime.timedelta(days=week_start_dt.weekday())
            pub_date = monday

            # Build RSS item
            item = rfeed.Item(
                title=title,
                link=link,
                description=cdata_html,
                guid=rfeed.Guid(week_guid),
                pubDate=pub_date,
            )
            rss_items.append(item)

        return rss_items

    def _build_week_cdata(self, source: Dict, events: List[Dict], week_label: str) -> str:
        """
        Build CDATA HTML for a weekly digest.
        Returns a CDATA-wrapped HTML string with clickable act links.
        """
        venue_name = source.get("name", "Event Venue")

        # Group events by date for better readability
        events_by_date = {}
        for event in events:
            start_ms = event.get("start_date_ms")
            if not start_ms:
                continue

            dt = datetime.datetime.fromtimestamp(
                start_ms / 1000.0, tz=datetime.timezone.utc
            )
            # Group by date string for grouping
            date_key = dt.strftime("%A, %B %d")
            if date_key not in events_by_date:
                events_by_date[date_key] = []
            events_by_date[date_key].append(event)

        # Build HTML
        html_parts = []
        html_parts.append(f"<h3>{week_label}</h3>")
        html_parts.append(f"<p><em>{venue_name}</em></p>")
        html_parts.append("<table>")

        # Header
        html_parts.append(
            "<tr><th>Date</th><th>Time</th><th>Show</th><th>Tickets</th></tr>"
        )

        # Rows
        for date_key in sorted(events_by_date):
            day_events = events_by_date[date_key]
            for event in sorted(day_events, key=lambda e: e.get("start_date_ms", 0)):
                start_ms = event.get("start_date_ms")
                end_ms = event.get("end_date_ms")
                title = event.get("title", "Unknown Show")
                full_url = event.get("full_url", "")

                # Build absolute URL
                if full_url and not full_url.startswith("http"):
                    abs_url = f"{source.get('base_url', self.venue_base_url)}{full_url}"
                else:
                    abs_url = full_url

                # Format times
                time_str = ""
                if start_ms:
                    start_dt = datetime.datetime.fromtimestamp(
                        start_ms / 1000.0, tz=datetime.timezone.utc
                    )
                    time_str = start_dt.strftime("%I:%M %p")

                # Sanitize title for HTML
                safe_title = self._escape_html(title)

                # Escape date
                safe_date = self._escape_html(date_key)

                # Build row
                html_parts.append(
                    f"<tr>"
                    f"<td>{safe_date}</td>"
                    f"<td>{time_str}</td>"
                    f"<td><a href=\"{abs_url}\" target=\"_blank\">{safe_title}</a></td>"
                    f"<td><a href=\"{abs_url}\" target=\"_blank\">Buy Tickets</a></td>"
                    f"</tr>"
                )

        html_parts.append("</table>")
        html_parts.append("")

        cdata_content = "\n".join(html_parts)
        return f"<![CDATA[{cdata_content}]]>"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def generate_full_feed(
        self,
        source_id: int,
        base_url: str = "",
        existing_rss_items: Optional[List[rfeed.Item]] = None,
    ) -> rfeed.Feed:
        """
        Generate a complete RSS feed for this calendar source.
        Can be merged with existing RSS items from other sources.
        """
        weekly_items = self.get_weekly_rss_items(source_id)

        # Merge with existing items if provided
        all_items = list(existing_rss_items or []) + weekly_items

        # Build feed
        feed_url = f"{base_url}/rss"
        feed = rfeed.Feed(
            title=self.feed_title,
            link=base_url or self.venue_base_url,
            description=self.feed_description,
            language="en-US",
            lastBuildDate=datetime.datetime.now(),
            items=all_items,
        )
        return feed

    def get_week_count(self, source_id: int) -> int:
        """Get the number of future weeks with events."""
        weeks = self.fetcher.get_all_future_weeks(source_id)
        return len([w for w in weeks if w.get("event_count") and w["event_count"] != 0])

    def get_event_count(self, source_id: int) -> int:
        """Get total event count across all future weeks."""
        weeks = self.fetcher.get_all_future_weeks(source_id)
        return sum(w.get("event_count", 0) for w in weeks)
