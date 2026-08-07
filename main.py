# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "feedparser",
#   "rfeed",
#   "httpx",
#   "playwright",
#   "readability-lxml",
#   "lxml_html_clean",
#   "beautifulsoup4",
#   "python-multipart",
#   "archiveis"
# ]
# ///

import asyncio
import feedparser
import datetime
import os
import logging
import re
import httpx
import sqlite3
import json
import secrets
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, Response, Depends, HTTPException, status, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from playwright.async_api import async_playwright, Browser
from readability import Document
from bs4 import BeautifulSoup
from archive_manager import ArchiveManager
from calendar_fetcher import CalendarFetcher
from calendar_feed import CalendarFeedGenerator

import rfeed

# Proper rfeed extension classes for maximum compatibility
class DCExtension(rfeed.Extension):
    def get_namespace(self):
        return {"xmlns:dc": "http://purl.org/dc/elements/1.1/"}

class DCCreator(rfeed.Serializable):
    def __init__(self, name):
        rfeed.Serializable.__init__(self)
        self.name = name
    def publish(self, handler):
        self.handler = handler
        self._write_element("dc:creator", self.name)

class AtomExtension(rfeed.Extension):
    def get_namespace(self):
        return {"xmlns:atom": "http://www.w3.org/2005/Atom"}

class AtomSelfLink(rfeed.Serializable):
    def __init__(self, url):
        rfeed.Serializable.__init__(self)
        self.url = url
    def publish(self, handler):
        self.handler = handler
        self._write_element("atom:link", None, {"href": self.url, "rel": "self", "type": "application/rss+xml"})

class CDATA(rfeed.Serializable):
    def __init__(self, text):
        rfeed.Serializable.__init__(self)
        self.text = text
    def publish(self, handler):
        handler.startElement("description", {})
        handler.ignorableWhitespace(f"<![CDATA[{self.text}]]>")
        handler.endElement("description")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "cache.db")
security = HTTPBasic()

# Auth Config
ADMIN_USER = os.environ.get("ADMIN_USER")
ADMIN_PASS = os.environ.get("ADMIN_PASS")
FEED_USER = os.environ.get("FEED_USER")
FEED_PASS = os.environ.get("FEED_PASS")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_USER:
        logger.error("ADMIN_USER not set in environment.")
        raise HTTPException(status_code=500, detail="ADMIN_USER not configured.")
    if not ADMIN_PASS:
        logger.error("ADMIN_PASS not set in environment.")
        raise HTTPException(status_code=500, detail="ADMIN_PASS not configured.")
    
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = ADMIN_USER.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = ADMIN_PASS.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def get_feed_user(credentials: Optional[HTTPBasicCredentials] = Depends(HTTPBasic(auto_error=False))):
    # If feed auth is not configured, allow public access
    if not FEED_USER or not FEED_PASS:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for this feed.",
            headers={"WWW-Authenticate": "Basic"},
        )
        
    is_correct_username = secrets.compare_digest(credentials.username.encode("utf8"), FEED_USER.encode("utf8"))
    is_correct_password = secrets.compare_digest(credentials.password.encode("utf8"), FEED_PASS.encode("utf8"))
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

class FeedCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self.timeout = 30.0
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            # Fallback for network filesystems (NFS/CIFS) that don't support WAL
            logger.warning(f"Could not enable WAL mode (likely a network mount): {e}. Falling back to TRUNCATE.")
            conn.execute("PRAGMA journal_mode=TRUNCATE")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            # Articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    guid TEXT PRIMARY KEY,
                    link TEXT,
                    title TEXT,
                    description TEXT,
                    source_title TEXT,
                    feed_url TEXT,
                    pub_date TIMESTAMP,
                    hydrated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrations
            try:
                conn.execute("ALTER TABLE articles ADD COLUMN source_title TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE articles ADD COLUMN feed_url TEXT")
            except sqlite3.OperationalError: pass
            try:
                conn.execute("ALTER TABLE articles ADD COLUMN retry_count INTEGER DEFAULT 0")
            except sqlite3.OperationalError: pass

            # Feeds table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feeds (
                    url TEXT PRIMARY KEY,
                    ignore_domains TEXT
                )
            """)
            # Global Ignores table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_ignores (
                    domain TEXT PRIMARY KEY
                )
            """)
            # Archive Domains table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_domains (
                    domain TEXT PRIMARY KEY
                )
            """)
            # Share Links table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS share_links (
                    alias TEXT PRIMARY KEY,
                    guid TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(guid) REFERENCES articles(guid)
                )
            """)
            conn.commit()

    def get_article(self, guid):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM articles WHERE guid = ?", (guid,))
            return cursor.fetchone()

    def save_article(self, guid, link, title, description, pub_date, source_title=None, feed_url=None, hydrated=0):
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("SELECT 1 FROM articles WHERE guid = ?", (guid,))
                is_new = cursor.fetchone() is None
                
                conn.execute("""
                    INSERT INTO articles (guid, link, title, description, source_title, feed_url, pub_date, hydrated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guid) DO UPDATE SET
                        feed_url = COALESCE(excluded.feed_url, articles.feed_url),
                        source_title = COALESCE(excluded.source_title, articles.source_title),
                        hydrated = CASE
                            WHEN articles.hydrated = 1 THEN 1
                            WHEN excluded.hydrated = 1 THEN 1
                            WHEN excluded.hydrated = 2 THEN 2
                            ELSE articles.hydrated
                        END,
                        title = CASE
                            WHEN articles.hydrated != 1 AND excluded.hydrated = 1 THEN excluded.title
                            ELSE articles.title
                        END,
                        description = CASE
                            WHEN articles.hydrated != 1 AND excluded.hydrated = 1 THEN excluded.description
                            ELSE articles.description
                        END
                """, (guid, link, title, description, source_title, feed_url, pub_date, hydrated))
                conn.commit()
                return is_new
        except Exception as e:
            logger.error(f"DB Error: {e}")
            return False

    def backfill_feed_urls(self):
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                feeds = conn.execute("SELECT url FROM feeds").fetchall()
                for f in feeds:
                    url = f['url']
                    domain = urlparse(url).netloc
                    if domain.startswith("www."): domain = domain[4:]
                    if domain:
                        conn.execute("UPDATE articles SET feed_url = ? WHERE feed_url IS NULL AND link LIKE ?", 
                                   (url, f"%{domain}%"))
                conn.commit()
                logger.info("Backfilled missing feed_urls for existing articles.")
        except Exception as e:
            logger.error(f"Backfill error: {e}")

    def get_latest_articles(self, limit=500):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT a.*, s.alias as share_alias 
                FROM articles a 
                LEFT JOIN share_links s ON a.guid = s.guid 
                WHERE a.hydrated = 1 ORDER BY a.pub_date DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def mark_as_retrying(self, guid):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE articles SET hydrated=4, retry_count=retry_count+1 WHERE guid=?",
                (guid,)
            )
            conn.commit()

    def promote_deferred_retries(self):
        with self._get_conn() as conn:
            conn.execute("UPDATE articles SET hydrated=3 WHERE hydrated=4")
            conn.commit()

    def get_unhydrated_count(self):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles WHERE hydrated IN (0, 3)")
            return cursor.fetchone()[0]

    def get_failed_articles(self, limit=100):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT guid, title, link, feed_url, pub_date FROM articles WHERE hydrated=2 ORDER BY pub_date DESC LIMIT ?", (limit,))
            return [{"guid": r[0], "title": r[1], "link": r[2], "feed_url": r[3], "pub_date": r[4]} for r in cursor.fetchall()]

    def get_queue_articles(self, limit=100):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT guid, title, link, feed_url, pub_date, hydrated FROM articles WHERE hydrated IN (0, 3, 4) ORDER BY pub_date DESC LIMIT ?", (limit,))
            return [{"guid": r[0], "title": r[1], "link": r[2], "feed_url": r[3], "pub_date": r[4], "status": r[5]} for r in cursor.fetchall()]

    def get_hydrated_articles(self, limit=50):
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT a.guid, a.title, a.link, a.feed_url, a.pub_date, a.description, s.alias 
                FROM articles a 
                LEFT JOIN share_links s ON a.guid = s.guid 
                WHERE a.hydrated=1 ORDER BY a.pub_date DESC LIMIT ?
            """, (limit,))
            return [{"guid": r[0], "title": r[1], "link": r[2], "feed_url": r[3], "pub_date": r[4], "content_html": r[5], "share_alias": r[6]} for r in cursor.fetchall()]

    def get_or_create_share_link(self, guid, url, feed_url, title, pub_date):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT alias FROM share_links WHERE guid=?", (guid,))
            row = cursor.fetchone()
            if row:
                return row[0]
            
            import string, random, re
            from urllib.parse import urlparse
            
            base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            emojis = []
            ranges = [
                (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF),
                (0x1F900, 0x1F9FF), (0x2600, 0x26FF), (0x2700, 0x27BF)
            ]
            for start, end in ranges:
                for i in range(start, end + 1):
                    emojis.append(chr(i))
            keyspace = base64_chars + "".join(emojis)

            def encode_url_safe(data_str):
                if not data_str: return ""
                num = int.from_bytes(str(data_str).encode('utf-8'), 'big')
                if num == 0: return keyspace[0]
                base = len(keyspace)
                res = []
                while num > 0:
                    res.append(keyspace[num % base])
                    num //= base
                return "".join(reversed(res))

            def slugify(text):
                text = str(text or "")
                text = re.sub(r'[^\w\s-]', '', text).strip().lower()
                return re.sub(r'[-\s]+', '-', text)[:50]

            target_domain = urlparse(url).netloc.lower().replace("www.", "") if url else "unknown"
            source_domain = urlparse(feed_url).netloc.lower().replace("www.", "") if feed_url else "unknown"
            story_id = slugify(title) or "article"
            date_str = str(pub_date).split('T')[0].split(' ')[0].replace("-", "") if pub_date else "nodate"
            
            while True:
                random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                items = [target_domain, source_domain, story_id, date_str, random_str]
                alias = "-".join(encode_url_safe(i) for i in items)
                
                cursor = conn.execute("SELECT 1 FROM share_links WHERE alias=?", (alias,))
                if not cursor.fetchone():
                    conn.execute("INSERT INTO share_links (alias, guid) VALUES (?, ?)", (alias, guid))
                    conn.commit()
                    return alias

    def get_stats(self):
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hydrated = 1 THEN 1 ELSE 0 END) as hydrated,
                    SUM(CASE WHEN hydrated = 0 THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN hydrated = 2 THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN hydrated IN (3, 4) THEN 1 ELSE 0 END) as retrying
                FROM articles
            """)
            row = cursor.fetchone()
            return {
                "total": row[0] or 0,
                "hydrated": row[1] or 0,
                "pending": row[2] or 0,
                "failed": row[3] or 0,
                "retrying": row[4] or 0
            }

    def get_unhydrated(self, limit=50):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM articles WHERE hydrated IN (0, 3) ORDER BY pub_date DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def add_feed(self, url, ignore_domains: List[str] = None):
        with self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO feeds (url, ignore_domains) VALUES (?, ?)", 
                        (url, json.dumps(ignore_domains or [])))
            conn.commit()

    def delete_feed(self, url):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM feeds WHERE url = ?", (url,))
            conn.commit()

    def get_feeds(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM feeds")
            return [dict(row) for row in cursor.fetchall()]

    def get_feeds_with_stats(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    f.*,
                    COUNT(a.guid) as total_count,
                    SUM(CASE WHEN a.hydrated = 1 THEN 1 ELSE 0 END) as hydrated_count,
                    SUM(CASE WHEN a.hydrated = 0 THEN 1 ELSE 0 END) as pending_count,
                    SUM(CASE WHEN a.hydrated = 2 THEN 1 ELSE 0 END) as failed_count
                FROM feeds f
                LEFT JOIN articles a ON f.url = a.feed_url
                GROUP BY f.url
            """)
            return [dict(row) for row in cursor.fetchall()]

    def add_global_ignore(self, domain):
        with self._get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO global_ignores (domain) VALUES (?)", (domain,))
            conn.commit()

    def delete_global_ignore(self, domain):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM global_ignores WHERE domain = ?", (domain,))
            conn.commit()

    def get_global_ignores(self) -> List[str]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT domain FROM global_ignores")
            return [row[0] for row in cursor.fetchall()]

    def add_archive_domain(self, domain):
        with self._get_conn() as conn:
            conn.execute("INSERT OR IGNORE INTO archive_domains (domain) VALUES (?)", (domain,))
            conn.commit()

    def delete_archive_domain(self, domain):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM archive_domains WHERE domain = ?", (domain,))
            conn.commit()

    def get_archive_domains(self) -> List[str]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT domain FROM archive_domains")
            return [row[0] for row in cursor.fetchall()]

    def get_feed_item_metrics(self):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            # We'll extract domain from the link using SQL or post-process.
            # Post-processing is easier for complex domain extraction.
            cursor = conn.execute("""
                SELECT 
                    link,
                    MAX(created_at) as last_updated,
                    guid,
                    title
                FROM articles
                GROUP BY guid -- This is just to get all data, we aggregate in Python
                ORDER BY last_updated DESC
            """)
            rows = cursor.fetchall()
            
            metrics = {}
            for row in rows:
                try:
                    domain = urlparse(row['link']).netloc
                    if domain.startswith("www."): domain = domain[4:]
                    if not domain: continue
                    
                    if domain not in metrics:
                        metrics[domain] = {
                            "domain": domain,
                            "count": 0,
                            "last_updated": row['last_updated'],
                            "last_guid": row['guid'],
                            "last_title": row['title']
                        }
                    
                    metrics[domain]["count"] += 1
                    if row['last_updated'] > metrics[domain]["last_updated"]:
                        metrics[domain]["last_updated"] = row['last_updated']
                        metrics[domain]["last_guid"] = row['guid']
                        metrics[domain]["last_title"] = row['title']
                except:
                    continue
            
            return sorted(metrics.values(), key=lambda x: x['last_updated'], reverse=True)

cache = FeedCache(DB_PATH)
last_refresh_time = None
refresh_task: Optional[asyncio.Task] = None

browser_instance: Optional[Browser] = None
playwright_manager = None
hydration_semaphore = asyncio.Semaphore(5)
archive_manager = ArchiveManager()
calendar_fetcher: Optional[CalendarFetcher] = None
calendar_feed_gen: Optional[CalendarFeedGenerator] = None

REPO_HANDLERS = {
    "github.com": {"selector": "article.markdown-body", "wait_for": "article.markdown-body"},
    "gitlab.com": {"selector": ".readme-holder .blob-content", "wait_for": ".readme-holder"},
    "codeberg.org": {"selector": ".readme", "wait_for": ".readme"},
    "bitbucket.org": {"selector": "#readme-section", "wait_for": "#readme-section"},
    "knowablemagazine.org": {"selector": ".article-container", "wait_for": ".article-container"}
}

async def expand_libhunt_newsletter(newsletter_url, client, feed_title, feed_url, combined_ignores, pub_dt):
    try:
        resp = await client.get(newsletter_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        stories = soup.select('li.story')
        expanded_count = 0
        for story in stories:
            if story.get('id') == 'sponsored':
                continue
                
            title_link = story.select_one('a.title')
            if not title_link: continue
            
            link = title_link.get('href', '')
            title = title_link.get_text(strip=True)
            desc_node = story.select_one('p.description')
            description = desc_node.get_text(strip=True) if desc_node else ""
            if "» Learn more" in description:
                description = description.split("» Learn more")[0].strip()
            
            if not link or any(domain in link for domain in combined_ignores):
                continue
            
            # Use the story link as GUID to avoid duplicates across newsletters
            is_new = cache.save_article(link, link, title, description, pub_dt, source_title=f"{feed_title}", feed_url=feed_url, hydrated=0)
            if is_new:
                expanded_count += 1
        return expanded_count
    except Exception as e:
        logger.error(f"Error expanding libhunt newsletter {newsletter_url}: {e}")
        return 0

async def refresh_feed(f, global_ignores):
    url = f['url']
    local_ignores = json.loads(f['ignore_domains'])
    combined_ignores = global_ignores + local_ignores
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            feed_data = feedparser.parse(resp.text)
            feed_title = getattr(feed_data.feed, 'title', 'Unknown Source')
            count = 0
            for entry in feed_data.entries:
                link = getattr(entry, 'link', '')
                guid = getattr(entry, 'id', link)
                if any(domain in link for domain in combined_ignores): continue
                
                title = getattr(entry, 'title', 'No Title')
                pd = getattr(entry, 'published_parsed', getattr(entry, 'updated_parsed', None))
                pub_dt = datetime.datetime(*pd[:6]).isoformat() if pd else datetime.datetime.now().isoformat()
                
                # Special handling for LibHunt newsletter feeds to break out individual stories
                if "libhunt.com" in url.lower() and "/newsletter/" in link.lower() and not link.lower().endswith("/feed"):
                    expanded = await expand_libhunt_newsletter(link, client, feed_title, url, combined_ignores, pub_dt)
                    count += expanded
                    continue # Skip saving the main newsletter edition entry

                # save_article now handles ON CONFLICT to backfill feed_url/source_title safely
                is_new = cache.save_article(guid, link, title, getattr(entry, 'description', ''), pub_dt, source_title=feed_title, feed_url=url, hydrated=0)
                if is_new:
                    count += 1
            if count > 0:
                logger.info(f"Added {count} new articles from {feed_title}")
    except Exception as e:
        logger.error(f"Error refreshing feed {url}: {e}")

ARCHIVE_MAX_RETRIES = 5

async def hydrate_and_save(article):
    guid = article['guid']
    url = article['link']
    try:
        title, content = await asyncio.wait_for(hydrate_article(url), timeout=180.0)
        # Use extracted title if available and current title is just the URL or placeholder
        final_title = title if title and (not article['title'] or article['title'] == url or article['title'] == "Manual Link") else article['title']
        
        if content:
            logger.info(f"SUCCESS: Hydrated '{article['title']}' ({guid})")
            cache.save_article(guid, url, final_title, 
                             content, 
                             article['pub_date'], 
                             source_title=article['source_title'],
                             feed_url=article.get('feed_url'),
                             hydrated=1)
        else:
            archive_domains = cache.get_archive_domains()
            retry_count = article.get('retry_count', 0)
            is_archive = archive_manager.should_archive(url, archive_domains)
            max_retries = ARCHIVE_MAX_RETRIES if is_archive else 1
            
            if retry_count < max_retries:
                prefix = "ARCHIVE RETRY" if is_archive else "STANDARD RETRY"
                logger.warning(f"{prefix} {retry_count + 1}/{max_retries}: Not ready yet for '{article['title']}' ({guid}).")
                cache.mark_as_retrying(guid)
            else:
                logger.warning(f"FAILURE: Could not extract content for '{article['title']}' ({guid}). Marking as failed.")
                cache.save_article(guid, url, final_title, 
                                 article['description'], 
                                 article['pub_date'], 
                                 source_title=article['source_title'],
                                 feed_url=article.get('feed_url'),
                                 hydrated=2)
    except Exception as e:
        archive_domains = cache.get_archive_domains()
        retry_count = article.get('retry_count', 0)
        is_archive = archive_manager.should_archive(url, archive_domains)
        max_retries = ARCHIVE_MAX_RETRIES if is_archive else 1
        
        if retry_count < max_retries:
            prefix = "ARCHIVE RETRY" if is_archive else "STANDARD RETRY"
            logger.error(f"{prefix} {retry_count + 1}/{max_retries}: Exception for '{article['title']}' ({guid}): {type(e).__name__}: {e}")
            cache.mark_as_retrying(guid)
        else:
            logger.exception(f"CRITICAL FAILURE: Error during hydration task for '{article['title']}' ({guid}): {type(e).__name__}: {e}")
            cache.save_article(guid, url, article['title'], 
                             article['description'], 
                             article['pub_date'], 
                             source_title=article['source_title'],
                             feed_url=article.get('feed_url'),
                             hydrated=2)

async def ensure_browser():
    global browser_instance
    if not browser_instance or not browser_instance.is_connected():
        logger.warning("Browser instance is disconnected or null. Restarting Playwright browser...")
        try:
            if browser_instance:
                await browser_instance.close()
        except Exception:
            pass
        browser_instance = await playwright_manager.firefox.launch(headless=True)

async def background_refresh_task():
    global last_refresh_time
    while True:
        try:
            logger.info("Starting background refresh cycle...")
            feeds = cache.get_feeds()
            global_ignores = cache.get_global_ignores()
            
            # Parallel feed refresh
            await asyncio.gather(*[refresh_feed(f, global_ignores) for f in feeds], return_exceptions=True)

            # Promote items that failed in the LAST cycle to be retried in THIS cycle
            cache.promote_deferred_retries()

            # Keep hydrating in batches until the queue is cleared
            while True:
                unhydrated_count = cache.get_unhydrated_count()
                if unhydrated_count == 0:
                    logger.info("Hydration queue is empty.")
                    break
                    
                logger.info(f"Hydration queue: {unhydrated_count} pending. Processing next batch...")
                latest_unhydrated = cache.get_unhydrated(50) 
                if not latest_unhydrated:
                    break
                    
                await ensure_browser()
                
                # Process batch in parallel (respecting semaphore)
                await asyncio.gather(*[hydrate_and_save(a) for a in latest_unhydrated], return_exceptions=True)
            
            # Refresh calendar sources
            if calendar_fetcher:
                try:
                    calendar_sources = calendar_fetcher.get_sources(enabled_only=True)
                    if calendar_sources:
                        logger.info(f"Refreshing {len(calendar_sources)} calendar source(s)...")
                        for source in calendar_sources:
                            await calendar_fetcher.fetch_all_lookahead(source["id"])
                            logger.info(f"Calendar source '{source['name']}' refreshed.")
                except Exception as ce:
                    logger.error(f"Error refreshing calendar sources: {ce}")
            last_refresh_time = datetime.datetime.now()
            logger.info("Background refresh cycle complete.")
        except Exception as e:
            logger.error(f"Error in background_refresh_task: {e}")
        
        await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_instance, playwright_manager, refresh_task, calendar_fetcher, calendar_feed_gen
    cache.backfill_feed_urls()
    playwright_manager = await async_playwright().start()
    browser_instance = await playwright_manager.firefox.launch(headless=True)
    refresh_task = asyncio.create_task(background_refresh_task())
    # Initialize calendar fetcher
    calendar_fetcher = CalendarFetcher(db_path=db_path)
    calendar_feed_gen = CalendarFeedGenerator(calendar_fetcher)
    yield
    refresh_task.cancel()
    if browser_instance: await browser_instance.close()
    if playwright_manager: await playwright_manager.stop()

app = FastAPI(lifespan=lifespan)

def clean_html(html_content: str, base_url: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all(['a', 'img', 'source']):
        for attr in ['src', 'href', 'srcset']:
            if tag.has_attr(attr):
                val = tag[attr]
                if attr == 'srcset':
                    parts = []
                    for part in val.split(','):
                        part = part.strip()
                        if not part: continue
                        subparts = part.split(' ')
                        subparts[0] = urljoin(base_url, subparts[0])
                        parts.append(" ".join(subparts))
                    tag[attr] = ", ".join(parts)
                else: tag[attr] = urljoin(base_url, val)
        for attr in tag.attrs.copy():
            if attr.startswith('data-src') or attr == 'data-original' or attr == 'data-lazy-src':
                tag['src'] = urljoin(base_url, tag[attr])
    for social in soup.select('[class*="social"], [class*="share"], [id*="social"], [id*="share"], .nav-footer, .header-container'):
        social.decompose()
    return str(soup)

async def hydrate_article(url: str) -> Tuple[Optional[str], Optional[str]]:
    if not browser_instance:
        logger.warning("Hydration skipped: Browser instance not initialized.")
        return None, None
    async with hydration_semaphore:
        page = None
        try:
            # Check if we should use an archive.is link
            archive_domains = cache.get_archive_domains()
            if archive_manager.should_archive(url, archive_domains):
                archived_url = await archive_manager.get_archived_url(url)
                if archived_url != url:
                    logger.info(f"Using archived URL for hydration: {archived_url} (was {url})")
                    url = archived_url

            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."): domain = domain[4:]
            logger.info(f"Hydrating: {url} (Domain: {domain})")
            
            handler = REPO_HANDLERS.get(domain)
            page = await browser_instance.new_page()
            await page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"})
            await page.goto(url, wait_until="domcontentloaded", timeout=180000)
            
            if domain in ["archive.ph", "archive.is", "archive.today", "archive.md", "archive.li", "archive.vn", "archive.fo"]:
                logger.info("On archive search page, looking for newest article link...")
                try:
                    await page.wait_for_selector('.TEXT-BLOCK a', timeout=15000)
                    newest_href = await page.evaluate("document.querySelector('.TEXT-BLOCK a').href")
                    if newest_href:
                        logger.info(f"Clicking newest archive snapshot: {newest_href}")
                        await page.goto(newest_href, wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    logger.warning(f"Could not find or navigate to newest article link on archive page: {e}")

            # Use the page title as a fallback
            page_title = await page.title()
            
            # ... evaluation script ...
            await page.evaluate("""async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0; let distance = 200;
                    let timer = setInterval(() => {
                        try {
                            window.scrollBy(0, distance); totalHeight += distance;
                            let scrollHeight = document.body ? document.body.scrollHeight : 0;
                            if(totalHeight >= scrollHeight || totalHeight > 4000){
                                clearInterval(timer); resolve();
                            }
                        } catch (e) {
                            clearInterval(timer); resolve();
                        }
                    }, 100);
                });
            }""")
            await page.evaluate("""() => {
                document.querySelectorAll('img').forEach(img => {
                    ['data-src', 'data-srcset', 'data-original', 'data-lazy-src'].forEach(attr => {
                        if (img.getAttribute(attr)) {
                            if (attr.includes('srcset')) img.srcset = img.getAttribute(attr);
                            else img.src = img.getAttribute(attr);
                        }
                    });
                });
            }""")
            await asyncio.sleep(1)
            
            if handler:
                try:
                    logger.info(f"Using specialized handler for {domain}")
                    await page.wait_for_selector(handler['wait_for'], timeout=5000)
                    readme_html = await page.inner_html(handler['selector'])
                    if readme_html: 
                        cleaned = clean_html(readme_html, url)
                        logger.info(f"Successfully extracted content using {domain} handler ({len(cleaned)} chars)")
                        return page_title, cleaned
                except Exception as e:
                    logger.warning(f"Specialized handler failed for {url}: {e}")
            
            content = await page.content()
            doc = Document(content)
            summary_html = doc.summary()
            extracted_title = doc.title()
            
            # Prefer extracted_title if it looks better than page_title
            final_title = extracted_title if extracted_title and len(extracted_title) > 5 else page_title

            if summary_html:
                cleaned = clean_html(summary_html, url)
                logger.info(f"Successfully extracted summary using readability ({len(cleaned)} chars)")
                return final_title, cleaned
            
            logger.warning(f"No content extracted for {url}")
            return final_title, None
        except Exception as e:
            logger.error(f"Hydration error for {url}: {type(e).__name__}: {str(e)}")
            return None, None
        finally:
            if page: await page.close()

@app.get("/rss")
async def get_rss(request: Request, username: Optional[str] = Depends(get_feed_user)):
    # Log User-Agent to debug aggregator access
    ua = request.headers.get("user-agent", "Unknown")
    logger.info(f"RSS Feed requested by: {ua}")
    
    latest_articles = cache.get_latest_articles(100)
    rss_items = []
    
    # Try to determine current base URL for the 'self' link
    base_url = str(request.base_url).rstrip('/')
    feed_url = f"{base_url}/rss"

    for art in latest_articles:
        link = art.get('link', '')
        domain = "unknown"
        if link:
            try:
                domain = urlparse(link).netloc
                if domain.startswith("www."):
                    domain = domain[4:]
            except:
                pass

        source = art.get('source_title') or "Unknown Source"
        creator_string = f"{domain} via {source}"

        desc = art.get('description', '')
        alias = art.get('share_alias')
        if alias:
            share_html = f'<p style="margin-bottom: 20px;"><a href="{base_url}/s/{alias}" style="display:inline-block;padding:10px 15px;background:#007bff;color:#fff;text-decoration:none;border-radius:5px;">Share Article</a></p><hr>'
            desc = share_html + desc

        item = rfeed.Item(
            title=art.get('title', 'No Title'), 
            link=link, 
            # We use our custom CDATA extension instead of the default description field
            # to prevent rfeed from escaping the HTML content.
            extensions=[
                DCCreator(creator_string),
                CDATA(desc)
            ],
            guid=rfeed.Guid(art.get('guid', art.get('link', ''))), 
            pubDate=datetime.datetime.fromisoformat(art['pub_date'])
        )
        rss_items.append(item)
        
        # Add calendar weekly digest items
        if calendar_feed_gen:
            try:
                calendar_sources = calendar_fetcher.get_sources(enabled_only=True)
                for source in calendar_sources:
                    weekly_items = calendar_feed_gen.get_weekly_rss_items(source["id"])
                    rss_items.extend(weekly_items)
            except Exception as ce:
                logger.error(f"Error generating calendar RSS items: {ce}")
    
    feed = rfeed.Feed(
        title="Unified Hydrated Feed", 
        link=base_url,
        description="Unified Feed Management with Full Text Hydration", 
        language="en-US",
        lastBuildDate=datetime.datetime.now(), 
        items=rss_items,
        extensions=[DCExtension(), AtomExtension(), AtomSelfLink(feed_url)]
    )
    
    xml_content = feed.rss()
    
    return Response(
        content=xml_content, 
        media_type="application/rss+xml",
        headers={"Content-Type": "application/rss+xml; charset=utf-8"}
    )

# Admin UI and Management
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(tab: str = "management", username: str = Depends(get_current_user)):
    feeds = cache.get_feeds_with_stats()
    ignores = cache.get_global_ignores()
    archive_domains = cache.get_archive_domains()
    stats = cache.get_stats()
    metrics = cache.get_feed_item_metrics()
    failed_items = cache.get_failed_articles()
    queue_items = cache.get_queue_articles()
    hydrated_items = cache.get_hydrated_articles()
    
    refresh_str = last_refresh_time.strftime("%Y-%m-%d %H:%M:%S") if last_refresh_time else "Never"
    
    # Tab activation logic
    mgmt_active = "active" if tab == "management" else ""
    metrics_active = "active" if tab == "metrics" else ""
    failed_active = "active" if tab == "failed" else ""
    hydrated_active = "active" if tab == "hydrated" else ""
    
    mgmt_show = "show active" if tab == "management" else ""
    metrics_show = "show active" if tab == "metrics" else ""
    failed_show = "show active" if tab == "failed" else ""
    hydrated_show = "show active" if tab == "hydrated" else ""

    stats_html = f"""
    <div class="row mb-4">
        <div class="col">
            <div class="card bg-primary text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['total']}</div>
                <small>Total Articles</small>
            </div>
        </div>
        <div class="col">
            <div class="card bg-success text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['hydrated']}</div>
                <small>Hydrated</small>
            </div>
        </div>
        <div class="col">
            <div class="card bg-warning text-dark text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['pending']}</div>
                <small>Pending</small>
            </div>
        </div>
        <div class="col">
            <div class="card bg-info text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['retrying']}</div>
                <small>Retrying (archive.ph)</small>
            </div>
        </div>
        <div class="col">
            <div class="card bg-danger text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['failed']}</div>
                <small>Failed</small>
            </div>
        </div>
    </div>
    <div class="alert alert-info py-2 shadow-sm d-flex justify-content-between align-items-center">
        <span><strong>Last Background Refresh:</strong> {refresh_str} (Interval: 10m)</span>
        <form action="/admin/force-refresh" method="post" style="margin:0">
            <button type="submit" class="btn btn-sm btn-warning">Force Refresh</button>
        </form>
    </div>
    """
    
    feed_rows = "".join([f"""
        <tr>
            <td>
                <div class="fw-bold">{f['url']}</div>
                <small class="text-muted"><code>{f['ignore_domains']}</code></small>
            </td>
            <td class="text-center"><span class="badge bg-primary">{f['total_count']}</span></td>
            <td class="text-center"><span class="badge bg-success">{f['hydrated_count'] or 0}</span></td>
            <td class="text-center"><span class="badge bg-warning text-dark">{f['pending_count'] or 0}</span></td>
            <td class="text-center"><span class="badge bg-danger">{f['failed_count'] or 0}</span></td>
            <td>
                <form action="/admin/delete-feed?tab=management" method="post" style="display:inline">
                    <input type="hidden" name="url" value="{f['url']}">
                    <button type="submit" class="btn btn-sm btn-outline-danger">Delete</button>
                </form>
            </td>
        </tr>
    """ for f in feeds])

    ignore_rows = "".join([f"""
        <li class="list-group-item d-flex justify-content-between align-items-center">
            {domain}
            <form action="/admin/delete-ignore?tab=management" method="post" style="display:inline">
                <input type="hidden" name="domain" value="{domain}">
                <button type="submit" class="btn btn-sm btn-outline-danger">x</button>
            </form>
        </li>
    """ for domain in ignores])

    archive_rows = "".join([f"""
        <li class="list-group-item d-flex justify-content-between align-items-center">
            {domain}
            <form action="/admin/delete-archive-domain?tab=management" method="post" style="display:inline">
                <input type="hidden" name="domain" value="{domain}">
                <button type="submit" class="btn btn-sm btn-outline-danger">x</button>
            </form>
        </li>
    """ for domain in archive_domains])

    metric_rows = "".join([f"""
        <tr>
            <td>
                <strong>{m['domain']}</strong>
            </td>
            <td class="text-center">{m['count']}</td>
            <td>{m['last_updated']}</td>
            <td>
                <a href="/admin/preview?guid={m['last_guid']}" class="btn btn-sm btn-outline-info" target="_blank">Preview</a>
                <small class="text-muted d-block text-truncate" style="max-width: 200px;">{m['last_title']}</small>
            </td>
        </tr>
    """ for m in metrics])

    status_map = {0: "Pending", 3: "Retrying", 4: "Deferred Retry"}
    queue_rows = "".join([f"""
        <tr>
            <td>
                <strong>{item['title']}</strong><br>
                <a href="{item['link']}" target="_blank" class="text-muted small">{item['link']}</a>
            </td>
            <td><small>{item['feed_url']}</small></td>
            <td><span class="badge bg-secondary">{status_map.get(item['status'], 'Unknown')}</span></td>
        </tr>
    """ for item in queue_items])

    failed_rows = "".join([f"""
        <tr>
            <td>
                <strong>{item['title']}</strong><br>
                <a href="{item['link']}" target="_blank" class="text-muted small">{item['link']}</a>
            </td>
            <td><small>{item['feed_url']}</small></td>
            <td>
                <form action="/admin/retry-item" method="post" style="margin:0">
                    <input type="hidden" name="guid" value="{item['guid']}">
                    <button type="submit" class="btn btn-sm btn-outline-warning">Retry</button>
                </form>
            </td>
        </tr>
    """ for item in failed_items])

    hydrated_html = "".join([f"""
        <div class="card mb-4 shadow-sm">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0"><a href="{item['link']}" target="_blank" style="text-decoration:none; color:inherit;">{item['title']}</a></h5>
                <small class="text-muted">{item['feed_url']}</small>
            </div>
            <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                {item['content_html']}
            </div>
            <div class="card-footer text-muted small d-flex gap-2 align-items-center">
                <a href="{item['link']}" target="_blank" class="btn btn-sm btn-outline-secondary">View Original URL</a>
                <a href="/admin/preview?guid={item['guid']}" target="_blank" class="btn btn-sm btn-outline-info">Full Preview</a>
                {
                    f'<a href="/s/{item["share_alias"]}" target="_blank" class="btn btn-sm btn-success">Public Share Link</a>' 
                    if item.get('share_alias') else 
                    f'<form action="/admin/generate-share-link" method="post" style="margin:0;"><input type="hidden" name="guid" value="{item["guid"]}"><input type="hidden" name="redirect_to" value="/admin?tab=hydrated"><button type="submit" class="btn btn-sm btn-outline-success">Generate Share Link</button></form>'
                }
            </div>
        </div>
    """ for item in hydrated_items])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RSS Aggregator Admin</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .table th {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05rem; }}
        </style>
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1>RSS Feed Management</h1>
                <a href="/rss" class="btn btn-outline-secondary" target="_blank">View RSS Feed</a>
                <a href="/admin/calendars" class="btn btn-outline-primary">Calendar Sources</a>
            </div>
            
            {stats_html}
            
            <ul class="nav nav-tabs mb-4" id="adminTabs">
                <li class="nav-item">
                    <a class="nav-link {mgmt_active}" href="/admin?tab=management">Management</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {metrics_active}" href="/admin?tab=metrics">Feed Item Metrics</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {failed_active}" href="/admin?tab=failed">Queue / Failed ({len(failed_items) + len(queue_items)})</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {hydrated_active}" href="/admin?tab=hydrated">Hydrated Items</a>
                </li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane fade {mgmt_show}" id="feeds">
                    <div class="row">
                        <div class="col-md-9">
                            <div class="card mb-4 shadow-sm">
                                <div class="card-header bg-primary text-white">Add New Feed</div>
                                <div class="card-body">
                                    <form action="/admin/add-feed?tab=management" method="post">
                                        <div class="row">
                                            <div class="col-md-7">
                                                <input type="url" name="url" class="form-control" placeholder="https://example.com/rss" required>
                                            </div>
                                            <div class="col-md-3">
                                                <input type="text" name="ignores" class="form-control" placeholder="ads.com, track.it">
                                            </div>
                                            <div class="col-md-2">
                                                <button type="submit" class="btn btn-primary w-100">Add</button>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>

                            <div class="card shadow-sm mb-4">
                                <div class="card-header bg-dark text-white">Active Feeds</div>
                                <div class="card-body p-0">
                                    <table class="table table-hover mb-0 align-middle">
                                        <thead class="table-light">
                                            <tr>
                                                <th>Feed URL & Ignores</th>
                                                <th class="text-center">Total</th>
                                                <th class="text-center">Hydrated</th>
                                                <th class="text-center">Pending</th>
                                                <th class="text-center">Failed</th>
                                                <th>Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>{feed_rows}</tbody>
                                    </table>
                                </div>
                            </div>

                            <div class="card mb-4 shadow-sm">
                                <div class="card-header bg-success text-white">Bulk Import Feeds</div>
                                <div class="card-body">
                                    <form action="/admin/bulk-import?tab=management" method="post">
                                        <div class="mb-3">
                                            <textarea name="urls" class="form-control" rows="3" placeholder="https://site1.com/rss&#10;https://site2.com/feed" required></textarea>
                                        </div>
                                        <button type="submit" class="btn btn-success btn-sm">Bulk Add</button>
                                    </form>
                                </div>
                            </div>

                            <div class="card mb-4 shadow-sm">
                                <div class="card-header bg-warning text-dark">Quick Add Individual Links</div>
                                <div class="card-body">
                                    <form action="/admin/add-links?tab=management" method="post">
                                        <div class="mb-3">
                                            <textarea name="links" class="form-control" rows="3" placeholder="https://example.com/article1&#10;https://example.com/article2" required></textarea>
                                            <div class="form-text">Paste one or more URLs to add them to the hydration queue.</div>
                                        </div>
                                        <button type="submit" class="btn btn-warning btn-sm">Add to Digest</button>
                                    </form>
                                </div>
                            </div>
                        </div>

                        <div class="col-md-3">
                            <div class="card shadow-sm mb-4">
                                <div class="card-header bg-secondary text-white">Global Ignore List</div>
                                <div class="card-body">
                                    <form action="/admin/add-ignore?tab=management" method="post" class="mb-3">
                                        <div class="input-group input-group-sm">
                                            <input type="text" name="domain" class="form-control" placeholder="example.com" required>
                                            <button class="btn btn-outline-primary" type="submit">Add</button>
                                        </div>
                                    </form>
                                    <ul class="list-group list-group-flush">{ignore_rows}</ul>
                                </div>
                            </div>

                            <div class="card shadow-sm">
                                <div class="card-header bg-info text-white">Archive Domains</div>
                                <div class="card-body">
                                    <form action="/admin/add-archive-domain?tab=management" method="post" class="mb-3">
                                        <div class="input-group input-group-sm">
                                            <input type="text" name="domain" class="form-control" placeholder="nytimes.com" required>
                                            <button class="btn btn-outline-primary" type="submit">Add</button>
                                        </div>
                                    </form>
                                    <ul class="list-group list-group-flush">{archive_rows}</ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="tab-pane fade {metrics_show}" id="metrics">
                    <div class="card shadow-sm">
                        <div class="card-header bg-secondary text-white">Feed Item Metrics (Aggregated by Domain)</div>
                        <div class="card-body p-0">
                            <table class="table table-striped table-hover mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Domain</th>
                                        <th class="text-center">Article Count</th>
                                        <th>Last Updated</th>
                                        <th>Latest Item</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {metric_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="tab-pane fade {failed_show}" id="failed">
                    <div class="card shadow-sm mb-4">
                        <div class="card-header bg-secondary text-white">
                            <span class="mb-0">Pending / Retrying Queue ({len(queue_items)})</span>
                        </div>
                        <div class="card-body p-0">
                            <table class="table table-striped table-hover mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Title / URL</th>
                                        <th>Feed</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {queue_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div class="card shadow-sm">
                        <div class="card-header bg-danger text-white d-flex justify-content-between align-items-center">
                            <span class="mb-0">Failed Items ({len(failed_items)})</span>
                            <form action="/admin/retry-all-failed" method="post" style="margin:0">
                                <button type="submit" class="btn btn-sm btn-light text-danger">Retry All Failed</button>
                            </form>
                        </div>
                        <div class="card-body p-0">
                            <table class="table table-striped table-hover mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Title / URL</th>
                                        <th>Feed</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {failed_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="tab-pane fade {hydrated_show}" id="hydrated">
                    <div class="container-fluid p-0">
                        {hydrated_html}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.post("/admin/add-feed")
async def admin_add_feed(url: str = Form(...), ignores: str = Form(""), username: str = Depends(get_current_user)):
    ignore_list = [i.strip() for i in ignores.split(",") if i.strip()]
    cache.add_feed(url, ignore_list)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/bulk-import")
async def admin_bulk_import(urls: str = Form(...), username: str = Depends(get_current_user)):
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]
    for url in url_list:
        cache.add_feed(url)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/add-links")
async def admin_add_links(links: str = Form(...), username: str = Depends(get_current_user)):
    link_list = [l.strip() for l in links.split("\n") if l.strip()]
    now = datetime.datetime.now().isoformat()
    for link in link_list:
        # Use link as GUID for manual entries
        cache.save_article(link, link, "Manual Link", "", now, source_title="Manual", hydrated=0)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/delete-feed")
async def admin_delete_feed(url: str = Form(...), username: str = Depends(get_current_user)):
    cache.delete_feed(url)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/add-ignore")
async def admin_add_ignore(domain: str = Form(...), username: str = Depends(get_current_user)):
    cache.add_global_ignore(domain)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/delete-ignore")
async def admin_delete_ignore(domain: str = Form(...), username: str = Depends(get_current_user)):
    cache.delete_global_ignore(domain)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/add-archive-domain")
async def admin_add_archive_domain(domain: str = Form(...), username: str = Depends(get_current_user)):
    cache.add_archive_domain(domain)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/delete-archive-domain")
async def admin_delete_archive_domain(domain: str = Form(...), username: str = Depends(get_current_user)):
    cache.delete_archive_domain(domain)
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/retry-item")
async def admin_retry_item(guid: str = Form(...), username: str = Depends(get_current_user)):
    with cache._get_conn() as conn:
        conn.execute("UPDATE articles SET hydrated=0, retry_count=0 WHERE guid=?", (guid,))
        conn.commit()
    return RedirectResponse(url="/admin?tab=failed", status_code=303)

@app.post("/admin/retry-all-failed")
async def admin_retry_all_failed(username: str = Depends(get_current_user)):
    with cache._get_conn() as conn:
        conn.execute("UPDATE articles SET hydrated=0, retry_count=0 WHERE hydrated=2")
        conn.commit()
    return RedirectResponse(url="/admin?tab=failed", status_code=303)

@app.post("/admin/generate-share-link")
async def admin_generate_share_link(guid: str = Form(...), redirect_to: str = Form("/admin?tab=hydrated"), username: str = Depends(get_current_user)):
    art = cache.get_article(guid)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
        
    cache.get_or_create_share_link(
        guid=guid,
        url=art[1],
        feed_url=art[5],
        title=art[2],
        pub_date=art[6]
    )
    return RedirectResponse(url=redirect_to, status_code=303)

@app.post("/admin/force-refresh")
async def admin_force_refresh(username: str = Depends(get_current_user)):
    global refresh_task
    if refresh_task and not refresh_task.done():
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
    refresh_task = asyncio.create_task(background_refresh_task())
    logger.info("Background refresh task restarted by admin.")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/admin/calendars", response_class=HTMLResponse)
async def admin_calendars(username: str = Depends(get_current_user)):
    """View configured calendar sources."""
    sources = calendar_fetcher.get_sources(enabled_only=False) if calendar_fetcher else []
    
    calendar_form = ""
    sources = calendar_fetcher.get_sources(enabled_only=False) if calendar_fetcher else []
    
    # Build rows
    rows = ""
    for s in sources:
        # Count months
        months = calendar_fetcher.get_months_for_source(s["id"]) if calendar_fetcher else []
        status_badge = "<span class='badge bg-success'>Active</span>" if s["enabled"] else "<span class='badge bg-secondary'>Disabled</span>"
        rows += f"""
        <tr>
            <td><strong>{s["name"]}</strong></td>
            <td><code>{s["base_url"]}</code></td>
            <td><code>{s["collection_id"]}</code></td>
            <td>{len(months)} month(s)</td>
            <td>{status_badge}</td>
            <td>
                <form method="POST" action="/admin/refresh-calendar" class="d-inline">
                    <input type="hidden" name="source_id" value="{s['id']}">
                    <button type="submit" class="btn btn-sm btn-outline-primary">Refresh</button>
                </form>
                <button type="button" class="btn btn-sm btn-outline-info" data-bs-toggle="modal" data-bs-target="#mapCalendarModal{s['id']}">Mappings</button>
                <form method="POST" action="/admin/toggle-calendar" class="d-inline">
                    <input type="hidden" name="source_id" value="{s['id']}">
                    <button type="submit" class="btn btn-sm btn-outline-secondary">{"Disable" if s["enabled"] else "Enable"}</button>
                </form>
                <form method="POST" action="/admin/delete-calendar" class="d-inline">
                    <input type="hidden" name="source_id" value="{s['id']}">
                    <button type="submit" class="btn btn-sm btn-outline-danger">Delete</button>
                </form>
            </td>
        </tr>
        """
    # Generate field mapping modals for each source
    mapping_modals = ""
    for s in sources:
        mappings = calendar_fetcher.get_field_mappings(s["id"])
        mapping_rows = ""
        for m in mappings:
            mapping_rows += f"""
            <tr>
                <td>{m['internal_field']}</td>
                <td><code>{m['json_path']}</code></td>
                <td>{m['transform']}</td>
                <td>
                    <form method="POST" action="/admin/delete-calendar-field">
                        <input type="hidden" name="source_id" value="{s['id']}">
                        <input type="hidden" name="internal_field" value="{m['internal_field']}">
                        <button type="submit" class="btn btn-sm btn-outline-danger">×</button>
                    </form>
                </td>
            </tr>
            """
        
        if not mapping_rows:
            mapping_rows = "<tr><td colspan=4>No mappings configured</td></tr>"
        
        mapping_modals += f"""
        <div class="modal fade" id="mapCalendarModal{s['id']}" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Field Mappings – {s['name']}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p class="text-muted">Map JSON fields from the API to internal fields. All fields are required.</p>
                        
                        <div class="card mb-3">
                            <div class="card-body">
                                <form method="POST" action="/admin/set-calendar-mapping">
                                    <input type="hidden" name="source_id" value="{s['id']}">
                                    <div class="row g-3">
                                        <div class="col-md-3">
                                            <label class="form-label">Internal Field</label>
                                            <select name="internal_field" class="form-select form-select-sm" required>
                                                <option value="">Select...</option>
                                                <option value="event_id">event_id</option>
                                                <option value="title">title</option>
                                                <option value="full_url">full_url</option>
                                                <option value="start_date_ms">start_date_ms</option>
                                                <option value="end_date_ms">end_date_ms</option>
                                            </select>
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label">JSON Path</label>
                                            <input type="text" name="json_path" class="form-control form-control-sm" placeholder="id, structuredContent.startDate" required>
                                        </div>
                                        <div class="col-md-3">
                                            <label class="form-label">Transform</label>
                                            <select name="transform" class="form-select form-select-sm">
                                                <option value="identity" selected>identity (raw)</option>
                                                <option value="str">str</option>
                                                <option value="int">int</option>
                                                <option value="float">float</option>
                                                <option value="trim">trim</option>
                                                <option value="multiply_1000">multiply_1000</option>
                                                <option value="iso_to_ms">iso_to_ms</option>
                                            </select>
                                        </div>
                                        <div class="col-md-2 align-self-end">
                                            <button type="submit" class="btn btn-primary btn-sm w-100">Save</button>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </div>
                        
                        <h6 class="mt-3">Current Mappings</h6>
                        <table class="table table-sm table-bordered">
                            <thead><tr><th>Field</th><th>JSON Path</th><th>Transform</th><th></th></tr></thead>
                            <tbody>
                                {mapping_rows}
                            </tbody>
                        </table>
                        
                        <div class="alert alert-info mt-3">
                            <small>
                                <strong>Preset Examples:</strong><br>
                                Squarespace: <code>id, title, fullUrl, startDate, endDate</code><br>
                                Eventbrite: <code>id, name, url, start_date, end_date</code><br>
                                Generic: <code>id, title, url, start_time, end_time</code>
                            </small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
        """
    
    calendar_modal = """
    <div class="modal fade" id="addCalendarModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <form method="POST" action="/admin/add-calendar">
                    <div class="modal-header">
                        <h5 class="modal-title">Add Calendar Source</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Name</label>
                            <input type="text" name="name" class="form-control" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Base URL</label>
                            <input type="text" name="base_url" class="form-control" placeholder="https://thedentheatre.com" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">API Template</label>
                            <input type="text" name="api_template" class="form-control" placeholder="/api/open/GetItemsByMonth?month={{month}}&collectionId={{collection_id}}">
                            <small class="text-muted">Use {{month}} and {{collection_id}} as placeholders</small>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Collection ID</label>
                            <input type="text" name="collection_id" class="form-control" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Referer Template (optional)</label>
                            <input type="text" name="referer_template" class="form-control" placeholder="https://thedentheatre.com/calendar?view=calendar&month={{month}}">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Lookahead Months</label>
                            <input type="number" name="lookahead_months" class="form-control" value="3" min="1" max="12">
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary">Add Calendar</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    {mapping_modals}
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """
    
    html = f"""
    <html>
    <head>
        <title>Calendar Sources</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-4">
            <div class="mb-3">
                <a href="/admin" class="btn btn-outline-secondary">&larr; Back to Admin</a>
            </div>
            <div class="card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5>Calendar Sources</h5>
                    <button type="button" class="btn btn-primary btn-sm" data-bs-toggle="modal" data-bs-target="#addCalendarModal">Add Calendar</button>
                </div>
                <div class="card-body">
                    {calendar_form}
                    <table class="table table-striped">
                        <thead><tr><th>Name</th><th>Base URL</th><th>Collection ID</th><th>Months</th><th>Status</th><th>Actions</th></tr></thead>
                        <tbody>
{rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
{calendar_modal}
    </body>
    </html>
    """ 
    return html


@app.post("/admin/add-calendar")
async def admin_add_calendar(
    name: str = Form(...),
    base_url: str = Form(...),
    api_template: str = Form(...),
    collection_id: str = Form(...),
    referer_template: str = Form(""),
    lookahead_months: int = Form(3),
    username: str = Depends(get_current_user),
):
    """Add a new calendar source."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    calendar_fetcher.add_source(
        name=name,
        base_url=base_url,
        api_template=api_template,
        collection_id=collection_id,
        referer_template=referer_template,
        lookahead_months=lookahead_months,
    )
    logger.info(f"Calendar source added: {name}")
    return RedirectResponse(url="/admin/calendars", status_code=303)


@app.post("/admin/delete-calendar")
async def admin_delete_calendar(
    source_id: int = Form(...),
    username: str = Depends(get_current_user),
):
    """Delete a calendar source."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    calendar_fetcher.delete_source(source_id)
    logger.info(f"Calendar source {source_id} deleted")
    return RedirectResponse(url="/admin/calendars", status_code=303)


@app.post("/admin/toggle-calendar")
async def admin_toggle_calendar(
    source_id: int = Form(...),
    username: str = Depends(get_current_user),
):
    """Toggle a calendar source on/off."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    source = calendar_fetcher._get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    calendar_fetcher.enable_source(source_id) if not source["enabled"] else calendar_fetcher.disable_source(source_id)
    logger.info(f"Calendar source {source_id} {'enabled' if source['enabled'] else 'disabled'}")
    return RedirectResponse(url="/admin/calendars", status_code=303)


@app.post("/admin/refresh-calendar")
async def admin_refresh_calendar(
    source_id: int = Form(...),
    username: str = Depends(get_current_user),
):
    """Force refresh a calendar source."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    await calendar_fetcher.fetch_all_lookahead(source_id)
    logger.info(f"Calendar source {source_id} refreshed")
    return RedirectResponse(url="/admin/calendars", status_code=303)
@app.get("/admin/preview", response_class=HTMLResponse)
@app.post("/admin/set-calendar-mapping")
async def admin_set_calendar_mapping(
    source_id: int = Form(...),
    internal_field: str = Form(...),
    json_path: str = Form(...),
    transform: str = Form(default="identity"),
    username: str = Depends(get_current_user),
):
    """Set or update a field mapping for a calendar source."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    
    # Validate required fields
    required_fields = calendar_fetcher.get_required_fields()
    if internal_field not in required_fields:
        return RedirectResponse(url="/admin/calendars", status_code=303)
    
    # Check if source exists
    source = calendar_fetcher._get_source(source_id)
    if not source:
        return RedirectResponse(url="/admin/calendars", status_code=303)
    
    calendar_fetcher.set_field_mapping(source_id, internal_field, json_path, transform)
    logger.info(f"Calendar mapping updated: source {source_id}, field {internal_field} <- {json_path} ({transform})")
    return RedirectResponse(url="/admin/calendars", status_code=303)


@app.post("/admin/delete-calendar-field")
async def admin_delete_calendar_field(
    source_id: int = Form(...),
    internal_field: str = Form(...),
    username: str = Depends(get_current_user),
):
    """Delete a field mapping for a calendar source."""
    if not calendar_fetcher:
        raise HTTPException(status_code=500, detail="Calendar fetcher not initialized")
    
    # Check if source exists
    source = calendar_fetcher._get_source(source_id)
    if not source:
        return RedirectResponse(url="/admin/calendars", status_code=303)
    
    calendar_fetcher.delete_field_mapping(source_id, internal_field)
    logger.info(f"Calendar mapping deleted: source {source_id}, field {internal_field}")
    return RedirectResponse(url="/admin/calendars", status_code=303)


@app.get("/admin/preview", response_class=HTMLResponse)
async def admin_preview(guid: str, username: str = Depends(get_current_user)):
    art = cache.get_article(guid)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
        
    with cache._get_conn() as conn:
        c = conn.execute("SELECT alias FROM share_links WHERE guid=?", (guid,))
        row = c.fetchone()
        alias = row[0] if row else None
    
    # SQLite row to dict or handle by index
    # guid, link, title, description, source_title, feed_url, pub_date, hydrated, created_at
    title = art[2]
    content = art[3]
    link = art[1]
    
    if alias:
        share_btn = f'<a href="/s/{alias}" class="btn btn-outline-success" target="_blank">Public Share Link</a>'
    else:
        share_btn = f"""
            <form action="/admin/generate-share-link" method="post" style="display:inline; margin:0;">
                <input type="hidden" name="guid" value="{guid}">
                <input type="hidden" name="redirect_to" value="/admin/preview?guid={guid}">
                <button type="submit" class="btn btn-outline-success">Generate Share Link</button>
            </form>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Preview: {title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <div class="mb-4">
                <a href="/admin" class="btn btn-outline-secondary">&larr; Back to Admin</a>
                <a href="{link}" class="btn btn-outline-primary" target="_blank">Original Link</a>
                {share_btn}
            </div>
            <div class="card shadow-sm">
                <div class="card-body">
                    <h1 class="mb-4">{title}</h1>
                    <hr>
                    <div class="article-content">
                        {content}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.get("/s/{alias}", response_class=HTMLResponse)
async def public_share_link(alias: str):
    with cache._get_conn() as conn:
        cursor = conn.execute("SELECT guid FROM share_links WHERE alias=?", (alias,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Share link not found")
        guid = row[0]
        
    art = cache.get_article(guid)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
        
    title = art[2]
    content = art[3]
    link = art[1]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <div class="mb-4 text-center">
                <a href="{link}" class="btn btn-primary" target="_blank">View Original Source</a>
            </div>
            <div class="card shadow-sm">
                <div class="card-body">
                    <h1 class="mb-4 text-center">{title}</h1>
                    <hr>
                    <div class="article-content" style="max-width: 800px; margin: 0 auto;">
                        {content}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
