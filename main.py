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
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, Response, Depends, HTTPException, status, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from playwright.async_api import async_playwright, Browser
from readability import Document
from bs4 import BeautifulSoup
from archive_manager import ArchiveManager

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
            conn.commit()

    def get_article(self, guid):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM articles WHERE guid = ?", (guid,))
            return cursor.fetchone()

    def save_article(self, guid, link, title, description, pub_date, source_title=None, feed_url=None, hydrated=0):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO articles (guid, link, title, description, source_title, feed_url, pub_date, hydrated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guid) DO UPDATE SET
                        feed_url = COALESCE(excluded.feed_url, articles.feed_url),
                        source_title = COALESCE(excluded.source_title, articles.source_title),
                        hydrated = CASE WHEN excluded.hydrated > articles.hydrated THEN excluded.hydrated ELSE articles.hydrated END,
                        description = CASE WHEN excluded.hydrated > articles.hydrated THEN excluded.description ELSE articles.description END
                """, (guid, link, title, description, source_title, feed_url, pub_date, hydrated))
                conn.commit()
        except Exception as e:
            logger.error(f"DB Error: {e}")

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
            cursor = conn.execute("SELECT * FROM articles WHERE hydrated = 1 ORDER BY pub_date DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_unhydrated_count(self):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles WHERE hydrated = 0")
            return cursor.fetchone()[0]

    def get_stats(self):
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hydrated = 1 THEN 1 ELSE 0 END) as hydrated,
                    SUM(CASE WHEN hydrated = 0 THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN hydrated = 2 THEN 1 ELSE 0 END) as failed
                FROM articles
            """)
            row = cursor.fetchone()
            return {
                "total": row[0] or 0,
                "hydrated": row[1] or 0,
                "pending": row[2] or 0,
                "failed": row[3] or 0
            }

    def get_unhydrated(self, limit=50):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM articles WHERE hydrated = 0 ORDER BY pub_date DESC LIMIT ?", (limit,))
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

browser_instance: Optional[Browser] = None
playwright_manager = None
hydration_semaphore = asyncio.Semaphore(5)
archive_manager = ArchiveManager()

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
            cache.save_article(link, link, title, description, pub_dt, source_title=f"{feed_title}", feed_url=feed_url, hydrated=0)
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
                    if expanded > 0:
                        count += expanded
                        continue # Skip saving the main newsletter edition entry

                # save_article now handles ON CONFLICT to backfill feed_url/source_title safely
                cache.save_article(guid, link, title, getattr(entry, 'description', ''), pub_dt, source_title=feed_title, feed_url=url, hydrated=0)
                count += 1
            if count > 0:
                logger.info(f"Added {count} new articles from {feed_title}")
    except Exception as e:
        logger.error(f"Error refreshing feed {url}: {e}")

async def hydrate_and_save(article):
    guid = article['guid']
    url = article['link']
    try:
        content = await hydrate_article(url)
        if content:
            logger.info(f"SUCCESS: Hydrated '{article['title']}' ({guid})")
            cache.save_article(guid, url, article['title'], 
                             content, 
                             article['pub_date'], 
                             source_title=article['source_title'],
                             feed_url=article.get('feed_url'),
                             hydrated=1)
        else:
            logger.warning(f"FAILURE: Could not extract content for '{article['title']}' ({guid}). Marking as failed.")
            cache.save_article(guid, url, article['title'], 
                             article['description'], 
                             article['pub_date'], 
                             source_title=article['source_title'],
                             feed_url=article.get('feed_url'),
                             hydrated=2)
    except Exception as e:
        logger.error(f"CRITICAL FAILURE: Error during hydration task for '{article['title']}' ({guid}): {e}")
        cache.save_article(guid, url, article['title'], 
                         article['description'], 
                         article['pub_date'], 
                         source_title=article['source_title'],
                         feed_url=article.get('feed_url'),
                         hydrated=2)

async def background_refresh_task():
    global last_refresh_time
    while True:
        try:
            logger.info("Starting background refresh cycle...")
            feeds = cache.get_feeds()
            global_ignores = cache.get_global_ignores()
            
            # Parallel feed refresh
            await asyncio.gather(*[refresh_feed(f, global_ignores) for f in feeds], return_exceptions=True)

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
                    
                # Process batch in parallel (respecting semaphore)
                await asyncio.gather(*[hydrate_and_save(a) for a in latest_unhydrated], return_exceptions=True)
            
            last_refresh_time = datetime.datetime.now()
            logger.info("Background refresh cycle complete.")
        except Exception as e:
            logger.error(f"Error in background_refresh_task: {e}")
        
        await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_instance, playwright_manager
    cache.backfill_feed_urls()
    playwright_manager = await async_playwright().start()
    browser_instance = await playwright_manager.firefox.launch(headless=True)
    refresh_task = asyncio.create_task(background_refresh_task())
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

async def hydrate_article(url: str) -> Optional[str]:
    if not browser_instance:
        logger.warning("Hydration skipped: Browser instance not initialized.")
        return None
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
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            
            # ... evaluation script ...
            await page.evaluate("""async () => {
                await new Promise((resolve) => {
                    let totalHeight = 0; let distance = 200;
                    let timer = setInterval(() => {
                        window.scrollBy(0, distance); totalHeight += distance;
                        if(totalHeight >= document.body.scrollHeight || totalHeight > 4000){
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
                        return cleaned
                except Exception as e:
                    logger.warning(f"Specialized handler failed for {url}: {e}")
            
            content = await page.content()
            doc = Document(content)
            summary_html = doc.summary()
            if summary_html:
                cleaned = clean_html(summary_html, url)
                logger.info(f"Successfully extracted summary using readability ({len(cleaned)} chars)")
                return cleaned
            
            logger.warning(f"No content extracted for {url}")
            return None
        except Exception as e:
            logger.error(f"Hydration error for {url}: {type(e).__name__}: {str(e)}")
            return None
        finally:
            if page: await page.close()

@app.get("/rss")
async def get_rss(request: Request, username: Optional[str] = Depends(get_feed_user)):
    # Log User-Agent to debug aggregator access
    ua = request.headers.get("user-agent", "Unknown")
    logger.info(f"RSS Feed requested by: {ua}")
    
    latest_articles = cache.get_latest_articles(500)
    rss_items = []
    
    # Try to determine current base URL for the 'self' link
    base_url = str(request.base_url).rstrip('/')
    feed_url = f"{base_url}/rss"

    for art in latest_articles:
        item = rfeed.Item(
            title=art.get('title', 'No Title'), 
            link=art.get('link', ''), 
            # We use our custom CDATA extension instead of the default description field
            # to prevent rfeed from escaping the HTML content.
            extensions=[
                DCCreator(art.get('source_title') or "Unknown Source"),
                CDATA(art.get('description', ''))
            ],
            guid=rfeed.Guid(art.get('guid', art.get('link', ''))), 
            pubDate=datetime.datetime.fromisoformat(art['pub_date'])
        )
        rss_items.append(item)
    
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
    
    refresh_str = last_refresh_time.strftime("%Y-%m-%d %H:%M:%S") if last_refresh_time else "Never"
    
    # Tab activation logic
    mgmt_active = "active" if tab == "management" else ""
    metrics_active = "active" if tab == "metrics" else ""
    mgmt_show = "show active" if tab == "management" else ""
    metrics_show = "show active" if tab == "metrics" else ""

    stats_html = f"""
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card bg-primary text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['total']}</div>
                <small>Total Articles</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-success text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['hydrated']}</div>
                <small>Hydrated</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-warning text-dark text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['pending']}</div>
                <small>Pending Hydration Queue</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card bg-danger text-white text-center p-3 shadow-sm">
                <div class="h4 mb-0">{stats['failed']}</div>
                <small>Failed</small>
            </div>
        </div>
    </div>
    <div class="alert alert-info py-2 shadow-sm">
        <strong>Last Background Refresh:</strong> {refresh_str} (Interval: 10m)
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
            </div>
            
            {stats_html}
            
            <ul class="nav nav-tabs mb-4" id="adminTabs">
                <li class="nav-item">
                    <a class="nav-link {mgmt_active}" href="/admin?tab=management">Management</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {metrics_active}" href="/admin?tab=metrics">Feed Item Metrics</a>
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

@app.get("/admin/preview", response_class=HTMLResponse)
async def admin_preview(guid: str, username: str = Depends(get_current_user)):
    art = cache.get_article(guid)
    if not art:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # SQLite row to dict or handle by index
    # guid, link, title, description, source_title, feed_url, pub_date, hydrated, created_at
    title = art[2]
    content = art[3]
    link = art[1]
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
