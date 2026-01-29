# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "feedparser",
#   "rfeed",
#   "httpx",
#   "pyyaml",
#   "playwright",
#   "readability-lxml",
#   "lxml_html_clean",
#   "beautifulsoup4",
#   "python-multipart"
# ]
# ///

import asyncio
import yaml
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

import rfeed

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
                    pub_date TIMESTAMP,
                    hydrated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migration: Add source_title if it doesn't exist
            try:
                conn.execute("ALTER TABLE articles ADD COLUMN source_title TEXT")
            except sqlite3.OperationalError:
                # Column already exists
                pass

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
            conn.commit()

    def get_article(self, guid):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM articles WHERE guid = ?", (guid,))
            return cursor.fetchone()

    def save_article(self, guid, link, title, description, pub_date, source_title=None, hydrated=0):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO articles (guid, link, title, description, source_title, pub_date, hydrated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (guid, link, title, description, source_title, pub_date, hydrated))
                conn.commit()
        except Exception as e:
            logger.error(f"DB Error: {e}")

    def get_latest_hydrated(self, limit=30):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM articles WHERE hydrated = 1 ORDER BY pub_date DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

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

cache = FeedCache(DB_PATH)

def sync_config_to_db():
    if not os.path.exists("config.yaml"): return
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            for domain in cfg.get("ignore_domains", []): cache.add_global_ignore(domain)
            for f_entry in cfg.get("feeds", []):
                if isinstance(f_entry, str): cache.add_feed(f_entry)
                elif isinstance(f_entry, dict) and 'url' in f_entry:
                    cache.add_feed(f_entry['url'], f_entry.get('ignore_domains', []))
        logger.info("Synced config.yaml to SQLite.")
    except Exception as e: logger.error(f"Sync error: {e}")

browser_instance: Optional[Browser] = None
playwright_manager = None
hydration_semaphore = asyncio.Semaphore(2)

REPO_HANDLERS = {
    "github.com": {"selector": "article.markdown-body", "wait_for": "article.markdown-body"},
    "gitlab.com": {"selector": ".readme-holder .blob-content", "wait_for": ".readme-holder"},
    "codeberg.org": {"selector": ".readme", "wait_for": ".readme"},
    "bitbucket.org": {"selector": "#readme-section", "wait_for": "#readme-section"}
}

async def background_refresh_task():
    while True:
        try:
            feeds = cache.get_feeds()
            global_ignores = cache.get_global_ignores()
            for f in feeds:
                url = f['url']
                local_ignores = json.loads(f['ignore_domains'])
                combined_ignores = global_ignores + local_ignores
                try:
                    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                        resp = await client.get(url)
                        feed_data = feedparser.parse(resp.text)
                        feed_title = getattr(feed_data.feed, 'title', 'Unknown Source')
                        for entry in feed_data.entries:
                            link = getattr(entry, 'link', '')
                            guid = getattr(entry, 'id', link)
                            if any(domain in link for domain in combined_ignores): continue
                            if not cache.get_article(guid):
                                title = getattr(entry, 'title', 'No Title')
                                pd = getattr(entry, 'published_parsed', getattr(entry, 'updated_parsed', None))
                                pub_dt = datetime.datetime(*pd[:6]).isoformat() if pd else datetime.datetime.now().isoformat()
                                cache.save_article(guid, link, title, getattr(entry, 'description', ''), pub_dt, source_title=feed_title, hydrated=0)
                except: pass

            latest_unhydrated = cache.get_unhydrated(20)
            for article in latest_unhydrated:
                content = await hydrate_article(article['link'])
                cache.save_article(article['guid'], article['link'], article['title'], 
                                 content if content else article['description'], 
                                 article['pub_date'], 
                                 source_title=article['source_title'],
                                 hydrated=1 if content else 2)
        except: pass
        await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_instance, playwright_manager
    sync_config_to_db()
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
    if not browser_instance: return None
    async with hydration_semaphore:
        page = None
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."): domain = domain[4:]
            handler = REPO_HANDLERS.get(domain)
            page = await browser_instance.new_page()
            await page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"})
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
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
                    await page.wait_for_selector(handler['wait_for'], timeout=5000)
                    readme_html = await page.inner_html(handler['selector'])
                    if readme_html: return clean_html(readme_html, url)
                except: pass
            content = await page.content()
            doc = Document(content)
            summary_html = doc.summary()
            return clean_html(summary_html, url) if summary_html else None
        except: return None
        finally:
            if page: await page.close()

@app.get("/rss")
async def get_rss(request: Request, username: Optional[str] = Depends(get_feed_user)):
    # Log User-Agent to debug aggregator access
    ua = request.headers.get("user-agent", "Unknown")
    logger.info(f"RSS Feed requested by: {ua}")
    
    latest_articles = cache.get_latest_hydrated(30)
    rss_items = []
    
    # Try to determine current base URL for the 'self' link
    base_url = str(request.base_url).rstrip('/')
    feed_url = f"{base_url}/rss"

    for art in latest_articles:
        desc = art.get('description', '')
        if desc and not desc.startswith("<![CDATA["):
            desc = f"<![CDATA[{desc}]]>"
            
        item = rfeed.Item(
            title=art.get('title', 'No Title'), 
            link=art.get('link', ''), 
            description=desc,
            # GUID should be a permalink if it's a URL
            guid=rfeed.Guid(art.get('guid', art.get('link', '')), isPermaLink=True), 
            pubDate=datetime.datetime.fromisoformat(art['pub_date'])
        )
        
        # Add Dublin Core creator for the source name (better than <author> for names)
        item.extensions.append(rfeed.Extension(
            namespace={"xmlns:dc": "http://purl.org/dc/elements/1.1/"},
            element={"dc:creator": art.get('source_title') or "Unknown Source"}
        ))
        
        rss_items.append(item)
    
    feed = rfeed.Feed(
        title="Unified Hydrated Feed", 
        link=base_url, # Link back to the main site/dashboard
        description="Unified Feed Management with Full Text Hydration", 
        language="en-US",
        lastBuildDate=datetime.datetime.now(), 
        items=rss_items
    )
    
    # Manually inject the necessary namespaces and the Atom self link 
    # since rfeed is a bit limited with top-level attributes
    xml = feed.rss()
    
    # 1. Add Namespaces
    xml = xml.replace('<rss version="2.0">', 
                     '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">')
    
    # 2. Inject Atom Self Link into the channel
    atom_link = f'<atom:link href="{feed_url}" rel="self" type="application/rss+xml" />'
    xml = xml.replace('<channel>', f'<channel>\n    {atom_link}')

    # Ensure XML declaration and UTF-8 encoding
    xml_content = '<?xml version="1.0" encoding="UTF-8" ?>\n' + xml
    
    return Response(
        content=xml_content, 
        media_type="application/rss+xml",
        headers={"Content-Type": "application/rss+xml; charset=utf-8"}
    )

# Admin UI and Management
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(username: str = Depends(get_current_user)):
    feeds = cache.get_feeds()
    ignores = cache.get_global_ignores()
    
    feed_rows = "".join([f"""
        <tr>
            <td>{f['url']}</td>
            <td><code>{f['ignore_domains']}</code></td>
            <td>
                <form action="/admin/delete-feed" method="post" style="display:inline">
                    <input type="hidden" name="url" value="{f['url']}">
                    <button type="submit" class="btn btn-sm btn-danger">Delete</button>
                </form>
            </td>
        </tr>
    """ for f in feeds])

    ignore_rows = "".join([f"""
        <li class="list-group-item d-flex justify-content-between align-items-center">
            {domain}
            <form action="/admin/delete-ignore" method="post" style="display:inline">
                <input type="hidden" name="domain" value="{domain}">
                <button type="submit" class="btn btn-sm btn-outline-danger">x</button>
            </form>
        </li>
    """ for domain in ignores])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RSS Aggregator Admin</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h1 class="mb-4">RSS Feed Management</h1>
            
            <div class="row">
                <div class="col-md-8">
                    <div class="card mb-4 shadow-sm">
                        <div class="card-header bg-primary text-white">Add New Feed</div>
                        <div class="card-body">
                            <form action="/admin/add-feed" method="post">
                                <div class="mb-3">
                                    <label class="form-label">RSS Feed URL</label>
                                    <input type="url" name="url" class="form-control" placeholder="https://example.com/rss" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Ignore Domains (comma separated)</label>
                                    <input type="text" name="ignores" class="form-control" placeholder="ads.com, track.it">
                                </div>
                                <button type="submit" class="btn btn-primary">Add Feed</button>
                            </form>
                        </div>
                    </div>

                    <div class="card mb-4 shadow-sm">
                        <div class="card-header bg-success text-white">Bulk Import Feeds</div>
                        <div class="card-body">
                            <form action="/admin/bulk-import" method="post">
                                <div class="mb-3">
                                    <label class="form-label">Feed URLs (one per line)</label>
                                    <textarea name="urls" class="form-control" rows="5" placeholder="https://site1.com/rss&#10;https://site2.com/feed" required></textarea>
                                </div>
                                <button type="submit" class="btn btn-success">Bulk Add</button>
                            </form>
                        </div>
                    </div>

                    <div class="card shadow-sm">
                        <div class="card-header bg-dark text-white">Active Feeds</div>
                        <div class="card-body p-0">
                            <table class="table table-hover mb-0">
                                <thead class="table-light">
                                    <tr><th>URL</th><th>Per-Feed Ignores</th><th>Action</th></tr>
                                </thead>
                                <tbody>{feed_rows}</tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card shadow-sm">
                        <div class="card-header bg-secondary text-white">Global Ignore List</div>
                        <div class="card-body">
                            <form action="/admin/add-ignore" method="post" class="mb-3">
                                <div class="input-group">
                                    <input type="text" name="domain" class="form-control" placeholder="example.com" required>
                                    <button class="btn btn-outline-primary" type="submit">Add</button>
                                </div>
                            </form>
                            <ul class="list-group">{ignore_rows}</ul>
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)