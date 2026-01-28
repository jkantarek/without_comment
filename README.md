# without_comment

A high-quality, automated RSS aggregator that hydrates summaries into full-text articles using Headless Firefox (Playwright) and the Mozilla Reader Mode algorithm.

## Features

- **Unified Feed**: Aggregates multiple RSS sources into a single, clean RSS feed.
- **Background Hydration**: Periodically fetches new articles and uses a headless browser to extract the primary content (preserving headers, images, and links).
- **Intelligent Extraction**:
  - Automatically triggers lazy-loading images via simulated scrolling.
  - Converts relative URLs to absolute paths.
  - Special handlers for Git repositories (GitHub, GitLab, Codeberg, Bitbucket) to extract READMEs.
- **SQLite Managed**: Feeds, global ignore lists, and cached articles are all stored in a persistent SQLite database.
- **Admin Dashboard**: A password-protected UI (`/admin`) to manage feeds, bulk import URLs, and configure domain-based filters.
- **Deployment Ready**: Optimized for Docker and Dokploy with `uv` for fast dependency management.

## Setup

### Environment Variables

Required for management:
- `ADMIN_USER`: Username for the admin dashboard.
- `ADMIN_PASS`: Password for the admin dashboard.

Optional for feed protection:
- `FEED_USER`: Username to protect the `/rss` endpoint.
- `FEED_PASS`: Password to protect the `/rss` endpoint.

### Running Locally

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
ADMIN_USER=admin ADMIN_PASS=password ./run.sh
```

### Running with Docker

```bash
docker-compose up -d
```

## Endpoints

- `GET /rss`: The unified, hydrated RSS feed.
- `GET /admin`: The management dashboard (requires auth).
- `GET /feeds`: List configured feeds (requires auth).
- `POST /feeds`: Add a new feed (requires auth).
