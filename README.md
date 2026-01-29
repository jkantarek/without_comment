# without_comment

A probably garbage, AI vibed, automated RSS aggregator. It's supposed to hydrate summaries into full-text articles using Headless Firefox (Playwright) and the Mozilla Reader Mode algorithm.

## Features

- **Unified Feed**: Aggregates multiple RSS sources into a single, RSS feed.
- **Background Hydration**: Periodically fetches new articles and uses a headless browser to extract the primary content (preserving headers, images, and links kinda sorta but not perfectly).
- **Slop-sourced Extraction**:
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

/rss to see the feed
/admin to see a barely workable same as it ever was UI to manage your feeds.
