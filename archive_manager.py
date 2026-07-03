import logging
import asyncio
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ArchiveManager:
    def __init__(self):
        pass

    def should_archive(self, url: str, archive_domains: list[str]) -> bool:
        """
        Check if the URL belongs to a domain that should be archived.
        """
        if not archive_domains:
            return False
            
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check for exact match or subdomain match
            for d in archive_domains:
                d = d.lower()
                if domain == d or domain.endswith("." + d):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking archive domain for {url}: {e}")
            return False

    async def get_archived_url(self, url: str) -> str:
        """
        Return the archive.ph search URL so the browser can navigate and find the newest snapshot.
        """
        logger.info(f"Delegating {url} to archive.ph search...")
        return f"https://archive.ph/{url}"
