import logging
import asyncio
from urllib.parse import urlparse
try:
    import archiveis
except ImportError:
    archiveis = None

logger = logging.getLogger(__name__)

class ArchiveManager:
    def __init__(self):
        if archiveis is None:
            logger.warning("archiveis library not installed. Archiving will be disabled.")

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
        Capture the URL using archive.is and return the archive link.
        """
        if archiveis is None:
            return url

        try:
            logger.info(f"Delegating {url} to archive.is...")
            # archiveis.capture is synchronous/blocking, so run in a thread
            # archiveis.capture returns the URL of the archived page
            archived_url = await asyncio.to_thread(archiveis.capture, url)
            
            if archived_url:
                logger.info(f"Successfully archived {url} -> {archived_url}")
                return archived_url
            else:
                logger.warning(f"archive.is returned no URL for {url}")
                return url
        except Exception as e:
            logger.error(f"Failed to archive {url}: {e}")
            return url
