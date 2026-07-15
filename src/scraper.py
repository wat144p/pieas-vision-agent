"""
Module 1: Visual Ingestion (Scraper)

Downloads unique images from PIEAS web pages, deduplicates by SHA-256,
and stores records in SQLite.
"""

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
import io

from . import database

# ---------- CONFIGURATION ----------
SOURCES = [
    "https://www.pieas.edu.pk/proclaims-grid.cshtml?type=NEWS",
    "https://www.pieas.edu.pk/proclaims-grid.cshtml?type=EVENT",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_DELAY = 3          # seconds between source fetches
MAX_IMAGE_WIDTH = 2048     # resize wider images to save space
DOWNLOAD_TIMEOUT = 30      # seconds

# Directories
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialize DB tables
database.init_db()


def compute_sha256(image_bytes: bytes) -> str:
    """Return the hex digest of the SHA-256 hash of given bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def resize_if_needed(image_bytes: bytes) -> bytes:
    """Resize image to max width if wider than MAX_IMAGE_WIDTH, keeping ratio."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)
            buffer = io.BytesIO()
            # Preserve original format if possible, else PNG
            fmt = img.format or "PNG"
            img.save(buffer, format=fmt)
            return buffer.getvalue()
    except Exception:
        # If anything fails, return original bytes
        pass
    return image_bytes


def extract_image_urls(page_url: str, html: str) -> list:
    """Parse HTML, extract all <img> src attributes, resolve to absolute URLs."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if src:
            absolute = urljoin(page_url, src)
            urls.append(absolute)
    return urls


def download_image(img_url: str) -> bytes | None:
    """Download image bytes from a URL, returning None on failure."""
    try:
        resp = requests.get(
            img_url,
            headers={"User-Agent": USER_AGENT},
            timeout=DOWNLOAD_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Failed to download {img_url}: {e}")
        return None


def process_source(source_url: str) -> list[tuple[str, str]]:
    """
    Scrape one source URL, download new unique images.
    Returns a list of (hash, filepath) for newly saved images.
    """
    logger.info(f"Scraping source: {source_url}")
    new_images = []

    try:
        resp = requests.get(
            source_url,
            headers={"User-Agent": USER_AGENT},
            timeout=DOWNLOAD_TIMEOUT,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch page {source_url}: {e}")
        return new_images

    img_urls = extract_image_urls(source_url, html)
    logger.info(f"Found {len(img_urls)} image(s) on {source_url}")

    for img_url in img_urls:
        # Avoid hammering server
        time.sleep(1)

        img_bytes = download_image(img_url)
        if img_bytes is None:
            continue

        # Deduplication
        img_hash = compute_sha256(img_bytes)
        if database.image_exists(img_hash):
            logger.info(f"Duplicate skipped: {img_hash[:12]}... from {img_url}")
            continue

        # Optional resize
        img_bytes = resize_if_needed(img_bytes)

        # Save to disk
        ext = Path(img_url).suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            ext = ".jpg"
        filename = f"{img_hash}{ext}"
        filepath = IMAGES_DIR / filename
        filepath.write_bytes(img_bytes)

        # Relative path for DB
        rel_path = str(filepath.relative_to(Path(__file__).resolve().parent.parent))

        # Insert record
        database.insert_image_record(img_hash, source_url, rel_path)
        new_images.append((img_hash, rel_path))
        logger.info(f"New image saved: {filename} (source: {source_url})")

    return new_images


def scrape_all_sources() -> list[tuple[str, str]]:
    """Iterate over all configured sources and scrape them.
    Returns combined list of (hash, filepath) for newly saved images.
    """
    all_new = []
    for source in SOURCES:
        new = process_source(source)
        all_new.extend(new)
        # Polite delay between sources
        time.sleep(REQUEST_DELAY)
    return all_new


if __name__ == "__main__":
    # Quick test: run scraper once
    logger.info("Starting manual scrape...")
    added = scrape_all_sources()
    logger.info(f"Total new images this run: {len(added)}")
    