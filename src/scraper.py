"""
Module 1: Visual Ingestion (Scraper)

Downloads unique images from PIEAS web pages and social media,
deduplicates by SHA-256, extracts EXIF geolocation, and stores records in SQLite.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io

from . import database

# Suppress SSL warnings (needed when behind proxy with SSL inspection)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------- CONFIGURATION ----------
# PIEAS proxy
PROXY = {
    "http": "http://172.30.10.11:3128",
    "https": "http://172.30.10.11:3128",
}

SOURCES = [
    # PIEAS official pages (confirmed working)
    "https://www.pieas.edu.pk/proclaims-grid.cshtml?type=NEWS",
    "https://www.pieas.edu.pk/proclaims-grid.cshtml?type=EVENT",
    # PIEAS main site — carousel/hero images
    "https://www.pieas.edu.pk/",
    # PIEAS departments
    "https://www.pieas.edu.pk/departments",
    # Additional pages to verify
    "https://www.pieas.edu.pk/research",
    "https://www.pieas.edu.pk/admissions",
    "https://www.pieas.edu.pk/contact-us",
]

# Social media — code ready, requires API tokens
# Facebook: https://developers.facebook.com/ → Create App → Get Page Access Token
# Instagram: Requires Facebook Graph API with instagram_basic permission
# Once tokens are obtained, set environment variables:
#   $env:FACEBOOK_PAGE_TOKEN = "your_token"
#   $env:INSTAGRAM_TOKEN = "your_token"
SOCIAL_SOURCES = []

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_DELAY = 1          # seconds between source fetches
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


def extract_gps(exif_data: dict) -> tuple[float, float] | None:
    """Extract GPS latitude and longitude from EXIF data if present."""
    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[gps_tag_name] = gps_value

    if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
        return None

    def convert_to_degrees(dms, ref):
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal

    try:
        lat = convert_to_degrees(
            gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N")
        )
        lon = convert_to_degrees(
            gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E")
        )
        return lat, lon
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def extract_exif_geolocation(image_bytes: bytes) -> tuple[float, float] | None:
    """Attempt to extract GPS coordinates from image EXIF metadata."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data:
            return extract_gps(exif_data)
    except Exception:
        pass
    return None


def resize_if_needed(image_bytes: bytes) -> bytes:
    """Resize image to max width if wider than MAX_IMAGE_WIDTH, keeping ratio."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)
            buffer = io.BytesIO()
            fmt = img.format or "PNG"
            img.save(buffer, format=fmt)
            return buffer.getvalue()
    except Exception:
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
            proxies=PROXY,
            verify=False,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"Failed to download {img_url}: {e}")
        return None


def process_source(source_url: str) -> list[tuple[str, str]]:
    """
    Scrape one source URL, download new unique images.
    Extracts EXIF geolocation if available.
    Returns a list of (hash, filepath) for newly saved images.
    """
    logger.info(f"Scraping source: {source_url}")
    new_images = []

    try:
        resp = requests.get(
            source_url,
            headers={"User-Agent": USER_AGENT},
            timeout=DOWNLOAD_TIMEOUT,
            proxies=PROXY,
            verify=False,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch page {source_url}: {e}")
        return new_images

    img_urls = extract_image_urls(source_url, html)
    logger.info(f"Found {len(img_urls)} image(s) on {source_url}")

    for img_url in img_urls:
        time.sleep(0.1)
        img_bytes = download_image(img_url)
        if img_bytes is None:
            continue

        img_hash = compute_sha256(img_bytes)
        if database.image_exists(img_hash):
            logger.info(f"Duplicate skipped: {img_hash[:12]}... from {img_url}")
            continue

        # Extract geolocation
        lat, lon = None, None
        gps_coords = extract_exif_geolocation(img_bytes)
        if gps_coords:
            lat, lon = gps_coords
            logger.info(f"GPS found: ({lat:.5f}, {lon:.5f})")

        img_bytes = resize_if_needed(img_bytes)

        ext = Path(img_url).suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            ext = ".jpg"
        filename = f"{img_hash}{ext}"
        filepath = IMAGES_DIR / filename
        filepath.write_bytes(img_bytes)

        rel_path = str(filepath.relative_to(Path(__file__).resolve().parent.parent))
        database.insert_image_record(img_hash, source_url, rel_path, lat, lon)
        new_images.append((img_hash, rel_path))
        logger.info(f"New image saved: {filename} (source: {source_url})")

    return new_images


def setup_social_sources():
    """Check for social media API tokens and add sources if available."""
    global SOCIAL_SOURCES

    fb_token = os.environ.get("FACEBOOK_PAGE_TOKEN")
    insta_token = os.environ.get("INSTAGRAM_TOKEN")

    if fb_token:
        pieas_fb_id = "PIEASOfficial"  # Replace with actual Facebook page ID/username
        fb_url = (
            f"https://graph.facebook.com/v18.0/{pieas_fb_id}/photos"
            f"?fields=images,created_time&access_token={fb_token}"
        )
        SOCIAL_SOURCES.append(fb_url)
        logger.info("Facebook source added via Graph API")

    if insta_token:
        pieas_insta_id = "pieas_official"  # Replace with actual Instagram username
        insta_url = (
            f"https://graph.instagram.com/{pieas_insta_id}/media"
            f"?fields=media_url,caption,timestamp&access_token={insta_token}"
        )
        SOCIAL_SOURCES.append(insta_url)
        logger.info("Instagram source added via Graph API")


def scrape_all_sources() -> list[tuple[str, str]]:
    """Iterate over all configured sources and scrape them."""
    all_new = []

    for source in SOURCES:
        new = process_source(source)
        all_new.extend(new)
        time.sleep(REQUEST_DELAY)

    setup_social_sources()

    for source in SOCIAL_SOURCES:
        new = process_source(source)
        all_new.extend(new)
        time.sleep(REQUEST_DELAY)

    return all_new


if __name__ == "__main__":
    logger.info("Starting manual scrape...")
    added = scrape_all_sources()
    logger.info(f"Total new images this run: {len(added)}")