"""
Module 1: Visual Ingestion (Scraper)

Downloads unique images from PIEAS web pages and social media,
deduplicates by SHA-256, extracts EXIF geolocation, and stores records in SQLite.
"""

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import instaloader

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
    # Additional pages
    "https://www.pieas.edu.pk/research",
    "https://www.pieas.edu.pk/admissions",
    "https://www.pieas.edu.pk/aboutcampuslife.cshtml",
    "https://www.pieas.edu.pk/students.cshtml",
    "https://www.pieas.edu.pk/international-students.cshtml",
    "https://www.pieas.edu.pk/rsd/",
]

# Social media sources
# Instagram: public scraping via instaloader (no API token needed)
# Facebook: requires Page Access Token — set $env:FACEBOOK_PAGE_TOKEN
INSTAGRAM_HANDLE = "pieas.official"
INSTAGRAM_POST_LIMIT = 20  # Number of recent posts to check
SOCIAL_SOURCES = []

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_DELAY = 1          # seconds between source fetches
SCAN_PASSES = 5            # number of times to fetch each page (catches dynamic content)
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


def process_source(source_url: str, passes: int = SCAN_PASSES) -> list[tuple[str, str]]:
    """
    Scrape one source URL, download new unique images.
    Fetches the page multiple times to catch dynamic/rotating content.
    Extracts EXIF geolocation if available.
    Returns a list of (hash, filepath) for newly saved images.
    """
    logger.info(f"Scraping source: {source_url} ({passes} passes)")
    new_images = []

    # Collect all image URLs across multiple page fetches
    all_img_urls = set()

    for attempt in range(passes):
        try:
            if attempt > 0:
                time.sleep(1)  # Brief gap between passes for rotation

            resp = requests.get(
                source_url,
                headers={"User-Agent": USER_AGENT},
                timeout=DOWNLOAD_TIMEOUT,
                proxies=PROXY,
                verify=False,
            )
            resp.raise_for_status()

            img_urls = extract_image_urls(source_url, resp.text)
            before = len(all_img_urls)
            all_img_urls.update(img_urls)
            new_in_pass = len(all_img_urls) - before
            logger.info(f"  Pass {attempt + 1}: found {len(img_urls)} URLs, {new_in_pass} new")

        except Exception as e:
            logger.error(f"Failed to fetch page {source_url} (pass {attempt + 1}): {e}")
            continue

    logger.info(f"Total unique image URLs across {passes} passes: {len(all_img_urls)}")

    # Download and process each unique image
    for img_url in all_img_urls:
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
    SOCIAL_SOURCES = []  # Reset each run

    fb_token = os.environ.get("FACEBOOK_PAGE_TOKEN")

    if fb_token:
        pieas_fb_id = "PIEAS.official.pk"
        fb_url = (
            f"https://graph.facebook.com/v18.0/{pieas_fb_id}/photos"
            f"?fields=images,created_time&access_token={fb_token}"
        )
        SOCIAL_SOURCES.append(("facebook", fb_url))
        logger.info("Facebook source added via Graph API")


def scrape_instagram() -> list[tuple[str, str]]:
    """
    Scrape recent images from the PIEAS Instagram account using instaloader.
    No API token required for public accounts.
    Returns a list of (hash, filepath) for newly saved images.
    """
    logger.info(f"Scraping Instagram: @{INSTAGRAM_HANDLE}")
    new_images = []

    try:
        L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            max_connection_attempts=3,
        )
        # Use the proxy
        L.context.proxy = "http://172.30.10.11:3128"

        # Load saved session if available (helps avoid rate limiting)
        session_file = Path(__file__).resolve().parent.parent / "instagram_session.pkl"
        if session_file.exists():
            try:
                with open(session_file, 'rb') as f:
                    L.context = pickle.load(f)
                logger.info("Loaded Instagram session from file")
            except Exception:
                logger.warning("Failed to load Instagram session, proceeding without")

        profile = instaloader.Profile.from_username(L.context, INSTAGRAM_HANDLE)
        logger.info(f"Found Instagram profile: {profile.full_name} ({profile.mediacount} posts)")

        post_count = 0
        for post in profile.get_posts():
            if post_count >= INSTAGRAM_POST_LIMIT:
                break

            # Skip videos
            if post.is_video:
                post_count += 1
                continue

            img_url = post.url
            logger.info(f"Checking Instagram post {post_count + 1}: {img_url}")

            img_bytes = download_image(img_url)
            if img_bytes is None:
                post_count += 1
                continue

            img_hash = compute_sha256(img_bytes)
            if database.image_exists(img_hash):
                logger.info(f"Duplicate skipped: {img_hash[:12]}... (Instagram)")
                post_count += 1
                continue

            # Extract geolocation (Instagram strips EXIF, but we check anyway)
            lat, lon = None, None
            gps_coords = extract_exif_geolocation(img_bytes)
            if gps_coords:
                lat, lon = gps_coords

            # Use Instagram caption as metadata hint (will help VLM later)
            caption = post.caption[:200] if post.caption else ""
            if caption:
                logger.info(f"Caption: {caption[:80]}...")

            img_bytes = resize_if_needed(img_bytes)

            filename = f"{img_hash}.jpg"
            filepath = IMAGES_DIR / filename
            filepath.write_bytes(img_bytes)

            rel_path = str(filepath.relative_to(Path(__file__).resolve().parent.parent))
            source_label = f"https://www.instagram.com/{INSTAGRAM_HANDLE}/"
            database.insert_image_record(img_hash, source_label, rel_path, lat, lon)
            new_images.append((img_hash, rel_path))
            logger.info(f"New Instagram image saved: {filename}")

            post_count += 1

    except Exception as e:
        logger.error(f"Instagram scraping failed: {e}")

    logger.info(f"Instagram: {len(new_images)} new images downloaded")
    return new_images


def scrape_all_sources() -> list[tuple[str, str]]:
    """Iterate over all configured sources and scrape them."""
    all_new = []

    # Web sources (multi-pass for dynamic content)
    for source in SOURCES:
        new = process_source(source, passes=SCAN_PASSES)
        all_new.extend(new)
        time.sleep(REQUEST_DELAY)

    # Facebook: SKIPPED (requires token, 2FA blocked)
    # setup_social_sources()
    # for source_type, source_url in SOCIAL_SOURCES:
    #     new = process_source(source_url, passes=1)
    #     all_new.extend(new)
    #     time.sleep(REQUEST_DELAY)

    # Instagram: SKIPPED (authentication not yet resolved)
    # insta_new = scrape_instagram()
    # all_new.extend(insta_new)

    logger.info("Social media sources skipped for now (Instagram/Facebook).")
    return all_new


if __name__ == "__main__":
    logger.info("Starting manual scrape...")
    added = scrape_all_sources()
    logger.info(f"Total new images this run: {len(added)}")