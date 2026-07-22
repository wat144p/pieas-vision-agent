"""End‑to‑end demo: scrape one source, analyse the first image, index it, and search."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper import process_source, scrape_all_sources
from src.vision import analyze_image
from src.indexer import index_single, search
from src import database

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("demo")

DEMO_SOURCE = "https://www.pieas.edu.pk/"

def main():
    logger.info("=== PIEAS Visual Intelligence Demo ===")

    # 1. Scrape one source
    logger.info("1. Scraping %s (1 pass)...", DEMO_SOURCE)
    new_images = process_source(DEMO_SOURCE, passes=1)
    if not new_images:
        logger.warning("No new images – trying full scrape...")
        new_images = scrape_all_sources()
    if not new_images:
        logger.error("Still no images. Aborting.")
        return

    img_hash, filepath = new_images[0]
    logger.info("   New image: %s (%s)", img_hash[:12], filepath)

    # 2. Analyse
    logger.info("2. Analysing image with VLM...")
    desc = analyze_image(filepath)
    if not desc:
        logger.error("VLM analysis failed.")
        return
    logger.info("   Scene: %s", desc.get("scene_description", "")[:80])

    # 3. Store & mark analysed
    database.store_description(img_hash, json.dumps(desc))
    database.mark_analyzed(img_hash)

    # 4. Index into ChromaDB
    logger.info("3. Indexing...")
    index_single(img_hash, desc)
    logger.info("   Indexed.")

    # 5. Search
    logger.info("4. Testing search: 'laboratory'")
    results = search("laboratory", top_k=3)
    for i, r in enumerate(results, 1):
        logger.info("   %d. %s... (distance: %.4f)", i, r["document"][:80], r["distance"])

    logger.info("=== Demo complete ===")

if __name__ == "__main__":
    main()