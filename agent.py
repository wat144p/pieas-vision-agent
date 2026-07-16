"""
PIEAS Visual Intelligence Agent — Main Orchestrator

Runs the full pipeline on a schedule:
1. Scrape new images
2. Analyze with VLM (Week 3-4)
3. Index into ChromaDB (Week 5)
"""

import schedule
import time
import logging
from pathlib import Path
import json

from src.scraper import scrape_all_sources
from src.vision import analyze_image
from src import database

# ---------- CONFIGURATION ----------
SCHEDULE_DAYS = ["monday", "thursday"]
SCHEDULE_TIME = "09:00"

# Logging
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def pipeline():
    """Execute one full run of the ingestion + analysis pipeline."""
    logger.info("=" * 50)
    logger.info("PIPELINE RUN STARTED")
    logger.info("=" * 50)

    # ----- Step 1: Scrape new images -----
    logger.info("Step 1/3: Scraping sources for new images...")
    try:
        new_images = scrape_all_sources()
        logger.info(f"Scraped {len(new_images)} new images")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        new_images = []

    # ----- Step 2: Analyze with VLM -----
    logger.info("Step 2/3: VLM analysis...")
    try:
        unanalyzed = database.get_unanalyzed_images()
        logger.info(f"Found {len(unanalyzed)} unanalyzed images")
        for img_hash, filepath in unanalyzed:
            try:
                description = analyze_image(filepath)
                if description:
                    database.store_description(img_hash, json.dumps(description))
                    database.mark_analyzed(img_hash)
                    logger.info(f"  {img_hash[:12]}... analyzed and stored.")
                else:
                    logger.warning(f"  {img_hash[:12]}... analysis returned None, skipping.")
            except Exception as e:
                logger.error(f"  Failed to analyze {img_hash[:12]}: {e}")
    except Exception as e:
        logger.error(f"VLM analysis step failed: {e}")

    # ----- Step 3: Index into ChromaDB (placeholder for Week 5) -----
    logger.info("Step 3/3: ChromaDB indexing — NOT YET IMPLEMENTED")
    # TODO Week 5: Import indexer.py and embed descriptions

    logger.info("=" * 50)
    logger.info("PIPELINE RUN COMPLETE")
    logger.info("=" * 50)


def run_once():
    """Run the pipeline immediately (for testing)."""
    logger.info("Running pipeline once (manual trigger)...")
    pipeline()


def start_scheduler():
    """Start the scheduled agent loop."""
    # Schedule for configured days
    for day in SCHEDULE_DAYS:
        getattr(schedule.every(), day).at(SCHEDULE_TIME).do(pipeline)

    logger.info(f"Agent started. Scheduled: {', '.join(SCHEDULE_DAYS)} at {SCHEDULE_TIME}")
    logger.info("Waiting for next scheduled run...")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        # Run immediately (for testing)
        run_once()
    else:
        # Start the scheduler
        start_scheduler()