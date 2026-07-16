"""One-time run of VLM analysis on all unanalyzed images.
   Uses the same logging and modules as agent.py."""

import json
import logging
from pathlib import Path

from src.vision import analyze_image
from src import database

# Same logging config as agent.py
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
logger = logging.getLogger("vlm_batch")

def main():
    unanalyzed = database.get_unanalyzed_images()
    logger.info(f"Found {len(unanalyzed)} unanalyzed images")
    for img_hash, filepath in unanalyzed:
        try:
            desc = analyze_image(filepath)
            if desc:
                database.store_description(img_hash, json.dumps(desc))
                database.mark_analyzed(img_hash)
                logger.info(f"  {img_hash[:12]}... analyzed and stored.")
            else:
                logger.warning(f"  {img_hash[:12]}... failed, skipping.")
        except Exception as e:
            logger.error(f"  {img_hash[:12]}... error: {e}")

if __name__ == "__main__":
    main()