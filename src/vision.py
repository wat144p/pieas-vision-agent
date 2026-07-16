"""
Module 2: Vision Analysis (Vision)
Uses a local VLM via Ollama to describe images and extract structured metadata.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import ollama

logger = logging.getLogger(__name__)

VISION_MODEL = "llava:34b"          # Use the 34B LLaVA model
OLLAMA_TIMEOUT = 120
MAX_RETRIES = 2

ANALYSIS_PROMPT = """
You are an expert visual analyst for the Pakistan Institute of Engineering and Applied Sciences (PIEAS).
Analyze the given image carefully and output a **valid JSON object** with the following fields.
Do NOT include any text outside the JSON object.

Required fields:
- scene_description: string, a concise 1‑2 sentence description of the scene.
- objects: list of strings, objects visible in the image.
- buildings_or_locations: string or null, name of the building or location if recognizable.
- event_type: string or null, the type of event if applicable.
- people_count: integer, approximate number of people visible (0 if none).
- visible_text: string or null, any clearly readable text in the image (OCR).
- relevant_tags: list of strings, 3‑7 keywords or tags that categorize the image content.
- category: string, a single high‑level category from: ["Academics", "Research", "Events", "Campus Life", "Facilities", "Admissions", "Other"].

Example output:
{
  "scene_description": "Students working on a robotic arm in a well‑lit laboratory.",
  "objects": ["students", "robotic arm", "computers"],
  "buildings_or_locations": "Robotics Lab, Department of Mechanical Engineering",
  "event_type": "Lab Work",
  "people_count": 4,
  "visible_text": "PIEAS Robotics Club",
  "relevant_tags": ["students", "robotics", "laboratory", "engineering"],
  "category": "Research"
}
"""

def analyze_image(image_path: str) -> Optional[Dict[str, Any]]:
    image_file = Path(image_path)
    if not image_file.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        image_file = project_root / image_path
    if not image_file.exists():
        logger.error(f"Image file not found: {image_file}")
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"Analyzing {image_file.name} (attempt {attempt+1})")
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": ANALYSIS_PROMPT,
                    "images": [str(image_file)]
                }],
                options={"temperature": 0.1}
            )
            content = response.get("message", {}).get("content", "")
            clean = content.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()
            result = json.loads(clean)
            required = ["scene_description", "objects", "buildings_or_locations",
                        "event_type", "people_count", "visible_text", "relevant_tags", "category"]
            if all(k in result for k in required):
                return result
            else:
                missing = [k for k in required if k not in result]
                logger.warning(f"Missing fields: {missing}. Retrying.")
                continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES:
                continue
            else:
                return None
        except Exception as e:
            logger.error(f"VLM analysis error: {e}")
            return None
    return None