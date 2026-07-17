"""
Module 2: Vision Analysis (Vision)
Uses a local VLM via Ollama to describe images and extract structured metadata.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

import ollama

logger = logging.getLogger(__name__)

# --- Configuration ---
VISION_MODEL = "llava:34b"
OLLAMA_TIMEOUT = 120          # (unused – kept for reference)
MAX_RETRIES = 2

# --- Prompt template ---
ANALYSIS_PROMPT = """
You are an expert visual analyst for the Pakistan Institute of Engineering and Applied Sciences (PIEAS).
Analyze the given image carefully and output a **valid JSON object** with the following fields.
Do NOT include any text outside the JSON object.

Required fields:
- scene_description: string, a concise 1‑2 sentence description of the scene.
- objects: list of strings, objects visible in the image (people, equipment, furniture, etc.).
- buildings_or_locations: string or null, name of the building or location if recognizable (e.g., "Main Auditorium", "Robotics Lab", "Cafeteria", "Library"). If the image is from PIEAS, try to identify the campus area.
- event_type: string or null, the type of event if applicable (e.g., "Convocation", "Seminar", "Cultural Show", "Sports Day", "Lab Work", "Campus Life").
- people_count: integer, approximate number of people visible (0 if none).
- visible_text: string or null, any clearly readable text in the image (OCR).
- relevant_tags: list of strings, 3‑7 keywords or tags that categorize the image content (e.g., "students", "laboratory", "robotics", "convocation", "campus building").
- category: string, a single high‑level category from: ["Academics", "Research", "Events", "Campus Life", "Facilities", "Admissions", "Other"].

Example output:
{
  "scene_description": "Students working on a robotic arm in a well‑lit laboratory with equipment in the background.",
  "objects": ["students", "robotic arm", "computers", "lab benches"],
  "buildings_or_locations": "Robotics Lab, Department of Mechanical Engineering",
  "event_type": "Lab Work",
  "people_count": 4,
  "visible_text": "PIEAS Robotics Club",
  "relevant_tags": ["students", "robotics", "laboratory", "engineering", "project"],
  "category": "Research"
}
"""


def repair_json(text: str) -> str:
    """
    Attempt to fix common JSON errors from VLM output:
    - Missing commas between key-value pairs
    - Trailing commas after last element
    - Extra text before/after JSON object
    """
    # Extract the outermost { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return text
    json_str = match.group(0)

    # Fix missing commas between objects/arrays by looking for patterns
    # "val"\n"key" -> "val",\n"key"
    json_str = re.sub(r'"\s*\n?\s*"', '",\n"', json_str)
    json_str = re.sub(r']\s*\n?\s*"', '],\n"', json_str)
    json_str = re.sub(r'}\s*\n?\s*"', '},\n"', json_str)
    json_str = re.sub(r']\s*\n?\s*\{', '],\n{', json_str)
    json_str = re.sub(r'}\s*\n?\s*\{', '},\n{', json_str)
    json_str = re.sub(r']\s*\n?\s*]', '],\n]', json_str)
    json_str = re.sub(r'}\s*\n?\s*}', '},\n}', json_str)
    json_str = re.sub(r'([0-9])\s*\n?\s*"', r'\1,\n"', json_str)   # number followed by "
    json_str = re.sub(r'(true|false|null)\s*\n?\s*"', r'\1,\n"', json_str)  # true/false/null followed by "

    # Remove trailing commas before } or ]
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    return json_str


def analyze_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyze an image using the VLM and return structured JSON.
    """
    image_file = Path(image_path)
    if not image_file.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        image_file = project_root / image_path

    if not image_file.exists():
        logger.error(f"Image file not found: {image_file}")
        return None

    # Resize for speed (optional but recommended)
    resized_path = None
    send_path = image_file
    try:
        from PIL import Image
        img = Image.open(image_file)
        max_dim = 1024
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            resized_path = image_file.with_suffix(".resized.jpg")
            img.save(resized_path, "JPEG", quality=85)
            send_path = resized_path
            logger.info(f"Resized {image_file.name} from {w}x{h} to {new_size[0]}x{new_size[1]}")
    except Exception as e:
        logger.warning(f"Could not resize image, sending original: {e}")

    try:
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(f"Analyzing {image_file.name} (attempt {attempt+1})")
                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": ANALYSIS_PROMPT,
                        "images": [str(send_path)]
                    }],
                    options={"temperature": 0.1}
                    # timeout removed – the library doesn't accept it directly.
                )
                content = response.get("message", {}).get("content", "")
                logger.debug(f"Raw VLM response: {content[:300]}...")

                # Clean up markdown fences
                clean = content.strip()
                if clean.startswith("```"):
                    lines = clean.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean = "\n".join(lines).strip()

                # First parse attempt
                try:
                    result = json.loads(clean)
                except json.JSONDecodeError:
                    logger.warning("Initial JSON parse failed, attempting repair...")
                    repaired = repair_json(clean)
                    try:
                        result = json.loads(repaired)
                        logger.info("JSON repair succeeded.")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON repair also failed: {e}. Raw: {clean[:500]}")
                        continue  # retry

                # Validate required fields
                required = [
                    "scene_description", "objects", "buildings_or_locations",
                    "event_type", "people_count", "visible_text", "relevant_tags", "category"
                ]
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
    finally:
        # Clean up temporary resized file
        if resized_path and resized_path.exists() and resized_path != image_file:
            try:
                resized_path.unlink()
                logger.debug(f"Temporary resized image deleted: {resized_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {resized_path}: {e}")