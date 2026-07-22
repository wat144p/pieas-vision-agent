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
MAX_RETRIES = 3

# --- Prompt template ---
ANALYSIS_PROMPT = """
You are an expert visual analyst for the Pakistan Institute of Engineering and Applied Sciences (PIEAS).
Analyze the given image carefully and output a **valid JSON object** with the following fields.
Do NOT include any text outside the JSON object.

CRITICAL: The "people_count" field MUST be a plain integer number (e.g., 0, 1, 5, 10). Do NOT use "10+", "about 10", "several", or any other text. Just a number.
CRITICAL: The "visible_text" field should be a SINGLE string or null. If there are multiple text items, combine them with commas or semicolons into one string.

Required fields:
- scene_description: string, a concise 1‑2 sentence description of the scene.
- objects: list of strings, objects visible in the image (people, equipment, furniture, etc.).
- buildings_or_locations: string or null, name of the building or location if recognizable (e.g., "Main Auditorium", "Robotics Lab", "Cafeteria", "Library"). If the image is from PIEAS, try to identify the campus area.
- event_type: string or null, the type of event if applicable (e.g., "Convocation", "Seminar", "Cultural Show", "Sports Day", "Lab Work", "Campus Life").
- people_count: integer, approximate number of people visible (0 if none). Output ONLY a number.
- visible_text: string or null, any clearly readable text in the image (OCR). Combine multiple text items into one string.
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
    - Non-numeric values in numeric fields
    - Multiple quoted strings in visible_text field
    - Missing quotes around strings
    """
    # Extract the outermost { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return text
    json_str = match.group(0)

    # Fix missing commas between objects/arrays
    json_str = re.sub(r'"\s*\n?\s*"', '",\n"', json_str)
    json_str = re.sub(r']\s*\n?\s*"', '],\n"', json_str)
    json_str = re.sub(r'}\s*\n?\s*"', '},\n"', json_str)
    json_str = re.sub(r']\s*\n?\s*\{', '],\n{', json_str)
    json_str = re.sub(r'}\s*\n?\s*\{', '},\n{', json_str)
    json_str = re.sub(r']\s*\n?\s*]', '],\n]', json_str)
    json_str = re.sub(r'}\s*\n?\s*}', '},\n}', json_str)
    json_str = re.sub(r'([0-9])\s*\n?\s*"', r'\1,\n"', json_str)
    json_str = re.sub(r'(true|false|null)\s*\n?\s*"', r'\1,\n"', json_str)

    # Remove trailing commas before } or ]
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # --- FIX: visible_text with multiple quoted strings ---
    # Safely find the "visible_text" field value and combine multiple strings
    # without crossing into the next JSON key.
    def fix_visible_text(text: str) -> str:
        key_pat = r'"visible_text"\s*:\s*'
        m = re.search(key_pat, text)
        if not m:
            return text
        start = m.end()  # right after the colon
        # Find the beginning of the next JSON key: comma + optional whitespace + quoted string + colon
        next_key = re.search(r'\s*,\s*"[^"]+"\s*:', text[start:])
        if next_key:
            end = start + next_key.start()
        else:
            # No next key – take until the end of the string (safe for last field)
            end = len(text)
        segment = text[start:end]
        quoted = re.findall(r'"([^"]*)"', segment)
        if len(quoted) > 1:
            combined = ', '.join(quoted)
            # Replace the whole segment with the combined string
            text = text[:start] + f' "{combined}"' + text[end:]
        return text

    json_str = fix_visible_text(json_str)

    # --- FIX: people_count with non‑numeric values ---
    def extract_number(value):
        """Extract the first number from a string or return 0"""
        num_match = re.search(r'\d+', str(value))
        if num_match:
            return num_match.group(0)
        return 0

    def fix_people_count(match):
        value = match.group(1)
        num = extract_number(value)
        return f'"people_count": {num}'

    # Handle quoted values: "people_count": "10+", "people_count": "10+ (as there are...", etc.
    json_str = re.sub(r'"people_count"\s*:\s*"([^"]+)"', fix_people_count, json_str)
    # Handle unquoted values: "people_count": 10+, "people_count": about 10, etc.
    json_str = re.sub(r'"people_count"\s*:\s*([^,}\n]+)',
                      lambda m: f'"people_count": {extract_number(m.group(1))}',
                      json_str)

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
                    logger.info("JSON parsed successfully without repair.")  # emoji removed
                except json.JSONDecodeError as e:
                    logger.warning(f"Initial JSON parse failed: {e}, attempting repair...")
                    repaired = repair_json(clean)
                    logger.debug(f"Repaired JSON: {repaired[:200]}...")
                    try:
                        result = json.loads(repaired)
                        logger.info("JSON repair succeeded.")
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON repair also failed: {e2}. Raw: {clean[:500]}")
                        if attempt < MAX_RETRIES:
                            continue
                        else:
                            return None

                # Validate required fields
                required = [
                    "scene_description", "objects", "buildings_or_locations",
                    "event_type", "people_count", "visible_text", "relevant_tags", "category"
                ]
                
                # Check which fields are missing
                missing = [k for k in required if k not in result]
                
                # Try to fix common issues with visible_text
                if "visible_text" in result and isinstance(result["visible_text"], list):
                    # If visible_text is a list, join it into a string
                    result["visible_text"] = ", ".join([str(item) for item in result["visible_text"]])
                
                if missing:
                    logger.warning(f"Missing fields: {missing}. Attempting to fill defaults.")
                    # Fill missing fields with defaults
                    if "people_count" not in result:
                        result["people_count"] = 0
                    if "buildings_or_locations" not in result:
                        result["buildings_or_locations"] = None
                    if "event_type" not in result:
                        result["event_type"] = None
                    if "visible_text" not in result:
                        result["visible_text"] = None
                    if "objects" not in result:
                        result["objects"] = []
                    if "relevant_tags" not in result:
                        result["relevant_tags"] = []
                    if "scene_description" not in result:
                        result["scene_description"] = "No description available"
                    if "category" not in result:
                        result["category"] = "Other"
                    
                    # Check again if we have all fields
                    if all(k in result for k in required):
                        logger.info("Filled missing fields with defaults, continuing...")
                    else:
                        logger.error("Still missing required fields after filling defaults")
                        continue

                # Ensure people_count is an integer
                try:
                    result["people_count"] = int(result["people_count"])
                except (ValueError, TypeError):
                    num_match = re.search(r'\d+', str(result["people_count"]))
                    result["people_count"] = int(num_match.group(0)) if num_match else 0
                
                # Ensure visible_text is a string or None
                if result.get("visible_text") is not None and not isinstance(result["visible_text"], str):
                    result["visible_text"] = str(result["visible_text"])
                
                logger.info(f"Successfully analyzed {image_file.name}")  # emoji removed
                return result

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