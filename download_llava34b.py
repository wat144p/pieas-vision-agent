"""
Downloads all required files for LLaVA-34B multimodal model.
- Base model: llava-v1.6-34b.Q4_K_M.gguf
- Vision projector: mmproj-model-f16.gguf
"""

import requests
import urllib3
from pathlib import Path
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROXIES = {
    "http": "http://172.30.10.11:3128",
    "https": "http://172.30.10.11:3128",
}

BASE_URL = "https://huggingface.co/cjpais/llava-v1.6-34b-gguf/resolve/main"
FILES = {
    "llava-v1.6-34b.Q4_K_M.gguf": BASE_URL + "/llava-v1.6-34b.Q4_K_M.gguf",
    "mmproj-model-f16.gguf": BASE_URL + "/mmproj-model-f16.gguf",
}
DEST_DIR = Path("C:/ollama_models")
DEST_DIR.mkdir(parents=True, exist_ok=True)


def download_file(filename, url):
    dest = DEST_DIR / filename
    # If the file already exists and has non-zero size, check if it's complete
    if dest.exists():
        # We assume a previously interrupted download; we can resume.
        # But if size is exactly the expected content-length, we could skip.
        # For simplicity, we'll attempt to resume by default.
        initial_pos = dest.stat().st_size
        headers = {"Range": f"bytes={initial_pos}-"}
        print(f"Resuming {filename} from byte {initial_pos:,} ({initial_pos/(1024**3):.2f} GB)")
    else:
        initial_pos = 0
        headers = {}

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.get(
            url,
            proxies=PROXIES,
            headers=headers,
            stream=True,
            verify=False,
            timeout=(10, 300),
        )
        # If the server returns 416 (Requested Range Not Satisfiable), the file is likely complete.
        if response.status_code == 416:
            print(f"{filename} appears to be already fully downloaded (range not satisfiable).")
            return

        response.raise_for_status()

        total_size = None
        if "content-length" in response.headers:
            total_size = int(response.headers["content-length"]) + initial_pos
            print(f"Total size: {total_size/(1024**3):.2f} GB")

        mode = "ab" if initial_pos else "wb"
        with open(dest, mode) as f, tqdm(
            desc=filename,
            initial=initial_pos,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"{filename} download complete.\n")
    except requests.exceptions.RequestException as e:
        print(f"\n{filename} download interrupted: {e}")
        print("Run the script again to resume.\n")
        exit(1)


if __name__ == "__main__":
    for fname, url in FILES.items():
        download_file(fname, url)
    print("All files downloaded successfully.")
    print("Next steps:")
    print("  1. Create Modelfile in C:\\ollama_models\\ with the following content:")
    print("     FROM llava-v1.6-34b.Q4_K_M.gguf")
    print("     ADAPTER mmproj-model-f16.gguf")   # <-- CORRECTED: ADAPTER, not PARAMETER
    print("  2. Run: ollama create llava:34b -f C:\\ollama_models\\Modelfile")
    print("  3. Verify: ollama list")