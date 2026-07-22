# PIEAS Autonomous Visual Intelligence Agent

An autonomous multimodal AI agent that continuously scrapes, analyzes, and indexes images from PIEAS university web pages. It uses a local vision‑language model (LLaVA‑34B) to generate rich descriptions and a semantic search engine (ChromaDB) so you can query the visual history of the campus with natural language.

## Architecture

```
Agent Scheduler → Web Scraper (scraper.py) → SQLite (pipeline.db)
         ↓                                    ↓
   Vision Analysis (vision.py + LLaVA‑34B)   Vector Indexer (indexer.py + ChromaDB)
         ↓                                    ↓
   Streamlit Dashboard (app/app.py) ← ChromaDB
```

## Modules

- **Scraper** (`src/scraper.py`): Downloads unique images from PIEAS domains, deduplicates by SHA‑256, extracts EXIF geolocation.
- **Vision** (`src/vision.py`): Sends images to a local LLaVA‑34B model (via Ollama), repairs malformed JSON, returns structured metadata.
- **Indexer** (`src/indexer.py`): Embeds textual descriptions with `all-MiniLM-L6-v2` and stores them in ChromaDB for semantic search.
- **Agent** (`agent.py`): Orchestrates the pipeline on a schedule (Mondays & Thursdays at 9 AM) – scrape → analyse → index.
- **Dashboard** (`app/app.py`): Streamlit web interface for natural‑language search, filtering, and result export.

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running with the `llava:34b` model (see below)
- Internet through proxy `http://172.30.10.11:3128`

### 1. Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Add proxy to activation**: Edit `venv\Scripts\Activate.ps1` and add these lines before the signature block:

```powershell
$env:HTTP_PROXY = "http://172.30.10.11:3128"
$env:HTTPS_PROXY = "http://172.30.10.11:3128"
$env:NO_PROXY = "localhost,127.0.0.1"
```

Reactivate after editing.

### 2. Install Dependencies
```powershell
pip install -r requirements.txt --proxy http://172.30.10.11:3128
pip install chromadb sentence-transformers streamlit pandas --proxy http://172.30.10.11:3128
pip install torch --index-url https://download.pytorch.org/whl/cpu --proxy http://172.30.10.11:3128
pip install torchvision --index-url https://download.pytorch.org/whl/cpu --proxy http://172.30.10.11:3128
```

### 3. LLaVA Model
```powershell
python download_llava34b.py
ollama create llava:34b -f C:\ollama_models\Modelfile
```

### 4. Run Pipeline
```powershell
python agent.py --now
```

### 5. Dashboard
```powershell
streamlit run app/app.py
```
Visit `http://localhost:8501`.

## Usage
- **Scheduled agent**: `python agent.py` – runs every Monday and Thursday at 9 AM.
- **Search**: Use the dashboard or `from src.indexer import search` directly.
- **Export**: Dashboard download button gives CSV.

## Known Limitations
- Proxy required; SSL verification disabled.
- No GPU – VLM analysis runs on CPU (~1.5 min/image).
- Instagram/Facebook scraping disabled due to authentication issues.
- Ollama must be running for analysis.

## File Inventory
| File | Purpose |
|------|---------|
| `src/scraper.py` | Web scraper |
| `src/vision.py` | VLM analysis & JSON repair |
| `src/indexer.py` | ChromaDB indexing & search |
| `src/database.py` | SQLite helpers |
| `agent.py` | Pipeline orchestrator |
| `app/app.py` | Streamlit dashboard |
| `download_llava34b.py` | Helper to download GGUF files |
| `demo.py` | End‑to‑end demonstration |