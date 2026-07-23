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
- [Ollama](https://ollama.com/) installed and running
- Internet access

> **If you are inside the PIEAS network**  
> The PIEAS campus network routes all external traffic through a proxy at `http://172.30.10.11:3128`.  
> Commands that download packages or data include a `--proxy` flag when needed.  
> If you are **outside** PIEAS, skip the proxy flags and do **not** set any proxy environment variables.

### 1. Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*Optional for PIEAS users* – add these lines to `venv\Scripts\Activate.ps1` (before the signature block) to set the proxy automatically:

```powershell
$env:HTTP_PROXY = "http://172.30.10.11:3128"
$env:HTTPS_PROXY = "http://172.30.10.11:3128"
$env:NO_PROXY = "localhost,127.0.0.1"
```

Reactivate after editing.

### 2. Install Dependencies

**Standard installation (no proxy):**
```powershell
pip install -r requirements.txt
pip install chromadb sentence-transformers streamlit pandas
pip install torch torchvision
```

**PIEAS proxy users:** add `--proxy http://172.30.10.11:3128` to each pip command.

> **Note:** PyTorch may require the latest **Microsoft Visual C++ Redistributable**. If you encounter a DLL load error, download and install it from [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 3. LLaVA Model

The vision pipeline uses the **llava‑v1.6‑34B** model via Ollama.

**a) Download the model files**  
Run the provided script to fetch the GGUF files into `C:\ollama_models`:

```powershell
python download_llava34b.py
```

**b) Create the Ollama Modelfile**  
Create a file named `Modelfile` inside `C:\ollama_models` with the following content:

```
FROM llava-v1.6-34b.Q4_K_M.gguf
ADAPTER mmproj-model-f16.gguf
```

(The download script will also print these instructions after finishing.)

**c) Build the Ollama model**  
```powershell
ollama create llava:34b -f C:\ollama_models\Modelfile
```

**d) Verify**  
```powershell
ollama list
```
You should see `llava:34b` in the list.

### 4. Run the Pipeline
```powershell
python agent.py --now
```
This scrapes, analyses, and indexes all available images.

### 5. Launch the Dashboard
```powershell
streamlit run app/app.py
```
Open `http://localhost:8501` to search and browse.

## Usage
- **Scheduled agent**: `python agent.py` – runs every Monday and Thursday at 9 AM.
- **Search**: Use the dashboard or `from src.indexer import search` from Python.
- **Export results**: Download search results as CSV from the dashboard.

## Known Limitations
- **Proxy inside PIEAS**: SSL verification is disabled for scraping when behind the PIEAS proxy.
- **VLM speed**: LLaVA‑34B is a large model. On CPU it analyses an image in ~1.5 minutes; on GPU it is much faster.
- **Social media disabled**: Instagram and Facebook scraping are implemented but currently disabled due to authentication issues.
- **Ollama required**: The `llava:34b` model must be loaded and Ollama must be running during analysis.

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
