"""
PIEAS Visual Intelligence Dashboard
Search, browse, filter, and export images using semantic search.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is in sys.path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer import search, collection as chroma_collection  # for stats
from src import database

# ---------- Page config ----------
st.set_page_config(
    page_title="PIEAS Visual Intelligence",
    page_icon="📸",
    layout="wide",
)

# ---------- Cached data helpers ----------
@st.cache_data(ttl=600)
def get_all_categories() -> list:
    """Return sorted list of distinct categories from the DB."""
    conn = database.get_connection()
    rows = conn.execute(
        "SELECT DISTINCT json_extract(description_json, '$.category') AS cat "
        "FROM descriptions"
    ).fetchall()
    conn.close()
    cats = sorted({row["cat"] for row in rows if row["cat"]})
    return cats


@st.cache_data(ttl=600)
def get_all_tags() -> list:
    """Return sorted list of all unique tags across all descriptions."""
    conn = database.get_connection()
    rows = conn.execute("SELECT description_json FROM descriptions").fetchall()
    conn.close()
    tags_set = set()
    for row in rows:
        try:
            desc = json.loads(row["description_json"])
            for t in desc.get("relevant_tags", []):
                tags_set.add(t)
        except Exception:
            pass
    return sorted(tags_set)


@st.cache_data(ttl=300)
def perform_search(query: str, top_k: int = 20, category: str = None):
    """Wrapper around indexer.search that returns results with full metadata."""
    results = search(query, top_k=top_k, category_filter=category)
    enriched = []
    conn = database.get_connection()
    for r in results:
        img_hash = r["id"]
        # get full description from DB
        row = conn.execute(
            "SELECT description_json FROM descriptions WHERE image_hash = ?",
            (img_hash,),
        ).fetchone()
        # get filepath
        img_row = conn.execute(
            "SELECT filepath, source_url FROM images WHERE hash = ?",
            (img_hash,),
        ).fetchone()
        if row and img_row:
            try:
                desc = json.loads(row["description_json"])
            except Exception:
                desc = {}
            enriched.append({
                "hash": img_hash,
                "filepath": img_row["filepath"],
                "source_url": img_row["source_url"],
                "distance": r["distance"],
                "description": desc,
            })
    conn.close()
    return enriched


# ---------- Sidebar ----------
st.sidebar.title("📸 PIEAS Visual Intelligence")
st.sidebar.markdown("Semantic search over PIEAS image archive")

# Stats
try:
    total_indexed = chroma_collection.count()
except Exception:
    total_indexed = 0
st.sidebar.metric("Images Indexed", total_indexed)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

all_cats = get_all_categories()
all_tags = get_all_tags()

selected_category = st.sidebar.selectbox("Category", ["All"] + all_cats)
selected_tags = st.sidebar.multiselect("Tags", all_tags)

# ---------- Main area ----------
st.title("Search PIEAS Images")

query = st.text_input("Search query (e.g., 'students in lab', 'convocation ceremony')", placeholder="Type a query...")

top_k = st.slider("Number of results", min_value=5, max_value=50, value=20)

if st.button("Search") or query:
    if not query or not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching..."):
            cat_filter = selected_category if selected_category != "All" else None
            results = perform_search(query, top_k=top_k, category=cat_filter)

        if not results:
            st.info("No results found. Try a different query.")
        else:
            # Filter by selected tags (case‑insensitive)
            if selected_tags:
                filtered = []
                for r in results:
                    desc_tags = r["description"].get("relevant_tags", [])
                    if any(
                        tag.lower() in [t.lower() for t in desc_tags]
                        for tag in selected_tags
                    ):
                        filtered.append(r)
                results = filtered

            st.success(f"Found {len(results)} results")

            # ---- Export button ----
            if results:
                df = pd.DataFrame([
                    {
                        "hash": r["hash"][:12],
                        "scene_description": r["description"].get("scene_description", ""),
                        "category": r["description"].get("category", ""),
                        "tags": ", ".join(r["description"].get("relevant_tags", [])),
                        "visible_text": r["description"].get("visible_text", ""),
                        "people_count": r["description"].get("people_count", ""),
                        "event_type": r["description"].get("event_type", ""),
                        "buildings_or_locations": r["description"].get("buildings_or_locations", ""),
                        "source_url": r["source_url"],
                        "filepath": r["filepath"],
                        "distance": r["distance"],
                    }
                    for r in results
                ])

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name="pieas_search_results.csv",
                    mime="text/csv",
                )

            # ---- Relevance helpers ----
            def similarity_score(distance: float) -> float:
                """Convert cosine distance to a percentage (0‑100)."""
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                return max(0.0, (1 - distance / 2) * 100)

            RELEVANCE_THRESHOLD = 1.0   # distance <= 1.0 => high relevance

            high_rel = [r for r in results if r["distance"] <= RELEVANCE_THRESHOLD]
            low_rel  = [r for r in results if r["distance"] > RELEVANCE_THRESHOLD]

            # ---- Display high‑relevance results ----
            if high_rel:
                st.subheader("Most relevant")
                cols = st.columns(3)
                for i, r in enumerate(high_rel):
                    col = cols[i % 3]
                    desc = r["description"]
                    img_path = PROJECT_ROOT / r["filepath"]
                    if img_path.exists():
                        col.image(str(img_path), width="stretch")
                    else:
                        col.warning(f"Image not found: {r['filepath']}")

                    snippet = desc.get("scene_description", "No description")[:120]
                    col.markdown(f"**{snippet}**")

                    # Relevance indicator
                    sim = similarity_score(r["distance"])
                    col.progress(int(sim), text=f"match: {sim:.0f}%")

                    col.markdown(f"🏷️ {desc.get('category', 'Other')}")
                    tags = ", ".join(desc.get("relevant_tags", []))
                    if tags:
                        col.markdown(f"🔖 {tags}")

                    with col.expander("Full metadata"):
                        st.markdown(f"**Scene:** {desc.get('scene_description', 'N/A')}")
                        st.markdown(f"**Category:** {desc.get('category', 'N/A')}")
                        st.markdown(f"**Event:** {desc.get('event_type', 'N/A')}")
                        st.markdown(f"**People:** {desc.get('people_count', 'N/A')}")
                        st.markdown(f"**Text:** {desc.get('visible_text', 'N/A')}")
                        st.markdown(f"**Buildings:** {desc.get('buildings_or_locations', 'N/A')}")
                        st.markdown(f"**Tags:** {', '.join(desc.get('relevant_tags', []))}")
                        st.markdown(f"**Distance:** {r['distance']:.4f}")
                        st.markdown(f"**Source:** {r['source_url']}")

            # ---- Display lower‑relevance results with a divider ----
            if low_rel:
                st.markdown("---")
                st.subheader("Other results")
                cols = st.columns(3)
                for i, r in enumerate(low_rel):
                    col = cols[i % 3]
                    desc = r["description"]
                    img_path = PROJECT_ROOT / r["filepath"]
                    if img_path.exists():
                        col.image(str(img_path), width="stretch")
                    else:
                        col.warning(f"Image not found: {r['filepath']}")

                    snippet = desc.get("scene_description", "No description")[:120]
                    col.markdown(f"**{snippet}**")

                    sim = similarity_score(r["distance"])
                    col.progress(int(sim), text=f"match: {sim:.0f}%")

                    col.markdown(f"🏷️ {desc.get('category', 'Other')}")
                    tags = ", ".join(desc.get("relevant_tags", []))
                    if tags:
                        col.markdown(f"🔖 {tags}")

                    with col.expander("Full metadata"):
                        st.markdown(f"**Scene:** {desc.get('scene_description', 'N/A')}")
                        st.markdown(f"**Category:** {desc.get('category', 'N/A')}")
                        st.markdown(f"**Event:** {desc.get('event_type', 'N/A')}")
                        st.markdown(f"**People:** {desc.get('people_count', 'N/A')}")
                        st.markdown(f"**Text:** {desc.get('visible_text', 'N/A')}")
                        st.markdown(f"**Buildings:** {desc.get('buildings_or_locations', 'N/A')}")
                        st.markdown(f"**Tags:** {', '.join(desc.get('relevant_tags', []))}")
                        st.markdown(f"**Distance:** {r['distance']:.4f}")
                        st.markdown(f"**Source:** {r['source_url']}")