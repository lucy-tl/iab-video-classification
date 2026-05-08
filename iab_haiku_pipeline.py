"""
P1 — Haiku Direct
=================
Pegasus segments the video into scenes (with descriptions). Claude Haiku then
classifies each scene by picking the best match from all 704 IAB 3.1 breadcrumb
paths in a single tool_use call. The matched node is walked up the parent chain
to fill T1–T4.

Prerequisite: run setup_taxonomy.py once to load the taxonomy into SQLite.

Usage:
    python3 iab_haiku_pipeline.py --video <ASSET_ID>
    python3 iab_haiku_pipeline.py --index <INDEX_ID>

Requirements:
    pip install anthropic twelvelabs

Environment variables:
    ANTHROPIC_API_KEY=sk-ant-...
    TWELVELABS_API_KEY=tlk_...
"""

import sqlite3
import json
import argparse
import os
import time
import difflib
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
from twelvelabs import TwelveLabs
from segmentation import get_or_run_scenes

DB_PATH = "iab_taxonomy_3.1.db"


# ─────────────────────────────────────────────
# TAXONOMY: load into memory once at runtime
# ─────────────────────────────────────────────

def load_taxonomy_cache(db_path: str = DB_PATH) -> dict:
    """Load taxonomy into memory as a dict keyed by unique_id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM iab_taxonomy")
    cache = {row["unique_id"]: dict(row) for row in cur.fetchall()}
    conn.close()
    return cache


def cache_get_row_by_id(unique_id: str, cache: dict) -> Optional[dict]:
    """Get any row by ID from in-memory cache."""
    return cache.get(unique_id)


def resolve_all_tiers(best_match_id: str, cache: dict) -> dict:
    """Walk up the parent chain from best_match_id to fill T1–T4."""
    tiers = {1: None, 2: None, 3: None, 4: None}
    node = cache_get_row_by_id(best_match_id, cache)

    while node:
        tier = node.get("tier")
        if tier in tiers:
            tiers[tier] = {"id": node["unique_id"], "name": node["name"]}
        parent_id = node.get("parent_id")
        node = cache_get_row_by_id(parent_id, cache) if parent_id else None

    return tiers


# ─────────────────────────────────────────────
# HAIKU T1–T4 CLASSIFICATION
# ─────────────────────────────────────────────

def classify_scene(
    scene: dict,
    cache: dict,
    haiku_client: anthropic.Anthropic,
    crumb_to_id: dict = None,
    all_crumbs: list = None,
) -> dict:
    """Classify a single scene via Haiku. Runs in a thread pool."""
    description = scene["scene_description"]

    haiku_result = classify_with_haiku(
        description, haiku_client,
        cache=cache, crumb_to_id=crumb_to_id, all_crumbs=all_crumbs,
    )
    if not haiku_result:
        return {"error": "Haiku could not find a matching taxonomy node"}

    best_id = haiku_result["node_id"]
    node    = cache_get_row_by_id(best_id, cache)
    if not node:
        return {"error": f"Haiku returned unknown ID: {best_id}"}

    confidence      = haiku_result.get("confidence", 0)
    second_node_id  = haiku_result.get("second_node_id")
    second_confidence = haiku_result.get("second_confidence", 0)
    confidence_gap  = confidence - second_confidence

    tiers        = resolve_all_tiers(best_id, cache)
    second_tiers = resolve_all_tiers(second_node_id, cache) if second_node_id else {}

    # Build breadcrumb strings for first and second choice
    first_parts  = [t["name"] for t in [tiers[1], tiers[2], tiers[3], tiers[4]] if t]
    second_parts = [t["name"] for t in [second_tiers.get(1), second_tiers.get(2),
                                         second_tiers.get(3), second_tiers.get(4)] if t]
    first_choice  = " > ".join(first_parts)
    second_choice = " > ".join(second_parts) if second_parts else None

    return {
        "scene_description":  description,
        "first_choice":       first_choice,
        "first_confidence":   confidence,
        "second_choice":      second_choice,
        "second_confidence":  second_confidence,
        "confidence_gap":     confidence_gap,
        "tier1": tiers[1],
        "tier2": tiers[2],
        "tier3": tiers[3],
        "tier4": tiers[4],
    }


def classify_with_haiku(
    scene_description: str,
    client: anthropic.Anthropic,
    cache: dict = None,
    crumb_to_id: dict = None,
    all_crumbs: list = None,
) -> Optional[dict]:
    """
    Tool-use with all 704 breadcrumb paths as a constrained enum.
    Returns dict with node_id, confidence, second_node_id, second_confidence, or None.
    """
    NONE = "None of the above"
    enum = (all_crumbs or []) + [NONE]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=[{
            "name": "classify",
            "description": "Classify the scene into the single most specific IAB content category",
            "input_schema": {
                "type": "object",
                "properties": {
                    "best_match":        {"type": "string",  "enum": enum},
                    "confidence":        {"type": "integer", "minimum": 0, "maximum": 100},
                    "second_match":      {"type": "string",  "enum": enum},
                    "second_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["best_match", "confidence", "second_match", "second_confidence"],
            },
        }],
        tool_choice={"type": "tool", "name": "classify"},
        messages=[{"role": "user", "content":
            f"Scene description: {scene_description}\n\n"
            f"Select the single most specific IAB Content Taxonomy 3.1 category "
            f"that best matches this scene (or '{NONE}' if nothing clearly fits), "
            f"your confidence (0-100), your second-best choice, and its confidence."
        }],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if not tool_block:
        print("⚠️  Haiku: no tool_use block in response")
        return None
    inp    = tool_block.input
    choice = inp.get("best_match")
    if not choice or choice == NONE:
        return None
    node_id = (crumb_to_id or {}).get(choice)
    if not node_id:
        # Fall back to closest match if Haiku returns a label not in the enum.
        matches = difflib.get_close_matches(choice, all_crumbs or [], n=1, cutoff=0.5)
        if matches:
            print(f"⚠️  Haiku returned '{choice}' (not in taxonomy) — fuzzy-matched to '{matches[0]}'")
            choice  = matches[0]
            node_id = (crumb_to_id or {}).get(choice)
        if not node_id:
            return None

    second        = inp.get("second_match")
    second_node_id = (crumb_to_id or {}).get(second) if second and second != NONE else None
    if second and second != NONE and not second_node_id:
        sec_matches = difflib.get_close_matches(second, all_crumbs or [], n=1, cutoff=0.5)
        if sec_matches:
            second_node_id = (crumb_to_id or {}).get(sec_matches[0])

    return {
        "node_id":          node_id,
        "confidence":       inp.get("confidence", 0),
        "second_node_id":   second_node_id,
        "second_confidence": inp.get("second_confidence", 0),
    }


# ─────────────────────────────────────────────
# TWELVELABS: ANALYZE VIDEO
# ─────────────────────────────────────────────

def format_time(seconds):
    """Convert seconds to mm:ss string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _run_single_video(asset_id, tl_client, haiku_client, cache, crumb_to_id=None, all_crumbs=None):
    """Run the full pipeline for one video."""

    print(f"\nAnalyzing video {asset_id}...")
    scenes = get_or_run_scenes(asset_id, tl_client)
    print(f"  {len(scenes)} scenes — classifying...")

    results_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(classify_scene, scene, cache, haiku_client, crumb_to_id, all_crumbs): i
            for i, scene in enumerate(scenes)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = {**scenes[i], **future.result()}
            except Exception as e:
                result = {**scenes[i], "error": str(e)}
            results_map[i] = result

    ordered = [results_map[i] for i in range(len(scenes))]
    ordered.sort(key=lambda r: r["start"])

    def fmt(t): return f"{t['name']} ({t['id']})" if t else "—"

    col_w = 170
    print("─" * col_w)
    print(f"  {'Time':<16}{'First choice [conf%/gap:N]':<52}{'Second choice [conf%]':<44}{'Tier 2':<30}{'Tier 3':<30}Tier 4")
    print("─" * col_w)
    for result in ordered:
        time_str = f"{format_time(result['start'])} – {format_time(result['end'])}"
        if "error" in result:
            print(f"  {time_str:<16}❌ {result['error']}")
        else:
            first  = f"{result.get('first_choice','—')} [{result.get('first_confidence',0)}%/gap:{result.get('confidence_gap',0)}]"
            second = f"{result.get('second_choice','—')} [{result.get('second_confidence',0)}%]"
            print(f"  {time_str:<16}{first:<52}{second:<44}{fmt(result.get('tier2')):<30}{fmt(result.get('tier3')):<30}{fmt(result.get('tier4'))}")
    print("─" * col_w)

    return ordered


def _init_clients(db_path):
    """Initialize clients and taxonomy cache. Returns None if env vars are missing."""
    tl_api_key    = os.environ.get("TWELVELABS_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not tl_api_key:
        print("❌ TWELVELABS_API_KEY is not set. Run: export TWELVELABS_API_KEY=tlk_...")
        return None
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")
        return None

    tl_client     = TwelveLabs(api_key=tl_api_key)
    haiku_client  = anthropic.Anthropic(api_key=anthropic_key)
    cache         = load_taxonomy_cache(db_path)

    def _breadcrumb(node_id):
        parts, node = [], cache.get(node_id)
        while node:
            parts.append(node["name"])
            pid = node.get("parent_id")
            node = cache.get(pid) if pid else None
        return " > ".join(reversed(parts))

    all_crumbs  = sorted([_breadcrumb(r["unique_id"]) for r in cache.values()])
    crumb_to_id = {_breadcrumb(r["unique_id"]): r["unique_id"] for r in cache.values()}
    print(f"  📊 {len(all_crumbs)} breadcrumb paths loaded for Haiku tool-use enum")
    return tl_client, haiku_client, cache, crumb_to_id, all_crumbs


def save_results_json(asset_id: str, results: list):
    """Save results to results/<asset_id>_v2.json."""
    os.makedirs("results", exist_ok=True)
    path = f"results/{asset_id}_v2.json"
    payload = {
        "asset_id": asset_id,
        "pipeline": "v2",
        "segments": [
            {
                "start":             r["start"],
                "end":               r["end"],
                "first_choice":      r.get("first_choice"),
                "first_confidence":  r.get("first_confidence"),
                "second_choice":     r.get("second_choice"),
                "second_confidence": r.get("second_confidence"),
                "confidence_gap":    r.get("confidence_gap"),
                "tier1":             r.get("tier1"),
                "tier2":             r.get("tier2"),
                "tier3":             r.get("tier3"),
                "tier4":             r.get("tier4"),
            }
            for r in results if "error" not in r
        ]
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  💾 Saved to {path}")


def analyze_video(asset_id: str, db_path: str = DB_PATH):
    """Run the pipeline on a single video asset."""
    if not asset_id or not asset_id.strip():
        print("❌ No video ID provided. Usage: python3 iab_haiku_pipeline.py --video <ASSET_ID>")
        return None

    clients = _init_clients(db_path)
    if not clients:
        return None

    results = _run_single_video(asset_id, *clients)
    if results:
        save_results_json(asset_id, results)
    return results


def analyze_index(index_id: str, db_path: str = DB_PATH):
    """Run the pipeline on all videos in a TwelveLabs index."""
    clients = _init_clients(db_path)
    if not clients:
        return None
    tl_client, haiku_client, cache, crumb_to_id, all_crumbs = clients

    videos = list(tl_client.indexes.videos.list(index_id))
    print(f"Found {len(videos)} videos in index {index_id}\n")

    all_results = {}
    for video in videos:
        asset_id = video.id
        try:
            results = _run_single_video(asset_id, tl_client, haiku_client, cache, crumb_to_id, all_crumbs)
            all_results[asset_id] = results
            if results:
                save_results_json(asset_id, results)
        except Exception as e:
            print(f"❌ Video {asset_id} failed: {e}")
            all_results[asset_id] = None

    output_path = f"results_{index_id}.md"
    with open(output_path, "w") as f:
        f.write(f"# IAB 3.1 Classification Results\n")
        f.write(f"Index: `{index_id}`\n\n")

        for asset_id, results in all_results.items():
            f.write(f"## Video `{asset_id}`\n\n")
            if not results:
                f.write("❌ Failed to process\n\n")
                continue
            f.write(f"| Time | Tier 1 | Tier 2 | Tier 3 | Tier 4 |\n")
            f.write(f"|------|--------|--------|--------|--------|\n")
            for r in results:
                time_str = f"{format_time(r['start'])} – {format_time(r['end'])}"
                if "error" in r:
                    f.write(f"| {time_str} | ❌ {r['error']} | | | |\n")
                else:
                    def fmt(t): return f"{t['name']} ({t['id']})" if t else "—"
                    f.write(f"| {time_str} | {fmt(r['tier1'])} | {fmt(r['tier2'])} | {fmt(r['tier3'])} | {fmt(r['tier4'])} |\n")
            f.write("\n")

    print(f"\n✅ Results saved to {output_path}")


# ─────────────────────────────────────────────
# DB INSPECTION
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 — Haiku Direct: IAB 3.1 classification via Pegasus + Claude Haiku")
    parser.add_argument("--video", metavar="ASSET_ID", help="Analyze a single video asset")
    parser.add_argument("--index", metavar="INDEX_ID", help="Analyze all videos in a TwelveLabs index")
    parser.add_argument("--db", default=DB_PATH, help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()

    if args.video:
        analyze_video(args.video, args.db)
    elif args.index:
        analyze_index(args.index, args.db)
    else:
        parser.print_help()
