"""
P2 — Pegasus Breadcrumbs
========================
Pegasus segments the video into scenes (with descriptions), then classifies each
scene by picking from all 619 IAB 3.1 leaf breadcrumb paths in a single sync call.
Full paths (e.g. "Sports > Extreme Sports > Climbing") are used instead of bare
leaf names to eliminate semantic ambiguity.

Two rounds per scene:
  Round 1 — All 619 leaf breadcrumbs as a constrained enum (~6,600 tokens).
            If Pegasus picks one → walk parent chain to fill T1–T4.
  Round 2 — Fallback to 37 T1 names if Round 1 returns "None of the above".

Prerequisite: run setup_taxonomy.py once to load the taxonomy into SQLite.

Usage:
    python3 iab_breadcrumbs_pipeline.py --video <ASSET_ID>
    python3 iab_breadcrumbs_pipeline.py --index <INDEX_ID>

Requirements:
    pip install twelvelabs

Environment variables:
    TWELVELABS_API_KEY=tlk_...
"""

import sqlite3
import json
import os
import time
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from twelvelabs import TwelveLabs
from twelvelabs.types.sync_response_format import SyncResponseFormat
from segmentation import get_or_run_scenes

DB_PATH     = "iab_taxonomy_3.1.db"
NONE_OPTION = "None of the above"


# ─────────────────────────────────────────────
# TAXONOMY CACHE
# ─────────────────────────────────────────────

def load_taxonomy_cache(db_path: str = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM iab_taxonomy")
    cache = {row["unique_id"]: dict(row) for row in cur.fetchall()}
    conn.close()
    return cache


def _breadcrumb(node_id: str, cache: dict) -> str:
    """Build full breadcrumb path, e.g. 'Sports > Extreme Sports > Climbing'."""
    parts = []
    node  = cache.get(node_id)
    while node:
        parts.append(node["name"])
        pid  = node.get("parent_id")
        node = cache.get(pid) if pid else None
    return " > ".join(reversed(parts))


def build_leaf_list(cache: dict) -> list:
    """Return all leaf nodes as [{"id", "name", "breadcrumb"}]."""
    all_parent_ids = {r["parent_id"] for r in cache.values() if r.get("parent_id")}
    leaves = [
        {
            "id":         r["unique_id"],
            "name":       r["name"],
            "breadcrumb": _breadcrumb(r["unique_id"], cache),
        }
        for r in cache.values()
        if r["unique_id"] not in all_parent_ids
    ]
    return sorted(leaves, key=lambda x: x["breadcrumb"])


def build_t1_list(cache: dict) -> list:
    """Return all T1 nodes as [{"id", "name"}]."""
    return [
        {"id": r["unique_id"], "name": r["name"]}
        for r in cache.values()
        if r["tier"] == 1
    ]


def resolve_all_tiers(best_id: str, cache: dict) -> dict:
    """Walk parent chain from best_id to fill T1–T4."""
    tiers = {1: None, 2: None, 3: None, 4: None}
    node  = cache.get(best_id)
    while node:
        tier = node.get("tier")
        if tier in tiers:
            tiers[tier] = {"id": node["unique_id"], "name": node["name"]}
        parent_id = node.get("parent_id")
        node = cache.get(parent_id) if parent_id else None
    return tiers


# ─────────────────────────────────────────────
# ENUM CALL — breadcrumbs for leaves, names for T1 fallback
# ─────────────────────────────────────────────

def _pick_leaf(
    tl_client:   TwelveLabs,
    asset_id:    str,
    start:       float,
    end:         float,
    description: str,
    leaf_list:   list,
) -> Optional[dict]:
    """Round 1: enum of all 619 leaf breadcrumb paths. Returns match dict or None."""
    crumbs       = [l["breadcrumb"] for l in leaf_list] + [NONE_OPTION]
    crumb_to_id  = {l["breadcrumb"]: l["id"] for l in leaf_list}

    prompt = (
        f'Scene ({start:.1f}s–{end:.1f}s): "{description}"\n\n'
        f"Select the most specific IAB content category that best describes this scene "
        f'(or "{NONE_OPTION}" if nothing clearly fits), your confidence (0-100), '
        f"your second-best choice, and its confidence."
    )

    try:
        response = tl_client.analyze(
            video={"type": "asset_id", "asset_id": asset_id},
            prompt=prompt,
            response_format=SyncResponseFormat(
                type="json_schema",
                json_schema={
                    "type": "object",
                    "properties": {
                        "first_choice":      {"type": "string",  "enum": crumbs},
                        "first_confidence":  {"type": "integer", "minimum": 0, "maximum": 100},
                        "second_choice":     {"type": "string",  "enum": crumbs},
                        "second_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["first_choice", "first_confidence", "second_choice", "second_confidence"],
                },
            ),
        )
        raw    = response.data
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not parsed:
            return None
        first_crumb  = parsed.get("first_choice")
        second_crumb = parsed.get("second_choice")
        if not first_crumb or first_crumb == NONE_OPTION:
            return None
        first_id  = crumb_to_id.get(first_crumb)
        second_id = crumb_to_id.get(second_crumb) if second_crumb and second_crumb != NONE_OPTION else None
        if not first_id:
            return None
        return {
            "first_id":          first_id,
            "first_confidence":  parsed.get("first_confidence", 0),
            "second_id":         second_id,
            "second_confidence": parsed.get("second_confidence", 0),
        }
    except Exception as e:
        print(f"    ⚠️  Round 1 failed ({start:.0f}–{end:.0f}s): {e}")
        return None


def _pick_t1(
    tl_client:   TwelveLabs,
    asset_id:    str,
    start:       float,
    end:         float,
    description: str,
    t1_list:     list,
) -> Optional[dict]:
    """Round 2: T1 fallback using bare T1 names. Returns match dict or None."""
    names      = [t["name"] for t in t1_list] + [NONE_OPTION]
    name_to_id = {t["name"]: t["id"] for t in t1_list}

    prompt = (
        f'Scene ({start:.1f}s–{end:.1f}s): "{description}"\n\n'
        f'Select the IAB Tier 1 category that best describes this scene '
        f'(or "{NONE_OPTION}" if nothing clearly fits), your confidence (0-100), '
        f"your second-best choice, and its confidence."
    )

    try:
        response = tl_client.analyze(
            video={"type": "asset_id", "asset_id": asset_id},
            prompt=prompt,
            response_format=SyncResponseFormat(
                type="json_schema",
                json_schema={
                    "type": "object",
                    "properties": {
                        "first_choice":      {"type": "string",  "enum": names},
                        "first_confidence":  {"type": "integer", "minimum": 0, "maximum": 100},
                        "second_choice":     {"type": "string",  "enum": names},
                        "second_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["first_choice", "first_confidence", "second_choice", "second_confidence"],
                },
            ),
        )
        raw    = response.data
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not parsed:
            return None
        first_name  = parsed.get("first_choice")
        second_name = parsed.get("second_choice")
        if not first_name or first_name == NONE_OPTION:
            return None
        first_id  = name_to_id.get(first_name)
        second_id = name_to_id.get(second_name) if second_name and second_name != NONE_OPTION else None
        if not first_id:
            return None
        return {
            "first_id":          first_id,
            "first_confidence":  parsed.get("first_confidence", 0),
            "second_id":         second_id,
            "second_confidence": parsed.get("second_confidence", 0),
        }
    except Exception as e:
        print(f"    ⚠️  Round 2 failed ({start:.0f}–{end:.0f}s): {e}")
        return None


# ─────────────────────────────────────────────
# TWO-ROUND SCENE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_scene(
    scene:     dict,
    cache:     dict,
    leaf_list: list,
    t1_list:   list,
    tl_client: TwelveLabs,
    asset_id:  str,
) -> dict:
    start = scene["start"]
    end   = scene["end"]
    desc  = scene.get("scene_description", "")

    def _build_result(match, tiers, second_tiers):
        confidence_gap = match["first_confidence"] - match["second_confidence"]
        # Build breadcrumb strings
        first_parts  = [t["name"] for t in [tiers.get(1), tiers.get(2), tiers.get(3), tiers.get(4)] if t]
        second_parts = [t["name"] for t in [second_tiers.get(1), second_tiers.get(2),
                                             second_tiers.get(3), second_tiers.get(4)] if t]
        first_choice  = " > ".join(first_parts) if first_parts else None
        second_choice = " > ".join(second_parts) if second_parts else None
        return {
            "first_choice":      first_choice,
            "first_confidence":  match["first_confidence"],
            "second_choice":     second_choice,
            "second_confidence": match["second_confidence"],
            "confidence_gap":    confidence_gap,
            "tier1": tiers.get(1),
            "tier2": tiers.get(2),
            "tier3": tiers.get(3),
            "tier4": tiers.get(4),
        }

    # Round 1: breadcrumb enum over all 619 leaves
    match = _pick_leaf(tl_client, asset_id, start, end, desc, leaf_list)
    if match:
        tiers        = resolve_all_tiers(match["first_id"], cache)
        second_tiers = resolve_all_tiers(match["second_id"], cache) if match.get("second_id") else {}
        return _build_result(match, tiers, second_tiers)

    print(f"    Round 1 no match ({start:.0f}–{end:.0f}s) — trying T1 fallback...")

    # Round 2: bare T1 names
    match = _pick_t1(tl_client, asset_id, start, end, desc, t1_list)
    if match:
        tiers        = resolve_all_tiers(match["first_id"], cache)
        second_tiers = resolve_all_tiers(match["second_id"], cache) if match.get("second_id") else {}
        return _build_result(match, tiers, second_tiers)

    print(f"    Round 2 no match ({start:.0f}–{end:.0f}s) — scene unclassifiable")
    return {
        "first_choice": None, "first_confidence": 0,
        "second_choice": None, "second_confidence": 0, "confidence_gap": 0,
        "tier1": None, "tier2": None, "tier3": None, "tier4": None,
    }


# ─────────────────────────────────────────────
# MAIN VIDEO PIPELINE
# ─────────────────────────────────────────────

def format_time(seconds):
    m = int((seconds or 0) // 60)
    s = int((seconds or 0) % 60)
    return f"{m}:{s:02d}"


def _run_single_video(
    asset_id:  str,
    tl_client: TwelveLabs,
    cache:     dict,
    leaf_list: list,
    t1_list:   list,
) -> list:

    print(f"\nAnalyzing video {asset_id}...")
    scenes = get_or_run_scenes(asset_id, tl_client)
    print(f"  {len(scenes)} scenes — classifying...")

    # Classify scenes in parallel (each scene is 1–2 sync calls)
    results_map = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(classify_scene, scene, cache, leaf_list, t1_list, tl_client, asset_id): i
            for i, scene in enumerate(scenes)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                tiers = future.result()
            except Exception as e:
                tiers = {
                    "first_choice": None, "first_confidence": 0,
                    "second_choice": None, "second_confidence": 0, "confidence_gap": 0,
                    "tier1": None, "tier2": None, "tier3": None, "tier4": None,
                }
                print(f"  ❌ Scene {i} failed: {e}")
            results_map[i] = {**scenes[i], **tiers}

    ordered = sorted(results_map.values(), key=lambda r: r["start"])

    def fmt(t):
        return f"{t['name']} ({t['id']})" if t else "—"

    col_w = 170
    print("─" * col_w)
    print(f"  {'Time':<16}{'First choice [conf%/gap:N]':<52}{'Second choice [conf%]':<44}{'Tier 2':<35}{'Tier 3':<35}Tier 4")
    print("─" * col_w)
    for r in ordered:
        time_str = f"{format_time(r['start'])} – {format_time(r['end'])}"
        first  = f"{r.get('first_choice','—')} [{r.get('first_confidence',0)}%/gap:{r.get('confidence_gap',0)}]"
        second = f"{r.get('second_choice','—')} [{r.get('second_confidence',0)}%]"
        print(f"  {time_str:<16}{first:<52}{second:<44}{fmt(r.get('tier2')):<35}{fmt(r.get('tier3')):<35}{fmt(r.get('tier4'))}")
    print("─" * col_w)

    return ordered


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

def save_results_json(asset_id: str, results: list):
    os.makedirs("results", exist_ok=True)
    path = f"results/{asset_id}_v5.json"
    payload = {
        "asset_id": asset_id,
        "pipeline": "v5",
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
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  💾 Saved to {path}")


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINTS
# ─────────────────────────────────────────────

def _init(db_path: str):
    tl_api_key = os.environ.get("TWELVELABS_API_KEY")
    if not tl_api_key:
        print("❌ TWELVELABS_API_KEY is not set.")
        return None
    tl_client = TwelveLabs(api_key=tl_api_key)
    cache     = load_taxonomy_cache(db_path)
    leaf_list = build_leaf_list(cache)
    t1_list   = build_t1_list(cache)
    print(f"  📊 {len(leaf_list)} leaf breadcrumbs (Round 1) | {len(t1_list)} T1 nodes (Round 2 fallback)")
    return tl_client, cache, leaf_list, t1_list


def analyze_video(asset_id: str, db_path: str = DB_PATH):
    init = _init(db_path)
    if not init:
        return None
    tl_client, cache, leaf_list, t1_list = init
    results = _run_single_video(asset_id, tl_client, cache, leaf_list, t1_list)
    if results:
        save_results_json(asset_id, results)
    return results


def analyze_index(index_id: str, db_path: str = DB_PATH):
    init = _init(db_path)
    if not init:
        return None
    tl_client, cache, leaf_list, t1_list = init

    videos = list(tl_client.indexes.videos.list(index_id))
    print(f"Found {len(videos)} videos in index {index_id}\n")

    for video in videos:
        try:
            results = _run_single_video(video.id, tl_client, cache, leaf_list, t1_list)
            if results:
                save_results_json(video.id, results)
        except Exception as e:
            print(f"❌ Video {video.id} failed: {e}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P2 — Pegasus Breadcrumbs: IAB 3.1 classification via constrained leaf enum"
    )
    parser.add_argument("--video", metavar="ASSET_ID", help="Analyze a single video")
    parser.add_argument("--index", metavar="INDEX_ID", help="Analyze all videos in an index")
    parser.add_argument("--db",    default=DB_PATH,    help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()

    if args.video:
        analyze_video(args.video, args.db)
    elif args.index:
        analyze_index(args.index, args.db)
    else:
        parser.print_help()
