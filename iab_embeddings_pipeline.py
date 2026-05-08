"""
P3 — Embedding Similarity
=========================
One-time setup: Claude Haiku enriches each of the 704 IAB taxonomy nodes with a
visual scene description and 20 keywords. TwelveLabs Marengo embeds the enriched
text and saves to taxonomy_embeds.json.

Per video: Pegasus generates scene descriptions. Each description is embedded with
Marengo and matched to the closest taxonomy node via cosine similarity. No LLM
calls at classification time.

Prerequisite: run setup_taxonomy.py once to load the taxonomy into SQLite.

Setup (one-time, after taxonomy is loaded):
    python3 iab_embeddings_pipeline.py --setup

Usage:
    python3 iab_embeddings_pipeline.py --video <ASSET_ID>
    python3 iab_embeddings_pipeline.py --index <INDEX_ID>
    python3 iab_embeddings_pipeline.py --inspect   # verify taxonomy_embeds.json

Requirements:
    pip install anthropic twelvelabs

Environment variables:
    ANTHROPIC_API_KEY=sk-ant-...   # used during --setup only
    TWELVELABS_API_KEY=tlk_...
"""

import sqlite3
import json
import os
import math
import time
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
from twelvelabs import TwelveLabs
from segmentation import get_or_run_scenes

DB_PATH         = "iab_taxonomy_3.1.db"
EMBED_DB_PATH   = "taxonomy_embeds.json"


# ─────────────────────────────────────────────
# TAXONOMY: load into memory
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


def resolve_all_tiers(best_match_id: str, cache: dict) -> dict:
    """Walk up the parent chain from best_match_id to fill T1–T4."""
    tiers = {1: None, 2: None, 3: None, 4: None}
    node = cache.get(best_match_id)
    while node:
        tier = node.get("tier")
        if tier in tiers:
            tiers[tier] = {"id": node["unique_id"], "name": node["name"]}
        parent_id = node.get("parent_id")
        node = cache.get(parent_id) if parent_id else None
    return tiers


# ─────────────────────────────────────────────
# SETUP: enrich taxonomy nodes with Haiku + embed with TwelveLabs
# ─────────────────────────────────────────────

def enrich_node_with_haiku(node: dict, haiku_client: anthropic.Anthropic) -> str:
    """
    Enrich a taxonomy node with a visual scene description + 20 keywords using Haiku.
    The enriched text is later embedded so it matches the style of Pegasus scene descriptions.
    """
    breadcrumb = node.get("full_path") or node.get("name", "")

    prompt = f"""You are helping build a video content classification system.
For the IAB Content Taxonomy category below, provide two things:

1. "scene": 2-3 sentences describing what a VIDEO SCENE in this category typically looks like visually — people, settings, actions, objects, mood, camera style. Write it the way a video analysis AI would describe footage.

2. "keywords": 20 specific keyword phrases that would appear in video scene descriptions for this category. Mix visual descriptors, actions, settings, moods, and subject matter. Include synonyms and adjacent concepts. No generic filler like "content" or "media".

Return ONLY JSON with no markdown:
{{"scene":"...","keywords":["...","..."]}}

Category: {breadcrumb}"""

    scene    = ""
    keywords = []
    try:
        response = haiku_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed   = json.loads(raw)
        scene    = parsed.get("scene", "").strip()
        keywords = [k for k in parsed.get("keywords", []) if isinstance(k, str) and k.strip()][:20]
    except Exception as e:
        print(f"  ⚠️  Haiku enrichment failed for {breadcrumb}: {e}")

    # Fallback if Haiku returned too few
    if len(keywords) < 5:
        name = node.get("name", breadcrumb)
        keywords += [f"{name} scene", f"{name} footage", f"{name} visual",
                     f"{name} setting", f"{name} activity"]
        keywords = list(dict.fromkeys(keywords))[:20]

    parts = [f"Hierarchy: {breadcrumb}."]
    if scene:
        parts.append(f"Visual scene: {scene}")
    parts.append(f"Keywords: {', '.join(keywords)}.")
    return " ".join(parts)


def get_text_embedding(text: str, tl_client: TwelveLabs) -> list:
    """Embed a text string using TwelveLabs Marengo. Returns the float vector."""
    resp = tl_client.embed.create(model_name="marengo3.0", text=text)
    return resp.text_embedding.segments[0].float_


def build_taxonomy_embeds(
    db_path: str = DB_PATH,
    embed_db_path: str = EMBED_DB_PATH,
    haiku_client: anthropic.Anthropic = None,
    tl_client: TwelveLabs = None,
    force: bool = False,
):
    """
    One-time setup: enrich all taxonomy nodes with Haiku, embed with Marengo.
    Saves to taxonomy_embeds.json. Skips already-embedded nodes unless force=True.
    """
    cache = load_taxonomy_cache(db_path)
    nodes = [n for n in cache.values() if n.get("full_path")]

    # Load existing embeddings to skip already-done nodes
    existing = {}
    if not force and os.path.exists(embed_db_path):
        try:
            with open(embed_db_path) as f:
                for row in json.load(f):
                    if row.get("embedding") and len(row["embedding"]) > 0:
                        existing[row["iab_id"]] = row
            print(f"  Found {len(existing)} existing embeddings — skipping those")
        except Exception:
            pass
    elif force:
        print("  --force: regenerating all embeddings from scratch")

    missing = [n for n in nodes if n["unique_id"] not in existing]
    print(f"  Enriching {len(missing)} nodes with Haiku...")

    # Step 1: Haiku enrichment in parallel
    enriched = {}
    def enrich(node):
        rich_text = enrich_node_with_haiku(node, haiku_client)
        return node["unique_id"], rich_text

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(enrich, n): n for n in missing}
        done = 0
        for future in as_completed(futures):
            uid, rich_text = future.result()
            enriched[uid] = rich_text
            done += 1
            if done % 50 == 0:
                print(f"    Enriched {done}/{len(missing)}")

    print(f"  Embedding {len(enriched)} nodes with TwelveLabs Marengo...")

    # Step 2: TwelveLabs embedding (sequential to avoid rate limits)
    new_rows = []
    for i, node in enumerate([n for n in nodes if n["unique_id"] in enriched]):
        uid       = node["unique_id"]
        rich_text = enriched[uid]
        try:
            embedding = get_text_embedding(rich_text, tl_client)
            new_rows.append({
                "iab_id":    uid,
                "breadcrumb": node.get("full_path", ""),
                "rich_text": rich_text,
                "embedding": embedding,
            })
            if (i + 1) % 50 == 0:
                print(f"    Embedded {i + 1}/{len(enriched)}")
        except Exception as e:
            print(f"  ⚠️  Embedding failed for {uid}: {e}")

    # Merge with existing and save
    all_rows = list(existing.values()) + new_rows
    with open(embed_db_path, "w") as f:
        json.dump(all_rows, f)

    print(f"  ✅ Saved {len(all_rows)} embeddings to {embed_db_path}")
    return all_rows


# ─────────────────────────────────────────────
# COSINE SIMILARITY SEARCH
# ─────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def find_best_match(scene_embedding: list, embed_db: list) -> Optional[dict]:
    """Return the top-2 taxonomy nodes by cosine similarity to the scene embedding."""
    first       = None
    first_score = -2.0
    second      = None
    second_score = -2.0
    for node in embed_db:
        score = cosine_similarity(scene_embedding, node.get("embedding", []))
        if score > first_score:
            second       = first
            second_score = first_score
            first        = node
            first_score  = score
        elif score > second_score:
            second       = node
            second_score = score
    if not first:
        return None
    result = {**first, "score": first_score}
    if second:
        result["second_id"]    = second.get("iab_id")
        result["second_name"]  = second.get("breadcrumb", "")
        result["second_score"] = second_score
    else:
        result["second_id"]    = None
        result["second_name"]  = None
        result["second_score"] = 0.0
    return result


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

def classify_scene(scene: dict, embed_db: list, cache: dict, tl_client: TwelveLabs) -> dict:
    """Embed a scene description and find the closest taxonomy match via cosine similarity."""
    description = scene["scene_description"]

    embedding = get_text_embedding(description, tl_client)
    match     = find_best_match(embedding, embed_db)

    if not match:
        return {"error": "No match found"}

    iab_id         = match["iab_id"]
    second_iab_id  = match.get("second_id")
    tiers          = resolve_all_tiers(iab_id, cache)
    second_tiers   = resolve_all_tiers(second_iab_id, cache) if second_iab_id else {}
    confidence_gap = round(match.get("score", 0) - match.get("second_score", 0), 4)

    # Note: confidence scores are cosine similarity floats (0.0–1.0), not multiplied by 100
    return {
        "scene_description":  description,
        "first_choice":       match.get("breadcrumb", ""),
        "first_confidence":   round(match.get("score", 0), 4),
        "second_choice":      match.get("second_name", ""),
        "second_confidence":  round(match.get("second_score", 0), 4),
        "confidence_gap":     confidence_gap,
        "tier1":  tiers[1],
        "tier2":  tiers[2],
        "tier3":  tiers[3],
        "tier4":  tiers[4],
    }


# ─────────────────────────────────────────────
# TWELVELABS: ANALYZE VIDEO
# ─────────────────────────────────────────────

def format_time(seconds):
    m = int((seconds or 0) // 60)
    s = int((seconds or 0) % 60)
    return f"{m}:{s:02d}"


def _run_single_video(asset_id: str, tl_client: TwelveLabs, cache: dict, embed_db: list) -> list:
    """Run the embedding pipeline for one video."""

    print(f"\nAnalyzing video {asset_id}...")
    scenes = get_or_run_scenes(asset_id, tl_client)
    print(f"  {len(scenes)} scenes — classifying...")

    results_map = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(classify_scene, scene, embed_db, cache, tl_client): i
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
    print(f"  {'Time':<16}{'First choice [score/gap:N]':<52}{'Second choice [score]':<44}{'Tier 2':<30}{'Tier 3':<30}Tier 4")
    print("─" * col_w)
    for r in ordered:
        time_str = f"{format_time(r['start'])} – {format_time(r['end'])}"
        if "error" in r:
            print(f"  {time_str:<16}❌ {r['error']}")
        else:
            gap    = r.get("confidence_gap", 0)
            first  = f"{r.get('first_choice','—')} [{r.get('first_confidence',0):.3f}/gap:{gap:.4f}]"
            second = f"{r.get('second_choice','—')} [{r.get('second_confidence',0):.3f}]"
            print(f"  {time_str:<16}{first:<52}{second:<44}{fmt(r.get('tier2')):<30}{fmt(r.get('tier3')):<30}{fmt(r.get('tier4'))}")
    print("─" * col_w)

    return ordered


def _init_clients(db_path: str = DB_PATH, embed_db_path: str = EMBED_DB_PATH):
    """Initialize clients and load taxonomy + embedding DB."""
    tl_key  = os.environ.get("TWELVELABS_API_KEY")
    ant_key = os.environ.get("ANTHROPIC_API_KEY")

    if not tl_key:
        print("❌ TWELVELABS_API_KEY is not set.")
        return None

    if not os.path.exists(embed_db_path):
        print(f"❌ {embed_db_path} not found — run: python3 iab_embeddings_pipeline.py --setup")
        return None

    tl_client     = TwelveLabs(api_key=tl_key)
    haiku_client  = anthropic.Anthropic(api_key=ant_key) if ant_key else None
    cache         = load_taxonomy_cache(db_path)

    with open(embed_db_path) as f:
        embed_db = json.load(f)
    print(f"  Loaded {len(embed_db)} taxonomy embeddings")

    return tl_client, haiku_client, cache, embed_db


def save_results_json(asset_id: str, results: list):
    """Save results to results/<asset_id>_p3.json."""
    os.makedirs("results", exist_ok=True)
    path = f"results/{asset_id}_v6.json"
    payload = {
        "asset_id": asset_id,
        "pipeline": "v6",
        "segments": [
            {
                "start":             r["start"],
                "end":               r["end"],
                # first_confidence and second_confidence are cosine similarity floats (0.0–1.0)
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


def analyze_video(asset_id: str, db_path: str = DB_PATH, embed_db_path: str = EMBED_DB_PATH):
    """Run the pipeline on a single video asset."""
    if not asset_id or not asset_id.strip():
        print("❌ No video ID provided.")
        return None
    clients = _init_clients(db_path, embed_db_path)
    if not clients:
        return None
    tl_client, _, cache, embed_db = clients
    results = _run_single_video(asset_id, tl_client, cache, embed_db)
    if results:
        save_results_json(asset_id, results)
    return results


def analyze_index(index_id: str, db_path: str = DB_PATH, embed_db_path: str = EMBED_DB_PATH):
    """Run the pipeline on all videos in a TwelveLabs index."""
    clients = _init_clients(db_path, embed_db_path)
    if not clients:
        return None
    tl_client, _, cache, embed_db = clients

    videos = list(tl_client.indexes.videos.list(index_id))
    print(f"Found {len(videos)} videos in index {index_id}\n")

    all_results = {}
    for video in videos:
        asset_id = video.id
        try:
            results = _run_single_video(asset_id, tl_client, cache, embed_db)
            all_results[asset_id] = results
            if results:
                save_results_json(asset_id, results)
        except Exception as e:
            print(f"❌ Video {asset_id} failed: {e}")
            all_results[asset_id] = None

    output_path = f"results_{index_id}_v4.md"
    with open(output_path, "w") as f:
        f.write(f"# IAB 3.1 Classification Results — Embedding Similarity (v4)\n")
        f.write(f"Index: `{index_id}`\n\n")
        for asset_id, results in all_results.items():
            f.write(f"## Video `{asset_id}`\n\n")
            if not results:
                f.write("❌ Failed\n\n")
                continue
            f.write("| Time | Score | Tier 1 | Tier 2 | Tier 3 | Tier 4 |\n")
            f.write("|------|-------|--------|--------|--------|--------|\n")
            def fmt(t): return f"{t['name']} ({t['id']})" if t else "—"
            for r in results:
                t = f"{format_time(r['start'])} – {format_time(r['end'])}"
                if "error" in r:
                    f.write(f"| {t} | ❌ {r['error']} | | | | |\n")
                else:
                    score = f"{r.get('first_confidence', 0):.3f}"
                    f.write(f"| {t} | {score} | {fmt(r.get('tier1'))} | {fmt(r.get('tier2'))} | {fmt(r.get('tier3'))} | {fmt(r.get('tier4'))} |\n")
            f.write("\n")
    print(f"\n✅ Results saved to {output_path}")


# ─────────────────────────────────────────────
# DB INSPECTION
# ─────────────────────────────────────────────

def inspect_embeds(embed_db_path: str = EMBED_DB_PATH):
    """Print stats about the embedding DB."""
    if not os.path.exists(embed_db_path):
        print(f"❌ {embed_db_path} not found — run --setup first")
        return
    with open(embed_db_path) as f:
        rows = json.load(f)
    total = len(rows)
    with_embed = sum(1 for r in rows if r.get("embedding") and len(r["embedding"]) > 0)
    dim = len(rows[0]["embedding"]) if with_embed else 0
    print(f"Total nodes:       {total}")
    print(f"With embeddings:   {with_embed}")
    print(f"Embedding dims:    {dim}")
    print(f"\nSample entries:")
    for row in rows[:3]:
        print(f"  [{row.get('iab_id')}] {row.get('breadcrumb')}")
        print(f"    {row.get('rich_text', '')[:100]}...")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3 — Embedding Similarity: IAB 3.1 classification via Marengo cosine search")
    parser.add_argument("--setup",   action="store_true", help="One-time: enrich taxonomy with Haiku + embed with TwelveLabs")
    parser.add_argument("--force",   action="store_true", help="With --setup: regenerate all embeddings even if they exist")
    parser.add_argument("--inspect", action="store_true", help="Print taxonomy_embeds.json stats")
    parser.add_argument("--video",   metavar="ASSET_ID",  help="Analyze a single video")
    parser.add_argument("--index",   metavar="INDEX_ID",  help="Analyze all videos in an index")
    parser.add_argument("--db",      default=DB_PATH,     help=f"SQLite DB path (default: {DB_PATH})")
    parser.add_argument("--embeds",  default=EMBED_DB_PATH, help=f"Embeddings JSON path (default: {EMBED_DB_PATH})")
    args = parser.parse_args()

    if args.setup:
        tl_key  = os.environ.get("TWELVELABS_API_KEY")
        ant_key = os.environ.get("ANTHROPIC_API_KEY")
        if not tl_key:
            print("❌ TWELVELABS_API_KEY is not set.")
        elif not ant_key:
            print("❌ ANTHROPIC_API_KEY is not set.")
        else:
            tl_client    = TwelveLabs(api_key=tl_key)
            haiku_client = anthropic.Anthropic(api_key=ant_key)
            build_taxonomy_embeds(args.db, args.embeds, haiku_client, tl_client, force=args.force)

    if args.inspect:
        inspect_embeds(args.embeds)

    if args.video:
        analyze_video(args.video, args.db, args.embeds)

    if args.index:
        analyze_index(args.index, args.db, args.embeds)

    if not any([args.setup, args.inspect, args.video, args.index]):
        parser.print_help()
