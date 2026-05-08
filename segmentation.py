"""
Shared segmentation module.

Runs Pegasus async once per video to produce scene descriptions + timestamps,
caches results to segments/<asset_id>.json. All three pipelines call
get_or_run_scenes() so segmentation runs once and is reused.
"""
import os, json, time
from twelvelabs import TwelveLabs

SEGMENTS_DIR = "segments"
MIN_SCENE_DURATION = 30.0  # seconds — merge shorter scenes into neighbours


def build_segment_definitions() -> list:
    """Segment definition passed to Pegasus async. Splits only on major topic changes."""
    return [
        {
            "id": "iab_classification",
            "description": "A scene in the video.",
            "fields": [
                {
                    "name": "scene_description",
                    "type": "string",
                    "description": (
                        "2-3 sentences describing what is visually happening: "
                        "people, actions, setting, objects, and overall tone."
                    ),
                },
                {
                    "name": "scene_continues",
                    "type": "boolean",
                    "description": (
                        "True if this segment is a continuation of the same broader topic or activity as the previous segment. "
                        "Only set to False if the content changes entirely — for example switching from a sports match to a cooking segment, "
                        "or from a car advertisement to a news broadcast. "
                        "Minor changes in camera angle, speaker, location, or visual style within the same topic should remain True."
                    ),
                },
            ],
        }
    ]


def _parse_raw(raw_segments: list) -> list:
    """Parse raw Pegasus segment objects into plain dicts."""
    parsed = []
    for seg in raw_segments:
        meta = seg.get("metadata", {}) or {}
        parsed.append({
            "start":            seg.get("startTime") or seg.get("start_time") or seg.get("start") or 0.0,
            "end":              seg.get("endTime")   or seg.get("end_time")   or seg.get("end")   or 0.0,
            "scene_description": meta.get("scene_description", ""),
            "scene_continues":  bool(meta.get("scene_continues", False)),
        })
    return parsed


def _collapse(group: list) -> dict:
    descs = [s["scene_description"] for s in group if s.get("scene_description")]
    return {
        "start": group[0]["start"],
        "end":   group[-1]["end"],
        "scene_description": " ".join(descs),
    }


def _merge_on_continues(parsed: list) -> list:
    """Merge consecutive segments where scene_continues=True."""
    if not parsed:
        return []
    merged, group = [], [parsed[0]]
    for seg in parsed[1:]:
        if seg.get("scene_continues"):
            group.append(seg)
        else:
            merged.append(_collapse(group))
            group = [seg]
    merged.append(_collapse(group))
    return merged


def _enforce_min_duration(scenes: list, min_dur: float = MIN_SCENE_DURATION) -> list:
    """
    Merge any scene shorter than min_dur seconds into the next scene
    (or the previous if it's the last scene).
    Repeats until all scenes meet the minimum or only one scene remains.
    """
    if not scenes:
        return scenes
    changed = True
    while changed and len(scenes) > 1:
        changed = False
        result = []
        i = 0
        while i < len(scenes):
            dur = scenes[i]["end"] - scenes[i]["start"]
            if dur < min_dur and i + 1 < len(scenes):
                # merge with next
                merged = _collapse([scenes[i], scenes[i + 1]])
                result.append(merged)
                i += 2
                changed = True
            else:
                result.append(scenes[i])
                i += 1
        scenes = result
    return scenes


def run_segmentation(asset_id: str, tl_client: TwelveLabs) -> list:
    """
    Run Pegasus async segmentation, merge scenes, enforce min duration.
    Returns list of {"start", "end", "scene_description"} dicts.
    Does not cache — use get_or_run_scenes() for cached access.
    """
    print(f"  Running Pegasus segmentation for {asset_id}...")
    task = tl_client.analyze_async.tasks.create(
        model_name="pegasus1.5",
        video={"type": "asset_id", "asset_id": asset_id},
        analysis_mode="time_based_metadata",
        response_format={
            "type": "segment_definitions",
            "segment_definitions": build_segment_definitions(),
        },
        min_segment_duration=10.0,
        max_segment_duration=60.0,
    )
    task_id = task.task_id
    print(f"  Task {task_id} created — polling...")
    poll_start = time.time()
    while True:
        status_obj = tl_client.analyze_async.tasks.retrieve(task_id)
        status = status_obj.status
        elapsed = int(time.time() - poll_start)
        elapsed_str = f"{elapsed}s" if elapsed < 60 else f"{elapsed // 60}m {elapsed % 60:02d}s"
        print(f"    [{elapsed_str:>6}] {status}")
        if status == "ready":
            break
        if status == "failed":
            raise RuntimeError(f"Segmentation task failed: {getattr(status_obj, 'error', {})}")
        time.sleep(5)

    result_data = status_obj.result.data
    if isinstance(result_data, str):
        result_data = json.loads(result_data)
    raw = result_data.get("iab_classification", [])
    print(f"  Pegasus returned {len(raw)} raw segments")

    parsed = _parse_raw(raw)
    scenes = _merge_on_continues(parsed)
    scenes = _enforce_min_duration(scenes)
    print(f"  → {len(scenes)} scenes after merge + min-duration enforcement")
    return scenes


def get_or_run_scenes(asset_id: str, tl_client: TwelveLabs, segments_dir: str = SEGMENTS_DIR) -> list:
    """Return cached scenes for asset_id if available, otherwise run segmentation and cache."""
    os.makedirs(segments_dir, exist_ok=True)
    cache_path = os.path.join(segments_dir, f"{asset_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        scenes = data.get("scenes", [])
        print(f"  Loaded {len(scenes)} cached scenes for {asset_id}")
        return scenes

    scenes = run_segmentation(asset_id, tl_client)

    with open(cache_path, "w") as f:
        json.dump({"asset_id": asset_id, "scenes": scenes}, f, indent=2)
    print(f"  Cached to {cache_path}")
    return scenes
