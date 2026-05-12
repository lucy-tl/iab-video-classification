# IAB Content Taxonomy 3.1 — Video Classification

Three approaches to automatically labeling video content with [IAB Content Taxonomy 3.1](https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/main/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv) categories. The scripts in this repo are written for the TwelveLabs SaaS API and would need to be modified for Bedrock.

---

## Background

Before running any pipeline, the IAB Content Taxonomy 3.1 TSV is loaded once into a local SQLite DB (`iab_taxonomy_3.1.db`) using `setup_taxonomy.py`. All pipelines read from this DB at runtime to resolve matched nodes into their full T1–T4 hierarchy.

All three pipelines share a common first step: **Pegasus 1.5 async segmentation**, which splits the video into scenes and generates a 2–3 sentence visual description per scene. Consecutive scenes flagged as continuations are merged; scenes shorter than 30 seconds are absorbed into neighbors.

**Segmentation prompt (shared):**
```
scene_description: "2-3 sentences describing what is visually happening:
people, actions, setting, objects, and overall tone."

scene_continues: "True if this segment is a continuation of the same broader
topic or activity as the previous segment. Only set to False if the content
changes entirely. Minor changes in camera angle, speaker, or location within
the same topic should remain True."
```

Each classified scene returns a `first_choice` (top IAB breadcrumb), `first_confidence` (0–100), `second_choice` (next best match), `second_confidence`, and `confidence_gap` (first minus second). A high gap means the model was decisive; a gap near 0 means it was uncertain between two options.

---

## Why Claude Haiku?

IAB classification is constrained selection, not open-ended generation — model quality differences matter less than cost and speed. We use **Claude Haiku** (fastest, cheapest Claude tier) because it handles the full 704-node taxonomy in a single structured call.

---

## Pipeline 1 — Haiku Direct (`iab_haiku_pipeline.py`)

1. Pegasus async → scene descriptions
2. Haiku `tool_use` picks from all 704 IAB breadcrumb paths as a constrained enum
3. Selected breadcrumb → walk parent chain to fill T1–T4

**Classification prompt (Haiku):**
```
Scene description: {scene_description}

Select the single most specific IAB Content Taxonomy 3.1 category that best
matches this scene (or 'None of the above' if nothing clearly fits), your
confidence (0-100), your second-best choice, and its confidence.
```
*Sent with 704 breadcrumb paths as enum in `tool_use` `input_schema`. Returns `best_match`, `confidence`, `second_match`, `second_confidence`.*

---

## Pipeline 2 — Pegasus Breadcrumbs (`iab_breadcrumbs_pipeline.py`)

1. Pegasus async → scene descriptions
2. Pegasus sync classify with all 619 leaf breadcrumbs in `response_format.json_schema.enum`
3. If no match → T1 fallback with 37 T1 names
4. Breadcrumb → walk parent chain to fill T1–T4

Full breadcrumb paths (e.g. `"Sports > Extreme Sports > Climbing"`) are used instead of bare leaf names to eliminate semantic ambiguity. The ~6,600 token enum lives in `response_format.json_schema.enum`, not the prompt, which has a 2,000 token limit.

**Classification prompt (Pegasus sync):**
```
Scene (0.0s–45.2s): "{scene_description}"

Select the most specific IAB content category that best describes this scene
(or "None of the above" if nothing clearly fits), your confidence (0-100),
your second-best choice, and its confidence.
```
*Paired with `response_format.json_schema` containing `first_choice`/`second_choice` enums of all 619 leaf breadcrumbs.*

---

## Pipeline 3 — Embedding Similarity (`iab_embeddings_pipeline.py`)

**One-time setup:**
1. Haiku enriches each of the 704 taxonomy nodes → visual scene description + 20 keywords
2. Marengo 3.0 text-embeds the enriched descriptions → saved to `taxonomy_embeds.json`

**Per video:**
1. Pegasus async → scene timestamps
2. Retrieve pre-computed Marengo visual embeddings from the index (generated at index time, variable-length clip segments typically 4–8 seconds)
3. For each Pegasus scene, average all overlapping Marengo segments → scene embedding
4. Cosine similarity against all 704 taxonomy node embeddings → top match → T1–T4

No LLM calls at classification time — just vector math. The video must be in a Marengo-enabled index for visual embeddings to be available.

**Haiku enrichment prompt (one-time, per taxonomy node):**
```
For the IAB Content Taxonomy category below, provide:
1. "scene": 2-3 sentences describing what a VIDEO SCENE in this category
   looks like visually — people, settings, actions, objects, mood, camera style.
2. "keywords": 20 keyword phrases that would appear in video scene descriptions.

Return ONLY JSON: {"scene":"...","keywords":["...",...]}

Category: {breadcrumb}
```

---

## LLM IAB Classification Benchmark

The closest published study ([arXiv:2510.13885](https://arxiv.org/abs/2510.13885), Oct 2025) tested 10 LLMs zero-shot on IAB 2.2 (690 categories, structurally similar to 3.1):

| Model | F1 | Accuracy |
|---|---|---|
| Claude 3.5 Sonnet | 0.55 | 0.52 |
| GPT OSS-120B | 0.53 | 0.55 |
| Gemini 2.0 Flash | 0.52 | 0.54 |
| DeepSeek | 0.52 | 0.51 |
| LLaMA 3.3 70B | 0.51 | 0.43 |

Top models are within ~0.03 F1 of each other. Claude 3.5 Sonnet leads slightly, but the gap is small enough that cost/speed is the deciding factor for a constrained classification task like this.

---

## Setup

### 1. Prerequisites

```bash
pip install anthropic twelvelabs
```

Set environment variables:
```bash
export TWELVELABS_API_KEY=tlk_...
export ANTHROPIC_API_KEY=sk-ant-...   # required for P1 and P3 setup only
```

### 2. Load the taxonomy (one-time)

Download `Content_Taxonomy_3.1.tsv` from the [IAB Taxonomies repo](https://github.com/InteractiveAdvertisingBureau/Taxonomies), then:

```bash
python3 setup_taxonomy.py --load-tsv Content_Taxonomy_3.1.tsv
python3 setup_taxonomy.py --inspect   # verify: should show 704 rows across T1–T4
```

This loads the taxonomy into a local SQLite DB (`iab_taxonomy_3.1.db`) that all three pipelines read from at runtime.

### 3. P3 only: build taxonomy embeddings (one-time)

```bash
python3 iab_embeddings_pipeline.py --setup
```

### 4. Run a pipeline

```bash
python3 iab_haiku_pipeline.py --video <ASSET_ID>
python3 iab_breadcrumbs_pipeline.py --video <ASSET_ID>
python3 iab_embeddings_pipeline.py --video <ASSET_ID> --index-id <INDEX_ID>

# or run on a full index
python3 iab_haiku_pipeline.py --index <INDEX_ID>
python3 iab_embeddings_pipeline.py --index <INDEX_ID>
```
