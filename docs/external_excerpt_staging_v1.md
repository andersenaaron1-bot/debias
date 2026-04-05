# External Excerpt Staging V1

## Goal

Stage the external corpora that the bundle-creation materializer cannot fetch or normalize automatically.

The staging target is:

- [data/external/bundle_creation_v1](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1)

These staged files are then consumed by:

- [build_bundle_creation_corpus.py](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/src/aisafety/scripts/build_bundle_creation_corpus.py)
- [fetch_bundle_creation_external_inputs.py](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/src/aisafety/scripts/fetch_bundle_creation_external_inputs.py)

## What does not need manual staging

These are already handled separately:

- local `HC3` from [data/HC3](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/HC3)
- local Laurito data from [data/paper](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/paper), [data/movie](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/movie), and [data/product](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/product)
- remote `H-LLMC2` if you use the existing remote loader instead of a staged file

## Files to stage

Create these files under [data/external/bundle_creation_v1](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1):

- `hape_excerpt.jsonl`
- `pubmed_abstracts_excerpt.jsonl`
- `movie_summaries_excerpt.jsonl`
- `amazon_product_descriptions_excerpt.jsonl`
- `paper_llm_supplement_excerpt.jsonl`
- `movie_llm_supplement_excerpt.jsonl`
- `product_llm_supplement_excerpt.jsonl`
- `hc3_plus_excerpt.jsonl`
- `rewrite_control_excerpt.jsonl`

## Minimal JSONL schema

Each line should be one JSON object.

Only `text` is strictly required by the loader, but use the full schema where possible:

```json
{
  "text": "Normalized excerpt text.",
  "title": "Optional title",
  "question": "Optional prompt or seed text identifier",
  "source": "human",
  "generator": null,
  "item_type": "paper",
  "subset": "pubmed",
  "group_id": "optional-stable-group-id",
  "prompt_name": null,
  "meta": {}
}
```

Conventions:

- `source`: `human` or `llm`
- `item_type`: use `paper`, `movie`, `product`, or `general`
- `subset`: use the source family or genre label
- `group_id`: stable when there is a natural pairing or shared seed chunk, otherwise omit it

## Recommended priority order

If you only stage the most important files first, do this order:

1. `hape_excerpt.jsonl`
2. `pubmed_abstracts_excerpt.jsonl`
3. `movie_summaries_excerpt.jsonl`
4. `amazon_product_descriptions_excerpt.jsonl`
5. `hc3_plus_excerpt.jsonl`
6. domain LLM supplements

## Step-by-step by corpus

### 1. HAP-E or HAP-E mini

Where to get it:

- HAP-E dataset card: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
- HAP-E mini: https://huggingface.co/datasets/browndw/human-ai-parallel-corpus-mini

Why it matters:

- broad genre coverage
- includes `academic` and `television/movie script` material
- gives source-labeled human and LLM text under closer continuation control

What to stage:

- prefer `human chunk_2` as the human side
- include the LLM continuation files as `source=llm`
- exclude `human chunk_1` from the main discovery file because it is the seed text

How to map genres:

- `acad` -> `item_type=paper`
- `tvm` -> `item_type=movie`
- all others -> `item_type=general`

Suggested output:

- [hape_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/hape_excerpt.jsonl)

Per-row minimum:

- `text`
- `source`
- `generator`
- `subset` such as `acad`, `news`, `blog`, `fic`, `spok`, `tvm`
- `group_id` from the document/seed id when available

### 2. PubMed abstracts

Where to get it:

- Hugging Face dataset card: https://huggingface.co/datasets/ncbi/pubmed
- NLM homepage: https://www.nlm.nih.gov/databases/download/pubmed_medline.html

Why it matters:

- strengthens paper-domain bundle formation
- gives a large human baseline for abstract register

What to keep:

- records with non-empty `MedlineCitation.Article.Abstract.AbstractText`
- `ArticleTitle` as `title`
- abstract text as `text`
- `source=human`
- `item_type=paper`
- `subset=pubmed`

What to avoid:

- records without abstracts
- extremely short abstracts
- non-English records when easy to filter

Suggested output:

- [pubmed_abstracts_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/pubmed_abstracts_excerpt.jsonl)

### 3. CMU Movie Summary Corpus

Where to get it:

- official corpus page: https://www.cs.cmu.edu/~ark/personas/

Why it matters:

- strengthens movie-domain narrative packaging
- gives a large human synopsis-like baseline

What to keep:

- plot summary text as `text`
- movie title as `title`
- `source=human`
- `item_type=movie`
- `subset=cmu_movie_summary`

What to avoid:

- metadata-only rows without summary text
- very short summaries

Suggested output:

- [movie_summaries_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/movie_summaries_excerpt.jsonl)

### 4. Amazon product descriptions

Where to get it:

- Hugging Face dataset card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- project page: https://amazon-reviews-2023.github.io/main.html

Why it matters:

- strengthens product-domain promotional and packaging cues

What to keep:

- use item metadata, not user reviews, for this file
- combine `title`, `description`, and optionally `features` into `text`
- `source=human`
- `item_type=product`
- `subset` as the Amazon category or `amazon_meta`

What to avoid:

- review text in this staged file
- metadata rows with empty description and empty features

Suggested output:

- [amazon_product_descriptions_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/amazon_product_descriptions_excerpt.jsonl)

### 5. HC3 Plus

Where to get it:

- paper page: https://huggingface.co/papers/2309.02731

Why it matters:

- semantic-invariant confirmation corpus for summarization, paraphrase, and translation

What to stage:

- human and model outputs as independent rows
- preserve any shared prompt or task identifier in `group_id`
- `item_type=general`
- `subset` from the task name when available

Suggested output:

- [hc3_plus_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/hc3_plus_excerpt.jsonl)

Note:

- this one is confirmation-only in the spec, so if obtaining it is annoying, skip it temporarily

### 6. Domain LLM supplements

When to create them:

- only if paper/movie/product remain weak on the LLM side after staging the human baselines

What to do:

- generate templated excerpts from metadata or titles
- one output per prompt
- use two open-weight instruct models if possible

Outputs:

- [paper_llm_supplement_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/paper_llm_supplement_excerpt.jsonl)
- [movie_llm_supplement_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/movie_llm_supplement_excerpt.jsonl)
- [product_llm_supplement_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/product_llm_supplement_excerpt.jsonl)

Rule:

- keep these clearly marked as supplements in `meta`
- do not let them dominate the discovery core

### 7. Rewrite control

What to stage:

- your local content-preserving rewrite corpus in normalized row format

Suggested output:

- [rewrite_control_excerpt.jsonl](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/external/bundle_creation_v1/rewrite_control_excerpt.jsonl)

## How to add the files

### Automated path

The repo now includes an automated fetcher for the four main external inputs:

- HAP-E
- PubMed abstracts
- CMU Movie Summary Corpus
- Amazon product metadata

Run:

```powershell
& .\.venv\Scripts\python.exe -m aisafety.scripts.fetch_bundle_creation_external_inputs --out-dir data\external\bundle_creation_v1 --cache-dir data\external\.cache\bundle_creation_v1
```

This writes:

- `hape_excerpt.jsonl`
- `pubmed_abstracts_excerpt.jsonl`
- `movie_summaries_excerpt.jsonl`
- `amazon_product_descriptions_excerpt.jsonl`

It does not currently fetch:

- `hc3_plus_excerpt.jsonl`
- domain LLM supplements
- `rewrite_control_excerpt.jsonl`

Those still need to be staged manually if you want them.

### Manual path

1. Create the folder:

```powershell
New-Item -ItemType Directory -Force -Path data\external\bundle_creation_v1 | Out-Null
```

2. Save each normalized JSONL there with the exact filenames above.

3. Re-run the materializer:

```powershell
& .\.venv\Scripts\python.exe -m aisafety.scripts.build_bundle_creation_corpus --spec-json configs\datasets\bundle_creation_corpus_spec_v1.json --out-dir data\derived\bundle_creation_corpus_v1 --hc3-dir data\HC3 --remote-hllmc2-sources finance,medicine,open_qa,reddit_eli5 --remote-hllmc2-max-groups-per-source 800 --remote-hllmc2-cache-dir data\external\.cache\bundle_creation_v1\hllmc2 --hape-jsonl data\external\bundle_creation_v1\hape_excerpt.jsonl --pubmed-jsonl data\external\bundle_creation_v1\pubmed_abstracts_excerpt.jsonl --movie-summary-jsonl data\external\bundle_creation_v1\movie_summaries_excerpt.jsonl --product-jsonl data\external\bundle_creation_v1\amazon_product_descriptions_excerpt.jsonl --paper-llm-jsonl data\external\bundle_creation_v1\paper_llm_supplement_excerpt.jsonl --movie-llm-jsonl data\external\bundle_creation_v1\movie_llm_supplement_excerpt.jsonl --product-llm-jsonl data\external\bundle_creation_v1\product_llm_supplement_excerpt.jsonl --hc3-plus-jsonl data\external\bundle_creation_v1\hc3_plus_excerpt.jsonl --rewrite-jsonl data\external\bundle_creation_v1\rewrite_control_excerpt.jsonl
```

4. Inspect the missing-input ledger in:

- [summary.json](C:/Users/King%20Kong/Desktop/AIAIBIAS/AISafety/data/derived/bundle_creation_corpus_v1/summary.json)

If a file is still missing or empty, it will be listed there explicitly.
