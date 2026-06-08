# `rsp` Command Reference

This repository contains the importable research package used to prepare datasets, run detector outputs, compute analysis CSVs, and render publication-ready tables and figures.

Install the standalone package, then run commands from the working directory that contains your `data/`, `extracted/`, and `results/` folders:`r`n`r`n```bash`r`nsource ~/.venv/rsp/bin/activate`r`npip install -e .`r`npython -m rsp.cli.<command>`r`n```

Legacy wrapper scripts have been removed. Use the package commands only.

## Command Index

```text
python -m rsp.cli.pipeline
python -m rsp.cli.prepare_datasets
python -m rsp.cli.compare_annotations
python -m rsp.cli.owi_preprocessing
python -m rsp.cli.balance_glc
python -m rsp.cli.resiliparse_outputs
python -m rsp.cli.score_gap
python -m rsp.cli.cutoff_sweep
python -m rsp.cli.text_length_sweep
python -m rsp.cli.publication_visuals
python -m rsp.cli.oop_visualise
```

## Dataset Layout

Active datasets and annotation files are organized under `data/`:

```text
data/
  OWI_slice/
    frisian/
      raw.jsonl
      cleaned.json
      labelstudio.json
      annotations/
      diff.json
      corrected.json
      enriched.json
    dutch/
    random/
  GLC/
  GLC_balanced/
  WiLI_2018/
  archive/
```

Detector outputs are written to `extracted/`:

```text
extracted/<dataset>/rp_outputs.csv
```

Analysis outputs are written to `results/`:

```text
results/score_gap/
results/cutoff_sweep/
results/length_sweep/
results/publication_visuals/
```

## Package Layout

Reusable functionality lives in `rsp/`:

```text
rsp/
  cli/                  command entrypoints
  datasets.py           dataset registry, loaders, JSON/JSONL/CSV helpers
  detectors.py          Resiliparse, FastText, URL, runtime adapters
  extractors.py         dataset extractors and extractor factory
  languages.py          ISO normalization, language maps, safe tag distance
  metrics.py            metrics, score-gap features, correctness flags
  outputs.py            prediction rows, CSV headers, batch output generation
  plots.py              shared publication plotting helpers and model styles
  preannotation.py      document cleaning and Label Studio task creation
  publication_visuals.py publication table and figure generation
  routing.py            cutoff, composite, and text-length analysis logic
```

## `pipeline`

Purpose: run the default research workflow stages in order. It is useful for repeated experiment regeneration, but it does not replace the human Label Studio annotation/correction gate.

Important: `pipeline` can prepare OWI Label Studio tasks, but OWI enrichment and OWI-based results require `data/OWI_slice/<slice>/corrected.json` files created after manual annotation and correction.

Example:

```bash
python -m rsp.cli.pipeline
```

Outputs depend on selected stages:

```text
data/OWI_slice/<slice>/labelstudio.json
data/OWI_slice/<slice>/enriched.json
extracted/<dataset>/rp_outputs.csv
results/score_gap/
results/cutoff_sweep/
results/length_sweep/
results/publication_visuals/
```

Use `--help` for the current stage flags:

```bash
python -m rsp.cli.pipeline --help
```

## `prepare_datasets`

Purpose: clean OWI raw documents and create Label Studio task files.

This command only prepares annotation inputs. Label Studio annotation and manual correction are required before OWI enrichment and OWI result generation.

Default input and output:

```text
data/OWI_slice/<slice>/raw.jsonl
data/OWI_slice/<slice>/cleaned.json
data/OWI_slice/<slice>/labelstudio.json
```

Example:

```bash
python -m rsp.cli.prepare_datasets
```

Useful flags:

```text
--slice-dir PATH       OWI slice root directory.
--slice NAME=PATH      Optional repeated slice override.
--limit N              Limit documents per slice where supported.
```

Check exact available flags:

```bash
python -m rsp.cli.prepare_datasets --help
```

## `compare_annotations`

Purpose: compare two Label Studio annotation exports and write a disagreement file for manual correction.

Examples:

```bash
python -m rsp.cli.compare_annotations data/OWI_slice/frisian/annotations/marten.json data/OWI_slice/frisian/annotations/timo.json --out data/OWI_slice/frisian/diff.json
python -m rsp.cli.compare_annotations data/OWI_slice/random/annotations/marten.json data/OWI_slice/random/annotations/timo.json --out data/OWI_slice/random/diff.json
```

Dutch currently has only Marten annotations, so there is no Dutch Timo comparison unless that annotation export is actually created later.

Output:

```text
data/OWI_slice/<slice>/diff.json
```

Flags:

```text
--out PATH             Output diff JSON path.
```

## `owi_preprocessing`

Purpose: enrich corrected OWI annotations into final evaluation-ready OWI Slice files.

This command expects human-corrected annotation files:

```text
data/OWI_slice/frisian/corrected.json
data/OWI_slice/dutch/corrected.json
data/OWI_slice/random/corrected.json
```

Example:

```bash
python -m rsp.cli.owi_preprocessing --slice-dir data/OWI_slice
```

Outputs:

```text
data/OWI_slice/<slice>/enriched.json
```

Flags:

```text
--slice-dir PATH       OWI slice root directory.
```

## `balance_glc`

Purpose: create a deterministic balanced GLC dataset from direct Resiliparse dataset output.

Default behavior:

- Preserves the `language/split.txt` layout.
- Balances each split independently.
- Downsamples each language to the smallest language count in that split.
- Uses a deterministic seed.

Example:

```bash
python -m rsp.cli.balance_glc --input data/GLC --output data/GLC_balanced --seed 42
```

Outputs:

```text
data/GLC_balanced/
data/GLC_balanced/balance_manifest.json
```

Flags:

```text
--input PATH           Input GLC dataset directory.
--output PATH          Output balanced dataset directory.
--seed N               Random seed. Default: 42.
--splits LIST          Split names where supported. Default: train,val,test.
```

## `resiliparse_outputs`

Purpose: run Resiliparse, FastText, URL detection, and runtime measurement over configured datasets and write normalized detector output CSVs.

Default output:

```text
extracted/<dataset>/rp_outputs.csv
```

Default datasets include:

```text
GLC
OWI_slice_frisian
OWI_slice_dutch
OWI_slice_random
WiLI_2018
commonlid
```

Run all defaults:

```bash
python -m rsp.cli.resiliparse_outputs
```

Run only CommonLID:

```bash
python -m rsp.cli.resiliparse_outputs --dataset commonlid
```

Override a dataset path:

```bash
python -m rsp.cli.resiliparse_outputs --dataset commonlid=path/to/commonlid.csv
python -m rsp.cli.resiliparse_outputs --dataset GLC=data/GLC_balanced
```

Useful flags:

```text
--dataset NAME         Run a configured dataset by name.
--dataset NAME=PATH    Override or add a dataset path.
--output-root PATH     Output root. Default: extracted.
--limit N              Limit rows where supported.
```

Use `--help` for exact detector/runtime flags:

```bash
python -m rsp.cli.resiliparse_outputs --help
```

## `score_gap`

Purpose: analyze ranking and score-gap behavior in existing `rp_outputs.csv` files, including FastText confidence-gap summaries.

Inputs:

```text
extracted/<dataset>/rp_outputs.csv
```

Example:

```bash
python -m rsp.cli.score_gap
```

Outputs:

```text
results/score_gap/<dataset>/
results/score_gap/<dataset>/score_gap_summary.csv
results/score_gap/<dataset>/ft/
```

Useful flags:

```text
--dataset NAME=PATH    Optional repeated dataset override.
--input-root PATH      Input root for extracted outputs.
--output-root PATH     Output root. Default: results/score_gap.
```

## `cutoff_sweep`

Purpose: evaluate cutoff-based routing behavior for Resiliparse composites over existing detector outputs.

Inputs:

```text
extracted/<dataset>/rp_outputs.csv
```

Example:

```bash
python -m rsp.cli.cutoff_sweep
```

Outputs:

```text
results/cutoff_sweep/<dataset>/cutoff_sweep.csv
results/cutoff_sweep/<dataset>/cutoff_comparison.png
```

Useful flags:

```text
--dataset NAME=PATH    Optional repeated dataset override.
--input-root PATH      Input root for extracted outputs.
--output-root PATH     Output root. Default: results/cutoff_sweep.
--cutoff N             Routing cutoff where supported.
```

## `text_length_sweep`

Purpose: compute text-length performance data from existing detector outputs. This command writes CSV analysis data only; publication plots are generated by `publication_visuals`.

Inputs:

```text
extracted/<dataset>/rp_outputs.csv
```

Example:

```bash
python -m rsp.cli.text_length_sweep
```

Outputs:

```text
results/length_sweep/<dataset>/length_sweep.csv
```

The CSV schema is:

```text
text_length_bin, model, support, coverage, accuracy, f1_macro, hybrid_cutoff
```

Text length is measured in characters. Current bins approximate word-count milestones:

```text
0-300        roughly up to 50 words
300-600      roughly 50-100 words
600-1200     roughly 100-200 words
1200-2400    roughly 200-400 words
2400-4800    roughly 400-800 words
4800+        roughly 800+ words
```

Useful flags:

```text
--dataset NAME=PATH    Optional repeated dataset override.
--input-root PATH      Input root for extracted outputs.
--output-root PATH     Output root. Default: results/length_sweep.
--hybrid-cutoff N      OOP cutoff for hybrid routing. Default: 1200.
```

## `publication_visuals`

Purpose: render publication-ready tables and visuals from existing detector outputs, precomputed metric/confusion files, and analysis CSVs.

Default mode reads `extracted/.../rp_outputs.csv`. Optional normalized inputs can be supplied with `--metrics-file` and `--confusion-file`.

Example:

```bash
python -m rsp.cli.publication_visuals
```

Use precomputed normalized files only:

```bash
python -m rsp.cli.publication_visuals --metrics-file path/to/metrics.csv --confusion-file path/to/confusion.csv --normalized-only
```

Use a length-sweep override:

```bash
python -m rsp.cli.publication_visuals --length-sweep GLC=results/length_sweep/GLC/length_sweep.csv
```

Generate demo data:

```bash
python -m rsp.cli.publication_visuals --generate-demo-data
```

Main outputs:

```text
results/publication_visuals/normalized/metrics.csv
results/publication_visuals/normalized/confusion.csv
results/publication_visuals/normalized/language_similarity.csv
results/publication_visuals/tables/baseline_table.tex
results/publication_visuals/tables/model_dataset_summary.csv
results/publication_visuals/tables/model_dataset_summary.tex
results/publication_visuals/tables/model_dataset_summary_extended.csv
results/publication_visuals/tables/model_dataset_summary_extended.tex
results/publication_visuals/tables/resource_level_summary.csv
results/publication_visuals/tables/resource_level_summary.tex
results/publication_visuals/tables/runtime_summary.csv
results/publication_visuals/tables/runtime_summary.tex
results/publication_visuals/tables/runtime_speedup_comparison.csv
results/publication_visuals/tables/runtime_speedup_comparison.tex
results/publication_visuals/resource_levels/resource_accuracy.pdf
results/publication_visuals/resource_levels/resource_f1_macro.pdf
results/publication_visuals/runtime/runtime_mean_seconds.pdf
results/publication_visuals/text_length/<dataset>/length_comparison.png
results/publication_visuals/text_length/<dataset>/length_comparison_accuracy.png
results/publication_visuals/confusion_matrices/
results/publication_visuals/language_similarity/<dataset>/
results/publication_visuals/language_similarity/oop_score_gap_correctness_summary.csv
results/publication_visuals/language_similarity/oop_score_gap_correctness_summary.tex
```

Current functionality:

- Baseline resource-level LaTeX table.
- Dataset-level model summary table.
- Extended model summary with FastText-backed cutoff composites and language-distance-aware composites. URL/domain language is used only as a routing trigger, never as a fallback prediction. The extended table reports Resiliparse URL trigger, RP+FastText, RP+FastText URL trigger, RP+FastText lang-aware, and RP+FastText lang-aware URL trigger separately so each added trigger can be measured as its own layer.
- Resource-level accuracy and macro-F1 plots.
- Runtime summary, baseline speedup comparison table, and mean runtime plot. The speedup table compares each model against Resiliparse and FastText baselines rather than a full pairwise matrix. URL/domain timing is reported as URL signal where available because it is a routing signal, not a classifier fallback.
- Text-length publication plots from `results/length_sweep/<dataset>/length_sweep.csv`.
- Confusion matrices by dataset, model, and language family. In default rp_outputs.csv mode these use the extended model set, including the separate URL-trigger and language-aware URL-trigger variants when the required routing columns are available.
- Global top-confusion matrices.
- Language-similarity visuals using Resiliparse rank-1 vs rank-2 `langcodes.tag_distance`.

Important language-similarity convention:

- Distance is `tag_distance(rank_1_lang_rp, rank_2_lang_rp)`.
- Distance is treated as a discrete language-tag distance signal in publication plots.
- Correctness is `label == rank_1_lang_rp`.
- Family error heatmaps use true family vs predicted family for incorrect Resiliparse rank-1 predictions.

Useful flags:

```text
--rp-output NAME=PATH      Optional repeated rp_outputs override.
--metrics-file PATH        Precomputed normalized metrics CSV.
--confusion-file PATH      Precomputed normalized confusion CSV.
--normalized-only          Use metrics/confusion inputs without deriving from rp_outputs.
--output-dir PATH          Output directory. Default: results/publication_visuals.
--hybrid-cutoff N          OOP cutoff for hybrid routing. Default: 1200.
--length-sweep NAME=PATH   Optional repeated length_sweep.csv override.
--generate-demo-data       Generate demo normalized inputs.
```

## `oop_visualise`

Purpose: create PCA visualisations from OOP score columns.

Example:

```bash
python -m rsp.cli.oop_visualise --input extracted/GLC/rp_outputs.csv --output-dir results/oop_visualise/GLC
```

Useful flags:

```text
--input PATH           Input CSV.
--output-dir PATH      Output directory.
--score-cols LIST      Comma-separated OOP score columns.
--label-col NAME       Label column.
```

## Label Studio Workflow Reminder

OWI data has a hard manual gate:

1. `prepare_datasets` writes `labelstudio.json`.
2. Human annotators label the tasks in Label Studio.
3. Annotation exports are saved under `data/OWI_slice/<slice>/annotations/`.
4. `compare_annotations` writes `diff.json` for double-annotated slices.
5. Disagreements are manually resolved into `corrected.json`.
6. `owi_preprocessing` writes `enriched.json`.
7. Detector outputs and downstream OWI results can then be generated.

Dutch has only Marten annotations in the current setup; Timo annotations apply to Frisian and random slices.

## Verification Commands

Syntax check:

```bash
python -m compileall rsp
```

Help checks:

```bash
python -m rsp.cli.pipeline --help
python -m rsp.cli.prepare_datasets --help
python -m rsp.cli.compare_annotations --help
python -m rsp.cli.owi_preprocessing --help
python -m rsp.cli.balance_glc --help
python -m rsp.cli.resiliparse_outputs --help
python -m rsp.cli.score_gap --help
python -m rsp.cli.cutoff_sweep --help
python -m rsp.cli.text_length_sweep --help
python -m rsp.cli.publication_visuals --help
python -m rsp.cli.oop_visualise --help
```

## Notes

- Use `GLC_balanced` for balanced GLC experiments.
- `results/length_sweep/<dataset>/length_sweep.csv` must be regenerated after text-length bin changes.
- Publication visuals skip missing optional inputs with compact `[skip]` messages.
- Dataset publication licensing belongs in the standalone dataset repositories.

