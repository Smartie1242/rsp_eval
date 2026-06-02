# rsp

`rsp` is the reusable command package extracted from the Resiliparse language-identification research project.

## Contents

- `rsp/`: Python package and CLI modules.
- `label-studio-config.xml`: Label Studio interface used for OWI Slice annotation.
- `reservoir.awk`: deterministic reservoir sampling helper for OWI retrieval samples.

## Usage

Create an environment and install the package in editable mode:

```bash
python -m venv ~/.venv/rsp
source ~/.venv/rsp/bin/activate
pip install -e .
```

For detector extraction, install the optional detector dependencies as well:

```bash
pip install -e ".[detectors]"
```

Commands are run as:

```bash
python -m rsp.cli.prepare_datasets --help
python -m rsp.cli.resiliparse_outputs --help
python -m rsp.cli.publication_visuals --help
```

Main command groups:

- OWI preparation: `prepare_datasets`, `compare_annotations`, `owi_preprocessing`.
- GLC balancing: `balance_glc`.
- Detector extraction: `resiliparse_outputs`.
- Analyses: `score_gap`, `cutoff_sweep`, `text_length_sweep`.
- Publication rendering: `publication_visuals`, `oop_visualise`.
- Orchestration: `pipeline`.

The package expects datasets and generated outputs in the working directory that runs the commands, typically under `data/`, `extracted/`, and `results/`.

## Annotation Assets

Use `label-studio-config.xml` when creating the Label Studio project for OWI Slice annotation. Use `reservoir.awk` to sample fixed-size OWI candidate slices before annotation preparation.

## License`r`n`r`nThis software is released under the MIT License. See `LICENSE`.

