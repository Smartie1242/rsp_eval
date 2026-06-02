# rsp

`rsp` is the reusable command package extracted from the Resiliparse language-identification research project.

## Contents

* `rsp/`: Python package and CLI modules.
* `label-studio-config.xml`: Label Studio interface used for OWI Slice annotation.
* `reservoir.awk`: deterministic reservoir sampling helper for OWI retrieval samples.

## Usage

Create an environment and install the package in editable mode:

```bash
python -m venv \~/.venv/rsp
source \~/.venv/rsp/bin/activate
pip install -e .
```

For detector extraction, install the optional detector dependencies as well:

```bash
pip install -e ".\[detectors]"
```

Commands are run as:

```bash
python -m rsp.cli.prepare\_datasets --help
python -m rsp.cli.resiliparse\_outputs --help
python -m rsp.cli.publication\_visuals --help
```

Main command groups:

* OWI preparation: `prepare\_datasets`, `compare\_annotations`, `owi\_preprocessing`.
* GLC balancing: `balance\_glc`.
* Detector extraction: `resiliparse\_outputs`.
* Analyses: `score\_gap`, `cutoff\_sweep`, `text\_length\_sweep`.
* Publication rendering: `publication\_visuals`, `oop\_visualise`.
* Orchestration: `pipeline`.

The package expects datasets and generated outputs in the working directory that runs the commands, typically under `data/`, `extracted/`, and `results/`.

## Annotation Assets

Use `label-studio-config.xml` when creating the Label Studio project for OWI Slice annotation. Use `reservoir.awk` to sample fixed-size OWI candidate slices before annotation preparation.

## This software is released under the MIT License. See `LICENSE`.

