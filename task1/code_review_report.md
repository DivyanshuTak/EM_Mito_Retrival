# Code Review Report

**TLDR:** For a bigger
project with 3-4 collaborators this code will become hard to maintain because too
many things are hardcoded (paths, model hparams, training hparams, output
paths, split logic). Main fix is to move to config-driven runs, split modules
cleanly, and track experiments so ablations are easy and reproducible.

## Disclosure

For this task, GPT-4 was used to refine language and correct grammatical
mistakes. The review itself, including the ideas and recommendations, reflects
my independent thinking and experience.

## 1. GENERATE_DATA.PY

For synthetic data generation this file is okay, but for team experiments I
would separate concerns more clearly. One function should handle sequence/data
loading, another should handle fitness score construction/transform, and noise
logic should be its own configurable block. In your current structure these
things are mixed, so if someone wants to change just one part (for example
fitness mapping or perturbation style), they still have to edit core flow code.

The reason I would split this is exactly for ablations and collaboration:
different people can swap sequence source, score logic, or noise setup without
touching the same lines. That reduces merge conflicts and keeps dataset logic
clean. It also helps when moving from synthetic to real datasets, because you
can keep the same interface and only switch the loader implementation.

I would pass these choices through config/CLI: sequence length, sample count,
noise type (gaussian, laplacian etc.), noise strength, and output path.
Then collaborators can run controlled comparisons without editing python each
time, and anyone reading the config immediately understands what was changed and
why for that run.

## 2. TRAINER.PY

This is the biggest place to refactor. Model architecture params, tokenizer
mapping, pad length, split logic, optimizer config, checkpoint names, and plot
paths are mixed inside one script. It works, but this hardcoding params makes team
experiments messy and untrackable.

I would keep the same core classes but drive them from a config file. In that
config file, I would define clear sections like `Data`, `Model`, `Training`,
`Eval`, `Logger`, and `Outputs`, and each section would contain its respective
parameters. `ProteinDataset` should take vocab mapping + max length from config.
Split should be seed-based or from a predefined split file for reproducibility.
Output artifacts should go to run-specific directories (`outputs/<run_id>/...`)
so collaborators do not overwrite each other.

Logging should move from print-only to structured logging. I would definitely
add wandb so multiple collaborators can track train/val curves and compare runs cleanly.

## 3. RUN_TRAINER.PY

This should stay as the main entrypoint, but i would also expose a cleaner CLI. Something
like:

`python run_trainer.py --config config.yaml --data data/protein_fitness.csv`

This makes runs easy to reproduce and easy to automate for sweeps/ablations.
People should not have to edit source files every time they create a run.

## Things that I would add to make the project modular and allow efficient collaboration

### 1. CONFIG

Config is the most important upgrade here. I would make config the single source
of truth for data params, model hparams, training setup, eval options, logging,
and output paths. This removes hardcoded values from scripts and makes ablations
easy to run, review, and reproduce across collaborators. 

I would also define wandb project name and run name from config params (such as model + dataset + seed + ablation tag), so each run is consistently
named, tracked, and comparable on the W&B portal without manual renaming.

### 2. EXPERIMENT REGISTRY

Besides config, I would add a simple run manifest `runs.json`
that stores run_id, git commit hash, config path, metrics, and output folder.
This makes it easy to track and log all runs over time, and later quickly find
and extract any specific run we need.

### 3. MODULE SPLIT + UTILS

I would split the code into focused modules (`data.py`, `model.py`, `trainer.py`, `metrics.py`,
`utils.py`) and keep `run_trainer.py` minimal. Utility code such as plotting, JSON I/O,
small helpers should be in utils, not mixed with core training flow.

### 4. REPRODUCIBILITY 

Set and log seeds either of Python/NumPy/PyTorch, and log package versions using environment manager such as UV.

### 5. SANITY CHECK TESTS

I would also just add a few basic checks: make sure the
data loader returns expected shapes, run one model forward pass, and run one
short train-step as sanity check in integration. These simple checks will catch most issues
early when multiple people are editing the code.

### 6. DOCUMENTATION / USAGE GUIDE

I would add a proper README + usage guide for collaborators. It should clearly
explain project structure, how to run training/eval with config, what each config
section means, where outputs/checkpoints are saved, and common troubleshooting
steps. I would also add a small contribution workflow (branch naming, PR style,
how to log experiments), so new teammates can start quickly without back-and-forth.

