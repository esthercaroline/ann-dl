# Artificial Neural Networks & Deep Learning

Welcome! This site is my portfolio for the **Artificial Neural Networks and Deep
Learning** course (Insper, 2026.2). It is a single repository that grows one
folder at a time — each exercise and project is added as a self-contained report
with the code that produced it and the figures it shows.

## About me

I'm **Esther Caroline**, a student at Insper. This repository collects my work
throughout the semester as a reproducible, publicly readable portfolio.

## What's here

Each deliverable lives under `docs/exercises/<slug>/` (or `docs/projects/`), with
its report in `index.md`, the scripts under `code/`, and the plots under
`figures/`.

| Deliverable | Status |
| --- | --- |
| [Data](exercises/data/index.md) — preparation & analysis | Complete |
| [Perceptron](exercises/perceptron/index.md) | Not started |
| [MLP](exercises/mlp/index.md) | Not started |
| [VAE](exercises/vae/index.md) | Not started |
| [Projects](projects/index.md) | Not started |

## Reproducing the work

Every report pins a fixed random seed and pulls its code from committed files, so
results can be re-run from a clean checkout:

```bash
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade
mkdocs serve -o
```
