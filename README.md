# dissertation-noise-ml-dl

# MSc dissertation: How Human-Like Noise Shapes Machine Learning Failures (CIFAR-N)

Code and reproducible experiments for my MSc Data Science dissertation
(University of Leeds, MATH5872M). The project studies how the *structured*,
class-conditional character of real human annotation noise affects classifier
robustness — and whether that structure can be exploited to defend against it —
using the CIFAR-10N and CIFAR-100N datasets.

## Overview

The work runs in three phases, all contained in the notebook:

- **Phase 1 — Characterising the noise.** Measures the real human confusion
  structure from CIFAR-N: the transition matrix, semantic locality
  (within-superclass confusion), and per-class error entropy.
- **Phase 2 — Classical classifiers on frozen features.** SVM, Fuzzy SVM, and
  Random Forest trained on frozen ConvNeXt-Tiny embeddings, across five noise
  conditions and a 0–70% sweep. Contains the headline result: a Fuzzy SVM
  significantly outperforms a standard SVM under structured noise but not under
  symmetric noise at the same rate.
- **Phase 3 — Deep learning, end-to-end.** ConvNeXt-Nano trained from scratch
  under four methods (cross-entropy, label smoothing, Co-teaching, DivideMix),
  plus a sweep across noise levels.

## Contents

- `dissertation-noise.ipynb` — the full pipeline, all three phases,
  organised into the same sections as the dissertation.

## Data

- **CIFAR-10 / CIFAR-100** (Krizhevsky, 2009) — standard image benchmarks.
- **CIFAR-10N / CIFAR-100N** (Wei et al., ICLR 2022) — real human annotations
  for the CIFAR training sets.

Both are publicly available; no new human data were collected.

## Reproducibility

Every expensive computation (embeddings, classical sweeps, deep training runs)
is cached to disk under a fixed random seed, so re-running the notebook
reproduces every figure and table exactly rather than recomputing them.

## Environment

Developed on Kaggle with a CUDA GPU. Key libraries: PyTorch, timm,
scikit-learn, NumPy, pandas, matplotlib.
