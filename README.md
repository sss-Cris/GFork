# GFork: Gated Fusion Network on Multi-Granularity Heterogeneous Pruning Graphs

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

Official implementation of **GFork**: a text classification framework that constructs phrase-level semantic-complete graphs, adds cross-sentence coreference edges, prunes graphs via attribute-enhanced Personalized PageRank (aPPR), and fuses BERT & GNN features with a gradient gating mechanism (G²).

---

## Features

* Phrase-level graphs & long-range dependencies
* Multi-edge types: `co-occurrence`, `syntax`, `coreference`, `same-unit`, `self-loop`
* Graph pruning via aPPR
* G² fusion of BERT & GNN representations
* Benchmark datasets: `20ng`, `R8`, `R52`, `Ohsumed`, `MR`

---

## Quick Start

### 1. Environment

```bash
conda create -n t2g_new python=3.10 -y
conda activate t2g_new
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 2. Dataset

```text
datasets/data/
├── 20ng.txt
└── corpus/20ng.clean.txt
```

### 3. Run Experiment

```bash
./run_exp.sh 20ng 0 Chunk
```

* `Chunk` / `original`: graph construction method
* `0`: GPU index

---

## Core Modules

* `build_graph.py`: phrase-level graph construction
* `ppr.py`: attribute-enhanced PPR pruning
* `train.py`: BERTGNNWithG2, 10-fold CV, metrics: Accuracy & Macro-F1

**Parameters**:

* Training: `--learning_rate`, `--dropout`, `--hidden_dim`, `--steps`, `--g2_p`
* Graph: `--edges`, `--window-size`, `--method`, `--weighted`

---

## Dependencies

```text
torch>=2.1.0
transformers>=4.30.0
spacy>=3.7.0
networkx>=3.0
scikit-learn>=1.3.0
tqdm
scipy
```

---
