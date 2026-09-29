# Ensemble Logistic Regression, Built From Scratch

Tree-structured ensembles of logistic regression models, implemented from raw gradient descent (no `sklearn.linear_model`), that progressively improve on a single classifier by letting sub-models specialize.

## Overview

A single logistic regression model fits one decision boundary to an entire dataset. If the data is really made up of distinct sub-populations, that's a compromise. This project builds and tests three increasingly capable alternatives, each implemented from scratch — sigmoid, log-likelihood cost, and gradients all hand-derived and hand-coded — on a binary wine-quality classification task:

1. A single logistic regression baseline.
2. A 3-node ensemble: one "routing" logistic regression model blends the predictions of two specialized ones.
3. A 7-node deep ensemble: the same idea extended to a 3-layer tree of seven logistic regression nodes.

Each stage measurably improves on the last.

## Key results

| Model | Test accuracy |
|---|---|
| Single logistic regression (Section 1) | 72.66% |
| 3-node ensemble (Section 2) | 75.62% |
| 7-node deep ensemble (Section 3) | 76.09% |

Full derivation of the math behind each model is in [`docs/architecture.md`](docs/architecture.md).

## Tech stack

- **Python**, **NumPy** — every model (weights, sigmoid, cost, gradients, training loop) is implemented directly in NumPy; no `sklearn.linear_model.LogisticRegression` or other pre-built classifier is used.
- **pandas** — data loading and exploration.
- **scikit-learn** — only for `train_test_split`, `StandardScaler`, and `accuracy_score` (data prep and evaluation, not modeling).
- **matplotlib** / **seaborn** — EDA plots and training cost curves.

## Getting started

```bash
git clone https://github.com/VS-2K01/ensemble-logistic-regression-from-scratch.git
cd ensemble-logistic-regression-from-scratch
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Then open `ensemble_logistic_regression.ipynb`. It reads `data/allwine.csv` directly (relative path), so run it from the repo root.

## Project structure

```
ensemble-logistic-regression-from-scratch/
├── ensemble_logistic_regression.ipynb
├── data/
│   └── allwine.csv
├── docs/
│   └── architecture.md
├── LICENSE
└── README.md
```

## Data

`data/allwine.csv` is a wine-quality dataset with 10 physicochemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, density, pH, sulphates, alcohol) and a binary quality label. It arrives already standardized and binarized — see `docs/architecture.md` for details. No personal data; nothing sensitive.

## Results / what I found

Each additional layer of model specialization improved test accuracy — the ensemble idea actually works on this dataset, not just in theory. The biggest single jump came from moving off a single decision boundary to the 3-node ensemble (+2.97 points); the deeper 7-node tree improved further but with diminishing returns, which is a common pattern when adding model capacity.

## Roadmap / possible extensions

- Generalize the tree to an arbitrary number of layers (the assignment's optional bonus task, not attempted here).
- Compare against `sklearn`'s built-in ensemble methods (e.g. `RandomForestClassifier`, `GradientBoostingClassifier`) as a sanity check on how a from-scratch mixture-of-experts approach stacks up against standard library ensembles.

## My role

I implemented this individually (no team). The baseline logistic regression in Section 1 adapts a standard vectorized logistic regression implementation for the core building blocks (sigmoid, cost, gradient descent loop); the ensemble math and code in Sections 2 and 3 — deriving the joint mixture likelihood and the gradients for every node via the chain rule, and implementing the training/prediction logic for both the 3-node and 7-node trees — is my own work.

## License and contact

Licensed under the [MIT License](LICENSE).

**Contact:** [github.com/VS-2K01](https://github.com/VS-2K01)
