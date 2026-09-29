# Architecture Deep Dive

How each model works, the math behind training it, and how the three sections build on each other.

## The core idea

A single logistic regression model fits one linear decision boundary to the whole dataset. If the data is really made up of several distinct clusters, each with its own relationship between features and label, one boundary is a compromise that fits none of them well. The fix explored here is to let several small logistic regression models specialize on different parts of the input space, and combine their outputs — an ensemble, structured as a tree.

## Section 1: Baseline logistic regression

Standard logistic regression: a linear score passed through a sigmoid,

```
p(x) = 1 / (1 + exp(-(w·x + b)))
```

fit by maximizing the log-likelihood of the observed labels (equivalently, minimizing the negative log-likelihood via gradient descent). This is implemented directly — sigmoid, cost, gradients, and the training loop are all hand-written, with no call to `sklearn.linear_model.LogisticRegression`.

## Section 2: Three-node ensemble

Three logistic regression models are arranged in a small tree:

- `LR_middle` looks at the input `x` and outputs a **routing probability** `h_M(x)` — how strongly this point should be handled by the "left" branch versus the "right" branch.
- `LR_left` and `LR_right` are two independent logistic regression models, each outputting its own classification probability, `h_L(x)` and `h_R(x)`.

The three models are **not trained independently** — they're trained jointly, by treating the routing probability as a mixture weight over the two branches' predictions:

```
P(y=1 | x) = h_L(x) · h_M(x)  +  h_R(x) · (1 − h_M(x))
```

In words: with probability `h_M(x)` the point is routed left and classified by `LR_left`; with probability `1 − h_M(x)` it's routed right and classified by `LR_right`. The overall probability of the observed label is the weighted mixture of both paths.

Training maximizes the joint log-likelihood of this mixture over all three models' parameters at once (30 parameters total — 10 features × 3 models). Each of the three models' gradients is derived separately via the chain rule through the mixture, since none of them has a standalone closed-form loss the way ordinary logistic regression does. Once trained, prediction blends both branches' probabilities directly rather than picking a single branch to route to.

## Section 3: Seven-node deep ensemble

The same idea, extended one layer deeper: a root node and two internal nodes route each point down a small binary tree, ending at one of four leaf models. Each of the four root-to-leaf paths has its own combined routing probability (a product of the routing decisions along that path), and the final prediction is the probability-weighted blend of all four leaves' outputs — the same mixture-of-experts principle as Section 2, just with more branches:

```
P(y=1 | x) = Σ (path routing probability × leaf classification probability)
             over all 4 leaves
```

All seven nodes (1 root + 2 internal + 4 leaves) are trained jointly, again via gradients derived by hand through the chain rule — the same approach as Section 2, generalized to more nodes and paths.

## Results

| Section | Model | Test accuracy |
|---|---|---|
| 1 | Single logistic regression | 72.66% |
| 2 | 3-node ensemble | 75.62% (+2.97 pts) |
| 3 | 7-node deep ensemble | 76.09% (further improvement) |

Each additional layer of specialization improved test accuracy, supporting the core idea: letting sub-models specialize on different regions of the input, rather than fitting one boundary to everything, gives a real accuracy improvement on this dataset.

## Data

`data/allwine.csv` contains 10 physicochemical measurements per wine (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, density, pH, sulphates, alcohol) plus a binary `quality` label. The features arrive already standardized (zero mean, unit variance) and the label already binarized to 0/1 — this is the dataset as provided for the assignment, not a transformation applied in this repo. The notebook additionally re-applies `StandardScaler` after the train/test split, which is a standard (if here redundant, since the input is already scaled) step in the pipeline.

## What's not included

The assignment had an optional bonus task (Task 4: generalizing the tree to an arbitrary number of layers), which wasn't attempted here — everything in this repo is Sections 1 through 3.
