# ML classification pipelines: tabular and image

Two end-to-end classification projects, built to answer a question each rather
than to demonstrate a library: **where does the accuracy actually come from?**

In both cases the answer turned out not to be the obvious one. On tabular data,
hyperparameter tuning across 8 configurations bought one tenth of a percentage
point. On images, a convolutional network with 32x the parameters of a plain
dense network scored *the same*, until the optimiser was changed.

## Results at a glance

| # | Problem | Best model | Result | The finding |
|---|---|---|---|---|
| 01 | Income classification (tabular, 40 features) | Logistic Regression, L1, `C=1` | **81.10%** test | Tuning gained +0.1 pt; the simple linear model beat an RBF SVM |
| 01 | Same data, unsupervised | K-Means, `k=2` | **72.05%** test | Only +2.4 pts over the majority-class baseline (69.64%) |
| 02 | Fashion-MNIST (10-class images) | 6-conv CNN + Adam | **91.41%** test | Adam, not depth, was worth the most: roughly 4 pts on the same architecture family |

## Notebooks

### [`01-income-classification-pipeline.ipynb`](notebooks/01-income-classification-pipeline.ipynb)

Census-style demographic data, binary income threshold, 26,215 rows reduced to
21,537 after cleaning, 10 raw columns expanded to 40 features.

**Preprocessing.** Missing values in `workclass` and `occupation` (roughly 1,400
rows each, heavily overlapping) dropped; duplicates removed. Categorical encoding
uses three strategies chosen per column by what the variable means: ordinal
integers for `education` because its levels have a real order, a single binary
column for `sex`, one-hot for the five genuinely unordered columns. The
collinearity check found `relationship_Husband` (r=0.406 with target) to be a
near-restatement of `marital-status_Married` (r=0.459), so the redundant column
was dropped. MinMax scaler fitted on the training split only.

**Classification.** Logistic Regression against SVM, 10-fold cross-validation,
then `GridSearchCV` over penalty/C/solver and kernel/C.

| Model | CV (10-fold) | Test accuracy | Best parameters |
|---|---|---|---|
| Logistic Regression, defaults | 0.8052 | 0.8101 | n/a |
| SVM (RBF), defaults | 0.7955 | 0.7976 | n/a |
| **Logistic Regression, tuned** | 0.8054 | **0.8110** | `C=1, penalty=l1, solver=saga` |
| SVM, tuned | 0.8061 | 0.8045 | `C=10, kernel=rbf` |

**Clustering.** K-Means with `k=2` on the same scaled features, labels withheld.
Prototypes are extracted as the nearest *real record* to each centroid, not the
centroid itself, because averaging one-hot columns produces a point that
corresponds to no actual person.

| | Accuracy |
|---|---|
| K-Means, best cluster-to-label mapping | 0.7205 |
| Majority-class baseline | 0.6964 |
| Tuned Logistic Regression | 0.8110 |

### [`02-fashion-mnist-dense-vs-cnn.ipynb`](notebooks/02-fashion-mnist-dense-vs-cnn.ipynb)

Five Keras models on Fashion-MNIST (60k train, 10k test, 28x28 greyscale), with
fixed seeds and identical preprocessing so that differences are attributable.

| # | Architecture | Params | Optimiser | Epochs | Test accuracy |
|---|---|---|---|---|---|
| 1 | Dense 128 | 101,770 | SGD | 10 | 0.8527 |
| 2 | Dense 256 to 128 | 235,146 | Adam | 10 | 0.8843 |
| 3 | 2 conv + dense 512, max-norm | 3,226,474 | SGD lr=0.002 | 5 | 0.8526 |
| 4 | 4 conv (2 blocks) + dense 512 | 1,676,266 | SGD lr=0.001 m=0.9 | 6 | 0.8763 |
| 5 | **6 conv (3 blocks) + dense 1024 to 512** | 1,997,034 | **Adam** | 8 | **0.9141** |

## What the numbers say

**Model 3 is the useful failure.** It has 32x the parameters of model 1 and the
convolutional structure that is supposed to matter for images, yet it scores
0.8526 against model 1's 0.8527, a dead heat. The diagnosis is in the training
curve, not the architecture: at `lr=0.002` with SGD and 5 epochs it stops while
still climbing steeply (train accuracy 0.8551 and rising, validation loss still
falling). It is undertrained, not incapable. Model 5 proves the point. Same
convolutional family, Adam instead of SGD, and it clears model 4's *final* score
after a single epoch.

Ranked by how much each variable moved the result: **optimiser** (roughly 4 pts)
beats **depth** (roughly 2.4 pts) beats **convolution vs dense** (nothing on its
own, decisive in combination). Parameter count predicted nothing, since the
3.23M-parameter model was joint worst.

**On the tabular side, the flat grid search is the signal.** When 8
configurations all land within 0.1 of each other, the ceiling is in the features,
not the model. The L1 penalty winning is consistent with that: on 40
mostly-sparse one-hot columns, built-in feature selection is the right inductive
bias, and it implies a near-linear decision boundary.

**The K-Means result is reported against its baseline on purpose.** 72.05% sounds
reasonable in isolation; next to a 69.64% majority-class baseline it is worth 2.4
points. The notebook also reports the flipped mapping (27.95%) because K-Means
cluster labels are arbitrary identifiers, so scoring only one of the two mappings
is a way to publish a number that is exactly backwards.

## Tech stack

**Core:** Python 3.12, NumPy, pandas, Matplotlib
**Classical ML:** scikit-learn (LogisticRegression, SVC, KMeans, GridSearchCV, KFold, MinMaxScaler)
**Deep learning:** TensorFlow / Keras (Sequential, Conv2D, MaxPooling2D, Dense, Dropout, max-norm, SGD, Adam)
**Hardware:** notebook 02 was built on a single T4 GPU. It runs on CPU, but the convolutional models will be slow.

## Running these

```bash
git clone <this-repo>
cd ml-classification-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Notebook **02 runs as-is**, since Keras downloads Fashion-MNIST on first use.

Notebook **01 needs its dataset placed in `data/`** first. No data files are
committed here. See [`data/README.md`](data/README.md) for the source, the schema
and the download steps.

## Notes

- Outputs are stripped from the committed notebooks to keep the repository small.
  Every number quoted above is recorded in the markdown cells beside the code
  that produced it.
- Notebook 02 passes `decay=` to `SGD`, which current Keras ignores with a
  warning, so the learning-rate schedule those models were meant to have never
  actually ran. It is left in place rather than silently corrected, since the
  reported numbers are the ones that were measured.

## License

MIT. See [LICENSE](LICENSE).
