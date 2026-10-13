# Datasets

**No data files are committed to this repository.** The `.gitignore` excludes
`*.csv` and `*.xlsx`, so anything you place in this folder stays untracked.

## Required download for notebook 01

| | |
|---|---|
| Save as | `data/income.csv` |
| Rows | 26,215 raw, reduced to 21,537 after dropping missing values and duplicates |
| Columns | 10 raw, expanded to 40 after encoding |
| Target | `income`, binary (0 / 1); 29.9% positive |

### Schema

| Column | Type | Notes |
|---|---|---|
| `income` | int | Target. `0` / `1` |
| `age` | int | 17 to 90 |
| `workclass` | categorical | 7 levels (Private, Self-emp-not-inc, Local-gov, State-gov, Self-emp-inc, Federal-gov, Without-pay). Roughly 1,400 missing |
| `education` | categorical, **ordered** | 16 levels, Preschool through Doctorate. Mapped to integers 1 to 16 |
| `marital-status` | categorical | 4 levels (Married, NotMarried, Separated, Widowed) |
| `occupation` | categorical | 14 levels. Roughly 1,400 missing |
| `relationship` | categorical | 6 levels |
| `race` | categorical | 5 levels |
| `sex` | binary | Mapped Male to 0, Female to 1 |
| `hours-per-week` | int | 1 to 99 |

### Source

This is a cleaned, reduced form of the **UCI Adult / Census Income** dataset:

- UCI repository: https://archive.ics.uci.edu/dataset/2/adult
- Kaggle mirror: https://www.kaggle.com/datasets/uciml/adult-census-income

> **The original UCI file is not a drop-in replacement.** The version the
> notebook was built against differs from the raw UCI data in three ways. The
> target is already binarised to `0`/`1`, whereas UCI ships it as the string
> `<=50K` / `>50K`. The `marital-status` column is collapsed from 7 levels to 4
> (Married, NotMarried, Separated, Widowed). The raw UCI columns `fnlwgt`,
> `education-num`, `capital-gain`, `capital-loss` and `native-country` are
> absent.
>
> To reproduce the exact numbers in the notebook you would need to apply those
> same transformations. Running it on the raw UCI file will work, but will give
> different figures.

The UCI Adult dataset is derived from the 1994 US Census and is released for
public research use. It is a well-known benchmark for fairness analysis, since
`sex` and `race` are among its features and the income label correlates with
both. That is worth keeping in mind before drawing conclusions from any model
trained on it.

## Downloaded automatically for notebook 02

| Asset | Fetched via | Size |
|---|---|---|
| Fashion-MNIST (60,000 train, 10,000 test, 28x28 greyscale, 10 classes) | `keras.datasets.fashion_mnist.load_data()` | ~30 MB |

No manual step. Keras caches it under `~/.keras/datasets/` after the first run.
Released by Zalando Research under the MIT licence:
https://github.com/zalandoresearch/fashion-mnist

## Licensing

Each dataset stays under its own licence and terms, so check the source links
before reusing any of them. The MIT licence on this repository covers the code in
`notebooks/` only, not any data you download.
