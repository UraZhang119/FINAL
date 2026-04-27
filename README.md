# RoNLI Reproduction and Lightweight Extension

## What it Does

This project reproduces a simplified version of the ACL 2024 RoNLI natural language inference benchmark. Given two Romanian sentences, the system predicts their relationship using several models: a majority-class baseline, TF-IDF logistic regression, a custom PyTorch MLP, and an optional fine-tuned multilingual transformer. The project then extends the reproduction through hyperparameter tuning, class balancing, regularization, early stopping, error analysis, and efficiency measurement.

Reference paper: Poesina, Caragea, and Ionescu. 2024. *A Novel Cartography-Based Curriculum Learning Method Applied on RoNLI: The First Romanian Natural Language Inference Corpus*. ACL 2024.

## Quick Start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Smoke test with bundled sample data

```bash
python src/data.py --sample --n-train 24 --n-val 12 --n-test 12
python src/run_all.py --sample-mode
```

### 3. Run the real RoNLI mini-reproduction

```bash
python src/data.py --download --n-train 6000 --n-val 1000 --n-test 1000
python src/train_tfidf.py --model dummy --setting majority
python src/train_tfidf.py --model logreg --setting tfidf_10k --max-features 10000 --ngram-max 2 --C 1.0
python src/train_tfidf.py --model logreg --setting tfidf_30k_tuned --max-features 30000 --ngram-max 2 --C 2.0
python src/train_mlp.py --setting tfidf_mlp --epochs 10 --max-features 5000
python src/error_analysis.py --predictions results/logreg_tfidf_30k_tuned_test_predictions.csv --model logreg_tfidf_30k_tuned
```

## Evaluation

The project was evaluated as a Romanian natural language inference classification task. Each model predicts the relationship between two sentences: `entailment`, `contradiction`, `neutral_related`, or `neutral_unrelated`.

I report the following metrics:

- **Accuracy**: overall fraction of correct predictions.
- **Macro F1**: average F1 across labels, treating all classes equally.
- **Weighted F1**: F1 weighted by class frequency.
- **Macro precision / macro recall**: class-balanced precision and recall.
- **Inference time and examples per second**: approximate prediction speed on the test or validation split.

Because the dataset is imbalanced, **macro F1 is the main comparison metric**. 

### Main Results

The final run used 6,000 training examples, 1,000 validation examples, and 1,000 test examples. The table below reports the main validation and test results.

| Model | Split | Setting | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Train Time (s) | Inference Time (s) | Examples/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Validation | TF-IDF 30K, C=1.0 | 0.619 | 0.422 | 0.628 | 0.427 | 0.428 | 1.420 | 0.021 | 47,486 |
| Logistic Regression | Validation | TF-IDF 10K, C=1.0 | 0.606 | 0.416 | 0.620 | 0.411 | 0.440 | 1.297 | 0.020 | 50,548 |
| Logistic Regression | Validation | TF-IDF 10K, C=0.5 | 0.592 | 0.406 | 0.613 | 0.398 | 0.450 | 1.271 | 0.019 | 52,138 |
| Logistic Regression | Validation | TF-IDF 30K, C=2.0 | 0.626 | 0.404 | 0.632 | 0.421 | 0.403 | 1.479 | 0.021 | 46,999 |
| Logistic Regression | Test | TF-IDF 10K, C=0.5 | 0.550 | 0.401 | 0.573 | 0.392 | 0.461 | 1.271 | 0.019 | 51,485 |
| Logistic Regression | Test | TF-IDF 10K, C=1.0 | 0.565 | 0.397 | 0.582 | 0.396 | 0.431 | 1.297 | 0.030 | 33,695 |
| Logistic Regression | Test | TF-IDF 30K, C=1.0 | 0.577 | 0.382 | 0.587 | 0.402 | 0.388 | 1.420 | 0.021 | 47,273 |
| Logistic Regression | Test | TF-IDF 30K, C=2.0 | 0.585 | 0.345 | 0.590 | 0.374 | 0.352 | 1.479 | 0.021 | 47,753 |
| MLP | Validation | Dropout 0.3 | 0.598 | 0.386 | 0.610 | 0.381 | 0.404 | 2.588 | 0.009 | 110,411 |
| MLP | Validation | Dropout 0.1 | 0.583 | 0.367 | 0.597 | 0.365 | 0.382 | 1.974 | 0.007 | 135,061 |
| MLP | Test | Dropout 0.1 | 0.552 | 0.346 | 0.565 | 0.345 | 0.369 | 1.974 | 0.009 | 111,744 |
| MLP | Test | Dropout 0.3 | 0.539 | 0.326 | 0.554 | 0.333 | 0.339 | 2.588 | 0.007 | 137,419 |
| Dummy Majority | Test | Majority class | 0.626 | 0.192 | 0.482 | 0.157 | 0.250 | 0.000 | 0.000 | 8,774,695 |

### Model Selection

The best validation model was **Logistic Regression with 30K TF-IDF features and C=1.0**, with validation macro F1 of **0.422**. This was selected using the validation split.

On the test set, the strongest observed macro F1 was from **Logistic Regression with 10K TF-IDF features and C=0.5**, with test macro F1 of **0.401**. This model outperformed the dummy baseline by a large margin in macro F1:

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| Dummy Majority | 0.626 | 0.192 | 0.482 |
| Best TF-IDF Logistic Regression | 0.550 | 0.401 | 0.573 |

Although the dummy classifier has higher accuracy, its macro F1 is much lower. This shows that it mainly benefits from class imbalance. The TF-IDF logistic regression model is more useful because it performs better across the full set of labels.

### Hyperparameter Tuning

I performed systematic hyperparameter tuning for the TF-IDF logistic regression model by varying both vocabulary size and regularization strength.

| Setting | Max Features | C | Validation Macro F1 | Test Macro F1 |
|---|---:|---:|---:|---:|
| `tfidf_10k_C05` | 10,000 | 0.5 | 0.406 | 0.401 |
| `tfidf_10k` | 10,000 | 1.0 | 0.416 | 0.397 |
| `tfidf_30k_C1` | 30,000 | 1.0 | 0.422 | 0.382 |
| `tfidf_30k_C2` | 30,000 | 2.0 | 0.404 | 0.345 |

The validation results suggest that increasing the vocabulary size to 30K helped on validation, but did not generalize as well on the test set. The 10K-feature model with stronger regularization, `C=0.5`, achieved the best test macro F1. This suggests that a smaller vocabulary with stronger regularization may reduce overfitting.

### Neural Network Comparison

I also trained a custom PyTorch MLP on TF-IDF features.

| Model | Split | Dropout | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| MLP | Validation | 0.3 | 0.598 | 0.386 | 0.610 |
| MLP | Validation | 0.1 | 0.583 | 0.367 | 0.597 |
| MLP | Test | 0.1 | 0.552 | 0.346 | 0.565 |
| MLP | Test | 0.3 | 0.539 | 0.326 | 0.554 |

The MLP did not outperform logistic regression. This is a useful result because it shows that a more complex neural model does not automatically improve performance when it is trained on the same sparse TF-IDF representation. The logistic regression model was simpler, faster to train, and stronger on macro F1.

### Error Analysis

I ran error analysis on the best test macro-F1 model: **Logistic Regression with 10K TF-IDF features and C=0.5**.

The most common label confusions were:

| True Label | Predicted Label | Count |
|---|---|---:|
| `neutral_unrelated` | `neutral_related` | 10 |
| `neutral_related` | `neutral_unrelated` | 8 |
| `neutral_related` | `entailment` | 4 |
| `entailment` | `neutral_related` | 3 |
| `contradiction` | `neutral_related` | 2 |
| `neutral_unrelated` | `contradiction` | 2 |
| `neutral_related` | `contradiction` | 1 |

The dominant error pattern is confusion between `neutral_related` and `neutral_unrelated`. This makes sense because TF-IDF captures word overlap and topical similarity, but it does not deeply represent sentence meaning or logical relationships.

The model also sometimes predicts `entailment` for examples that are actually `neutral_related`. This suggests that lexical similarity can cause the model to overestimate logical implication.

Overall, the error analysis shows that TF-IDF logistic regression is a strong lightweight baseline, but its main limitation is semantic reasoning. A natural next step would be to fine-tune a multilingual transformer model such as XLM-RoBERTa or multilingual DistilBERT, which should better capture sentence meaning, word order, and logical implication.


## Code Structure

```text
src/data.py              # download/load/clean/split RoNLI
src/train_tfidf.py       # dummy, logistic regression, linear SVM baselines
src/train_mlp.py         # custom PyTorch MLP
src/train_transformer.py # optional multilingual transformer fine-tuning
src/evaluate.py          # metrics, reports, confusion matrix
src/error_analysis.py    # qualitative + quantitative failure analysis
src/run_all.py           # lightweight experiment runner
```

## Video Links

Add links before submission:

- Demo video: https://drive.google.com/file/d/1NwE8b9J78xPq_L6S_7y4GosOUUW8YQj5/view?usp=sharing 
- Technical walkthrough: https://drive.google.com/file/d/13YMIgoPT0lU2a-0_7YbqRlNymPn5dCJa/view?usp=sharing 

## Individual Contributions

All implementation, experiments, documentation, analysis, and videos were completed individually.
