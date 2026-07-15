# Multi-Label Cyberbullying & Online Harassment Detection

A context-aware deep learning system that detects and classifies online harassment and toxic comments across multiple overlapping categories, fine-tuned on 159,000+ real-world social media comments.

---

## 🏗️ System Architecture

![System Architecture](./assets/architecture_diagram.png)

The pipeline covers the full lifecycle: comment retrieval → NLP preprocessing/embedding → transformer-based multi-label classification → context-aware decision rules for borderline cases (e.g., offensive humor vs. genuine harassment) → user-facing output with confidence scores → continuous learning from feedback.

## 🔬 Overview

Online toxicity rarely fits into a single category — a comment can simultaneously be *toxic*, *obscene*, and *insulting*. This project treats toxicity detection as a **multi-label classification problem** rather than a simple binary one, fine-tuning a transformer encoder to predict multiple co-occurring harassment categories per comment.

## 🧠 Approach

1. **Data** — [Jigsaw Toxic Comment Classification](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) dataset (159,000+ comments) labeled across 6 categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
2. **Preprocessing** — Custom NLP pipeline using NLTK: lowercasing, URL/HTML stripping, stopword removal, and lemmatization on raw social media text.
3. **Model** — Fine-tuned `distilbert-base-uncased` (via HuggingFace Transformers) with a dense classification head, trained with `BCEWithLogitsLoss` to natively support multi-label prediction.
4. **Training** — PyTorch training loop with Adam optimizer, run on GPU, tracking loss over epochs.
5. **Inference** — Real-time prediction function that outputs per-label toxicity probabilities using a sigmoid threshold (≥ 0.5).
6. **Analysis** — Label distribution and inter-label correlation visualizations, plus word clouds contrasting frequent terms in toxic vs. clean comments.

## 📊 Dataset Label Distribution

![Label Distribution](./assets/label_distribution.png)

The dataset is heavily imbalanced — `toxic` (15,294) and `obscene` (8,449) comments vastly outnumber rarer but more severe categories like `threat` (478) and `identity_hate` (1,405), motivating the multi-label (rather than single-label) formulation.

## 📉 Training

![Training Loss](./assets/training_loss.png)

Training loss decreased consistently across epochs, from ~0.055 to ~0.038.

## 📈 Validation Results

![Validation Metrics](./assets/validation_metrics.png)

| Metric | Score |
|---|---|
| **Validation Accuracy** | 92.46% |
| **F1 Score (Micro)** | 0.764 |
| **F1 Score (Macro)** | 0.55 |

**Per-label performance:**

| Label | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Toxic | 0.89 | 0.72 | 0.80 | 3056 |
| Severe Toxic | 0.69 | 0.18 | 0.28 | 321 |
| Obscene | 0.85 | 0.82 | 0.84 | 1715 |
| Threat | 0.75 | 0.08 | 0.15 | 74 |
| Insult | 0.74 | 0.75 | 0.75 | 1614 |
| Identity Hate | 0.78 | 0.33 | 0.47 | 294 |

Rare categories (`threat`, `severe_toxic`) show lower recall, a direct consequence of severe class imbalance in the training data.

## 💬 Sample Predictions

![Sample Predictions](./assets/sample_predictions.png)

```
Input: "You are amazing and I love your work!"
→ Safe / Non-Toxic

Input: "You are so stupid, just get off the internet."
→ Toxic (0.9162), Insult (0.5217)

Input: "I will kill you."
→ Toxic (0.6240)
```

## 🛠️ Tech Stack

`PyTorch` · `HuggingFace Transformers (DistilBERT)` · `NLTK` · `Scikit-learn` · `Pandas` · `Matplotlib` · `Seaborn` · `WordCloud`

## 🚀 Run it yourself

The notebook runs end-to-end in Google Colab:

1. Open `cyber_bullying_dectection.ipynb` in Colab (GPU runtime recommended)
2. Run all cells top to bottom — the dataset downloads automatically via `kagglehub`
3. Use `predict_cyberbullying("your text here")` to test the trained model on custom input

## 📄 Notebook

See [`cyber_bullying_dectection.ipynb`](./cyber_bullying_dectection.ipynb) for the full implementation.

---

*Academic research project — NLP Course, 2025. By Faiza Fatima.*
