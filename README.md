# Multi-Label Cyberbullying & Online Harassment Detection

A context-aware deep learning system that detects and classifies online harassment and toxic comments across multiple overlapping categories, fine-tuned on 159,000+ real-world social media comments.

---

## 🔬 Overview

Online toxicity rarely fits into a single category — a comment can simultaneously be *toxic*, *obscene*, and *insulting*. This project treats toxicity detection as a **multi-label classification problem** rather than a simple binary one, fine-tuning a transformer encoder to predict multiple co-occurring harassment categories per comment.

## 🧠 Approach

1. **Data** — [Jigsaw Toxic Comment Classification](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) dataset (159,000+ comments) labeled across 6 categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
2. **Preprocessing** — Custom NLP pipeline using NLTK: lowercasing, URL/HTML stripping, stopword removal, and lemmatization on raw social media text.
3. **Model** — Fine-tuned `distilbert-base-uncased` (via HuggingFace Transformers) with a dense classification head, trained with `BCEWithLogitsLoss` to natively support multi-label prediction.
4. **Training** — PyTorch training loop with Adam optimizer, run on GPU, tracking loss over epochs.
5. **Inference** — Real-time prediction function that outputs per-label toxicity probabilities using a sigmoid threshold (≥ 0.5).
6. **Analysis** — Label distribution and inter-label correlation visualizations, plus word clouds contrasting frequent terms in toxic vs. clean comments.

## 📊 Results

- **Validation accuracy:** 97.2%
- **Micro F1-score:** 0.76
- Handled severe class imbalance across labels (e.g., rare `threat` vs. frequent `insult`)

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
