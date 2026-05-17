# Part 3 — NLP and Sequence Modeling Mini Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![NLP](https://img.shields.io/badge/Task-Sentiment%20Classification-purple) ![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📌 Problem Statement

Build a complete NLP pipeline to classify customer support messages into three sentiment categories:

| Label | Meaning |
|---|---|
| `positive` | Customer is satisfied / happy |
| `neutral` | Informational or no strong sentiment |
| `negative` | Customer is frustrated / complaining |

Two approaches are compared: a **traditional TF-IDF + Logistic Regression baseline** and a deep learning **Bidirectional LSTM** with learned word embeddings.

---

## 📂 Repository Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md                              ← You are here
├── notebook.ipynb                         ← All 6 tasks
├── requirements.txt
├── customer_support_text_classification.csv
│
└── results/
    ├── task1_eda.png                      ← Class distribution + word counts
    ├── task2_top_words.png                ← Top 15 words per class
    ├── task4_baseline_confusion.png       ← Baseline model confusion matrices
    ├── task5_lstm_training.png            ← Bi-LSTM training curves
    ├── model_evaluation.png              ← ✅ Full evaluation dashboard
    └── sample_predictions.txt            ← ✅ 10 inference examples
```

---

## 📊 Dataset Description

**File:** `customer_support_text_classification.csv`

| Property | Value |
|---|---|
| Records | 1,500 customer support messages |
| Columns | ticket_id, channel, customer_message, sentiment_label, word_count, urgent_flag |
| Classes | 3 (positive: 479, neutral: 524, negative: 497) |
| Balance | Nearly balanced — no oversampling needed |
| Avg. word count | ~22 words per message |
| Channels | Chat, Email, Phone, Social |

---

## 🔧 Task 2 — Preprocessing Pipeline

```
Raw Text
  ↓  lowercase
  ↓  remove URLs, ticket numbers, digits
  ↓  remove punctuation
  ↓  tokenize (split on whitespace)
  ↓  remove stopwords (170 English stopwords)
  ↓  remove tokens with length ≤ 2
Clean Token List
```

---

## 🔢 Task 3 — Vectorization

| Method | Shape | Captures Order | Use |
|---|---|---|---|
| **Bag of Words** | (1200, 6000) sparse | No | Naive Bayes baseline |
| **TF-IDF** | (1200, 6000) sparse | No | Logistic Regression baseline |
| **Tokenizer sequences** | (1200, 50) integer | Yes | LSTM input |
| **Embedding layer** | (batch, 50, 64) dense | Yes | Learned representations |

### Why must text be vectorized?
Machine learning models operate on **numbers**, not strings. Vectorization maps each text to a fixed-length numerical array. Without this conversion, no model can process, compare, or classify text.

---

## 🏋️ Task 4 — Baseline Models

| Model | Vectorization | Test Accuracy | Macro F1 |
|---|---|---|---|
| Naive Bayes | Bag of Words | ~74% | ~0.74 |
| **Logistic Regression** | **TF-IDF** | **~79%** | **~0.79** |

Logistic Regression wins the baseline comparison — TF-IDF's IDF weighting suppresses common non-informative words more effectively than raw counts.

---

## 🔁 Task 5 — Sequence Model: Bidirectional LSTM

```
Input sequence  (batch × 50 integer tokens)
      │
Embedding(8000 → 64)      Word embeddings learned from scratch
      │
SpatialDropout1D(0.2)     Regularise embedding features
      │
Bidirectional LSTM(64)    Left-to-right + right-to-left reading
  return_sequences=True   → (batch × 50 × 128)
      │
GlobalMaxPooling1D        → (batch × 128)
      │
Dense(64, ReLU) → Dropout(0.4)
      │
Dense(3, Softmax)         → P(negative), P(neutral), P(positive)

Loss      : Categorical Cross-Entropy
Optimiser : Adam (lr = 0.001)
Stopping  : EarlyStopping (patience=8, restore best weights)
```

**Result:** Bi-LSTM achieves **~84% accuracy / ~0.84 Macro F1** — a meaningful improvement over the TF-IDF baseline due to capturing word order and contextual relationships.

---

## 💬 Task 6 — Attention & Transformer Reflection

### RNN Limitation
Vanilla RNNs compress all context into a single hidden state vector. For long sequences, early information is overwritten — the model **forgets** important context from many steps back.

### LSTM Solution
LSTMs add a dedicated **cell state** with three gates (forget / input / output) that allow gradients to flow unchanged across many steps. This solves the vanishing gradient problem and enables learning dependencies up to ~100 tokens apart.

### Attention Mechanism
Instead of relying on a single compressed context vector, attention computes a **weighted sum of all encoder states** for each decoder step. The model explicitly learns *which* past tokens to focus on — enabling perfect recall of any position regardless of distance.

### Transformers
Replace recurrence entirely with **Multi-Head Self-Attention** — every token attends to every other token simultaneously. This is fully parallelisable (unlike sequential LSTM), scales to billions of parameters, and powers BERT, GPT-4, Claude, Llama-3, and all modern LLMs.

---

## 📊 Final Results

| Model | Accuracy | Macro F1 | Sequence Aware |
|---|---|---|---|
| BoW + Naive Bayes | ~74% | ~0.74 | No |
| TF-IDF + LR | ~79% | ~0.79 | No |
| **Bi-LSTM** | **~84%** | **~0.84** | **Yes** |

---

## 🔮 Inference Examples

```python
predict_sentiment("I am absolutely delighted with the service!")
# → POSITIVE  (91% confidence)

predict_sentiment("My refund has been pending for 3 weeks. Unacceptable.")
# → NEGATIVE  (89% confidence)

predict_sentiment("How do I update my payment method?")
# → NEUTRAL   (82% confidence)
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Or open in **Google Colab** — upload notebook + CSV, then Runtime → Run all.
