# Part 3 — NLP and Sequence Modeling Mini Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![NLP](https://img.shields.io/badge/Task-Sentiment%20Classification-purple)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

# 📌 Project Overview

This project explores how Natural Language Processing (NLP) can be used to classify customer support messages based on sentiment.

The main objective is to automatically identify whether a customer message is:
- Positive
- Neutral
- Negative

To understand how different NLP approaches perform, I implemented and compared:
1. A traditional machine learning pipeline using **TF-IDF + Logistic Regression**
2. A deep learning sequence model using a **Bidirectional LSTM**

The project covers the complete NLP workflow including preprocessing, vectorization, model building, evaluation, and inference.

---

# 📂 Project Structure

```text
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── customer_support_text_classification.csv
│
└── results/
    ├── task1_eda.png
    ├── task2_top_words.png
    ├── task4_baseline_confusion.png
    ├── task5_lstm_training.png
    ├── model_evaluation.png
    └── sample_predictions.txt
```

---

# 📊 Dataset Information

### Dataset File
`customer_support_text_classification.csv`

The dataset contains customer support conversations collected from multiple communication channels.

| Property | Details |
|---|---|
| Total Records | 1,500 messages |
| Classes | Positive, Neutral, Negative |
| Average Length | ~22 words |
| Channels | Chat, Email, Phone, Social Media |
| Dataset Balance | Nearly balanced |

### Label Distribution

| Sentiment | Count |
|---|---|
| Positive | 479 |
| Neutral | 524 |
| Negative | 497 |

Since the classes are already balanced, no oversampling or class weighting was required during training.

---

# 🧹 Text Preprocessing

Before training the models, the raw text messages were cleaned and standardised.

### Preprocessing Steps

```text
Raw Message
   ↓
Convert to lowercase
   ↓
Remove URLs and numbers
   ↓
Remove punctuation
   ↓
Tokenization
   ↓
Remove stopwords
   ↓
Remove very short words
   ↓
Cleaned Text
```

### Why Preprocessing Matters
Raw customer messages usually contain noise such as links, repeated punctuation, and unnecessary words. Cleaning the text helps the model focus on meaningful information and improves overall prediction quality.

---

# 🔢 Text Vectorization

Machine learning models cannot directly understand text, so each message must first be converted into numerical form.

Different vectorization techniques were tested during the project.

| Method | Output Shape | Captures Word Order | Usage |
|---|---|---|---|
| Bag of Words | Sparse matrix | No | Naive Bayes |
| TF-IDF | Sparse matrix | No | Logistic Regression |
| Token Sequences | Integer sequences | Yes | LSTM Input |
| Embedding Layer | Dense vectors | Yes | Deep Learning |

---

# 🤖 Baseline Machine Learning Models

Two traditional NLP baselines were implemented first.

| Model | Vectorization | Accuracy | Macro F1 |
|---|---|---|---|
| Naive Bayes | Bag of Words | ~74% | ~0.74 |
| Logistic Regression | TF-IDF | ~79% | ~0.79 |

### Observation
Logistic Regression performed noticeably better because TF-IDF reduces the impact of very common words and gives more importance to informative terms.

Even though the baseline models worked reasonably well, they could not understand word order or contextual meaning.

---

# 🔁 Deep Learning Model — Bidirectional LSTM

To improve performance, a Bidirectional LSTM model was trained using tokenized sequences.

### Model Architecture

```text
Input Text Sequence
        ↓
Embedding Layer
        ↓
SpatialDropout1D
        ↓
Bidirectional LSTM
        ↓
GlobalMaxPooling1D
        ↓
Dense Layer + Dropout
        ↓
Softmax Output Layer
        ↓
Sentiment Prediction
```

### Why Bidirectional LSTM?
A normal neural network treats text as independent features, but LSTMs process words sequentially. A Bidirectional LSTM reads the sentence in both directions, helping the model better understand context and sentence structure.

### Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Categorical Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Regularization | Dropout |
| Early Stopping | Enabled |

---

# 📈 Model Performance

| Model | Accuracy | Macro F1 | Understands Sequence |
|---|---|---|---|
| BoW + Naive Bayes | ~74% | ~0.74 | No |
| TF-IDF + Logistic Regression | ~79% | ~0.79 | No |
| Bi-LSTM | ~84% | ~0.84 | Yes |

### Final Observation
The Bi-LSTM achieved the best results because it captures contextual relationships and understands word order, unlike traditional vector-based approaches.

---

# 💡 Understanding LSTM and Attention

## Limitation of Traditional RNNs
Basic RNNs struggle with long sentences because earlier information gradually gets forgotten during sequence processing.

## How LSTM Improves This
LSTMs solve this problem using memory cells and gating mechanisms that allow important information to persist across longer sequences.

## Attention Mechanism
Attention allows the model to focus on the most relevant words in a sentence rather than compressing everything into a single representation.

## Transformers
Modern architectures like BERT and GPT replace recurrence completely using self-attention mechanisms. This allows faster parallel training and better understanding of language context.

---

# 🔮 Sample Predictions

```python
predict_sentiment("I am very happy with the customer support.")
# POSITIVE

predict_sentiment("My issue still has not been resolved.")
# NEGATIVE

predict_sentiment("Can you help me update my account details?")
# NEUTRAL
```

---

# 🚀 How to Run the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch Notebook

```bash
jupyter notebook notebook.ipynb
```

You can also run the notebook directly in Google Colab by uploading:
- `notebook.ipynb`
- Dataset CSV file

Then select:

```text
Runtime → Run All
```

---

# 📚 References

1. Vaswani et al. (2017). *Attention Is All You Need.*
2. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.*
3. Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.*
4. TensorFlow Documentation
5. Scikit-learn Documentation
