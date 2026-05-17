Part 3 — NLP and Sequence Modeling Mini Project

📌 Problem Statement

In this project, I built a complete NLP pipeline to classify customer support messages into different sentiment categories. The objective was to understand how both traditional machine learning methods and deep learning models perform on text classification tasks.

The messages are classified into three sentiment labels:

Label	Meaning
positive	Customer is happy or satisfied
neutral	Informational message with no strong sentiment
negative	Complaint or frustrated response

To compare performance, I implemented:

A traditional TF-IDF + Logistic Regression model
A deep learning Bidirectional LSTM model with trainable embeddings
📂 Repository Structure
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
📊 Dataset Description

Dataset: customer_support_text_classification.csv

The dataset contains 1,500 customer support messages collected from different communication channels such as chat, email, phone, and social media.

Dataset Details
Property	Value
Total Records	1,500
Columns	ticket_id, channel, customer_message, sentiment_label, word_count, urgent_flag
Classes	positive, neutral, negative
Distribution	Nearly balanced
Average Message Length	Around 22 words

Since the dataset was already balanced, I did not apply any oversampling or undersampling techniques.

🔧 Task 2 — Text Preprocessing

Before training the models, I cleaned and processed the raw text data to remove unnecessary information and reduce noise.

Preprocessing Steps
Raw Text
   ↓
Convert to lowercase
   ↓
Remove URLs, numbers, and ticket IDs
   ↓
Remove punctuation
   ↓
Tokenization
   ↓
Remove stopwords
   ↓
Remove very short words
   ↓
Clean Tokens

The preprocessing pipeline helped improve the quality of input text before vectorization and model training.

🔢 Task 3 — Text Vectorization

Machine learning models cannot work directly with text, so the messages were converted into numerical representations using different vectorization methods.

Method	Shape	Captures Word Order	Purpose
Bag of Words	(1200, 6000) sparse matrix	No	Naive Bayes
TF-IDF	(1200, 6000) sparse matrix	No	Logistic Regression
Token Sequences	(1200, 50) integer sequences	Yes	LSTM input
Embedding Layer	(batch, 50, 64) dense vectors	Yes	Semantic learning
Why Vectorization is Needed

Models can only process numbers, not raw text. Vectorization converts each sentence into a numerical format that the model can understand and learn from.

🏋️ Task 4 — Baseline Models

I first trained traditional machine learning models to create a baseline for comparison.

Model	Vectorization	Test Accuracy	Macro F1
Naive Bayes	Bag of Words	~74%	~0.74
Logistic Regression	TF-IDF	~79%	~0.79

Among the baseline models, Logistic Regression performed better because TF-IDF gives lower importance to common words and highlights more meaningful words.

🔁 Task 5 — Bidirectional LSTM Model

After the baseline models, I implemented a Bidirectional LSTM model to capture contextual and sequential information from the text.

Model Architecture
Input Sequence (50 tokens)
        │
Embedding Layer
        │
SpatialDropout1D(0.2)
        │
Bidirectional LSTM(64)
        │
GlobalMaxPooling1D
        │
Dense(64, ReLU)
        │
Dropout(0.4)
        │
Dense(3, Softmax)
Training Details
Parameter	Value
Loss Function	Categorical Cross-Entropy
Optimizer	Adam
Learning Rate	0.001
Early Stopping	Patience = 8
Final Performance

The Bidirectional LSTM achieved:

Accuracy: ~84%
Macro F1 Score: ~0.84

The performance improvement came from the model’s ability to understand word order and contextual relationships between words.

💬 Task 6 — Attention and Transformers
Limitation of Basic RNNs

Traditional RNNs struggle with long sequences because earlier information gradually gets forgotten as the sequence becomes longer.

How LSTM Solves This

LSTMs use memory cells and gates to preserve important information for longer durations, which helps in learning long-term dependencies.

Attention Mechanism

Attention helps the model focus on the most relevant words in a sequence instead of compressing all information into a single vector.

Transformers

Transformers replace recurrence completely with self-attention mechanisms, allowing faster training and better scalability. Modern models such as BERT, GPT, Claude, and Llama are all based on Transformer architectures.

📊 Final Comparison
Model	Accuracy	Macro F1	Sequence Aware
BoW + Naive Bayes	~74%	~0.74	No
TF-IDF + Logistic Regression	~79%	~0.79	No
Bidirectional LSTM	~84%	~0.84	Yes
🔮 Sample Predictions
predict_sentiment("I am really satisfied with the support team.")
# → POSITIVE

predict_sentiment("My refund is still pending after two weeks.")
# → NEGATIVE

predict_sentiment("How can I reset my password?")
# → NEUTRAL
🚀 How to Run
pip install -r requirements.txt
jupyter notebook notebook.ipynb
