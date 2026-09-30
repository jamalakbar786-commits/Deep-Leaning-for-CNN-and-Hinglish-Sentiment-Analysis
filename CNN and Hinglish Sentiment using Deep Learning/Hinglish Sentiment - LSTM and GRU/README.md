# Hinglish Code-Mixed Sentiment Classification using LSTM and GRU Networks

## Project Overview
This project focuses on sentiment classification of **Hinglish (Hindi-English code-mixed)** text using **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** neural networks.

Indian users frequently communicate in a mixture of Hindi and English written in Roman script across customer support chats, social media, and messaging platforms. Traditional NLP systems struggle with this type of text because of language switching, inconsistent transliteration, and non-standard spellings.

The objective is to build, train, and compare LSTM and GRU models for multi-class sentiment classification (Positive, Negative, Neutral).

---

## Skills Gained
- Code-mixed text preprocessing
- Language-tag handling
- Tokenization strategy design (word, character, subword)
- Vocabulary construction
- Training embeddings from scratch
- LSTM architecture design
- GRU architecture design
- Model evaluation and comparison
- Error analysis for multilingual NLP systems

---

## Domain
**Conversational AI | Customer Support | Indian Language NLP**

---

## Problem Statement
Standard English NLP tools and pretrained embeddings do not effectively handle Hinglish text because:

- Languages switch within the same sentence.
- Transliteration spellings are inconsistent.
- Vocabulary differs significantly from standard English corpora.

This project develops sequence models capable of understanding and classifying sentiment in code-mixed text used by Indian users.

---

## Business Use Cases
- Automated sentiment triage for customer-support chats
- Social media sentiment monitoring
- Brand reputation tracking
- Feedback analysis for Indian-market applications
- Complaint routing and prioritization systems
- Input signal for intent-classification pipelines

---

## Dataset

### Primary Dataset
**IIITH Hindi-English Code-Mixed Sentiment Dataset**
- Source: Joshi et al. (COLING 2016)
- Size: 3,879 code-mixed comments
- Labels:
  - Positive
  - Negative
  - Neutral

Dataset Repository:
https://github.com/drimpossible/Sub-word-LSTM

### Alternative Dataset
**SemEval-2020 Task 9 (SentiMix Hindi-English)**
- Approximately 20,000 code-mixed tweets
- Word-level language tags
- Sentiment labels

Dataset:
https://competitions.codalab.org/competitions/20654

---

## Project Workflow

### 1. Data Exploration
- Analyze class distribution
- Measure degree of code-mixing
- Inspect data quality

### 2. Text Preprocessing
- Normalize transliteration variants
- Handle emojis, URLs, hashtags
- Clean noisy text
- Optional language tagging

### 3. Tokenization Strategies
Evaluate:
- Word-level tokenization
- Character-level tokenization
- Subword tokenization

### 4. Embedding Training
Train embeddings directly on the code-mixed corpus instead of relying on monolingual pretrained embeddings.

### 5. LSTM Model
- Embedding Layer
- LSTM Layers
- Fully Connected Classification Head

### 6. GRU Model
- Embedding Layer
- GRU Layers
- Fully Connected Classification Head

### 7. Evaluation & Comparison
Compare:
- Accuracy
- Precision
- Recall
- F1 Score
- Training Time
- Parameter Count

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score (Macro)
- Confusion Matrix
- Training Loss Curve
- Validation Loss Curve

---

## Expected Results

The project produces:

1. Trained LSTM sentiment classifier
2. Trained GRU sentiment classifier
3. Comparative performance analysis
4. Error analysis on heavily code-mixed samples
5. Training and evaluation reports

---

## Technical Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Natural Language Processing (NLP)

### Technical Tags
- Hinglish
- Code-Mixed Text
- LSTM
- GRU
- RNN
- Tokenization
- Word Embeddings
- Sentiment Analysis

---

## Deliverables

- Source code repository
- Data preprocessing pipeline
- Tokenization pipeline
- LSTM model implementation
- GRU model implementation
- Trained model files
- Evaluation report and comparison study

---