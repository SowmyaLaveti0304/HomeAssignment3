# Home Assignment 3 – Summer 2025

**Student Name:** Sowmya Laveti  
**Student ID:** 700771347  
**University:** University of Central Missouri  
**Course:** CS5720 – Neural Networks and Deep Learning

---

## Assignment Overview

This assignment covers key topics in NLP and deep learning:

- **Q1:** Implementing an RNN for Text Generation  
- **Q2:** NLP Preprocessing Pipeline  
- **Q3:** Named Entity Recognition (NER) using spaCy  
- **Q4:** Scaled Dot-Product Attention  
- **Q5:** Sentiment Analysis using HuggingFace Transformers

---

## Q1: Implementing an RNN for Text Generation

**Tasks completed:**
1. Loaded a text dataset (Shakespeare’s corpus).  
2. Converted text to character sequences.  
3. Defined a stateful LSTM-based RNN model.  
4. Trained the model for 10 epochs.  

**Temperature scaling:**
- **T < 1:** Sharpens the softmax, making predictions more deterministic.  
- **T > 1:** Flattens the distribution, increasing randomness and creativity.

---

## Q2: NLP Preprocessing Pipeline

**Tasks completed:**
1. Tokenized the sentence.  
2. Removed English stopwords using NLTK.  
3. Applied stemming with Porter Stemmer.  

**What is the difference between stemming and lemmatization? Provide examples with the word “running.”**
- **Stemming:** Heuristic suffix stripping (Porter): `running` → `run`  
- **Lemmatization:** Dictionary-based, POS-aware:  
  - As verb: `running` → `run`  
  - As noun: `running` → `running`  

**Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?**
- **Useful:** Reduces noise for tasks like topic modeling or keyword extraction.  
- **Harmful:** Loses meaning in tasks needing function words (e.g., sentiment analysis, negation).

---

## Q3: Named Entity Recognition with spaCy

**Tasks completed:**
- Extracted entity text, label, and character offsets from:  
  "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

**How does NER differ from POS tagging in NLP?**
- **POS Tagging:** Annotates every token with a grammatical tag (NN, VB, etc.).  
- **NER:** Identifies spans as entities (PERSON, ORG, DATE, etc.).

**Describe two applications that use NER in the real world**
1. **Financial News:** Extracts companies, tickers, dates, amounts for trading systems.  
2. **Search & Voice Assistants:** Detects locations, dates, organizations to interpret user queries.

---

## Q4: Scaled Dot-Product Attention

**Implementation:**
- Dot product of Q and Kᵀ, scaled by √d, softmaxed, then multiplied by V.  

**Why do we divide the attention score by √d in the scaled dot-product attention formula?**
- Prevents large dot products from creating extremely peaked softmax (avoids vanishing gradients).

**How does self-attention help the model understand relationships between words in a sentence?**
- Each token attends to every other token, capturing long-range dependencies and contextual relevance.

---

## Q5: Sentiment Analysis using HuggingFace Transformers

**Tasks completed:**
- Loaded a pre-trained `sentiment-analysis` pipeline.  
- Classified: "Despite the high price, the performance of the new MacBook is outstanding."

**What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?**
- **BERT:** Encoder-only, bidirectional context understanding.  
- **GPT:** Decoder-only, autoregressive (unidirectional) generation.

**Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.**

1. **Data Efficiency:** Requires less labeled data.  
2. **Compute Savings:** Leverages large-scale training investment.  
3. **Performance:** Achieves higher accuracy and faster convergence on downstream tasks.


