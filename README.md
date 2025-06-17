# Home Assignment 3 – Summer 2025  
Student Name: Sowmya Laveti  
Student ID: 700771347  
University: University of Central Missouri  
Course: CS5720 – Neural Networks and Deep Learning  

---

## Assignment Overview  
This assignment covers various key topics in NLP and deep learning including:

- Implementing an RNN for Text Generation
- Basic NLP preprocessing (tokenization, stopword removal, stemming)
- Named Entity Recognition (NER) using SpaCy
- Scaled Dot-Product Attention (as used in Transformers)
- Sentiment analysis using HuggingFace Transformers
## Q1: Implementing an RNN for Text Generation
### Tasks completed
- Text dataset is loaded
- Text is converted into a sequence of characters
- RNN model is defined using LSTM layers
- Model is trained with 10 EPOCHS
**Explain the role of temperature scaling in text generation and its effect on randomness.**
- Lower T<1 makes the distribution “peaky,” so the highest‐scoring characters dominate and output is more deterministic.
- Higher T>1 flattens the distribution, giving lower‐scoring characters a bigger chance and producing more random, creative (but potentially incoherent) text.

## Q2: NLP Preprocessing Pipeline
### Tasks completed
- Tokenized input sentence
- Removed common English stopwords using NLTK
- Applied stemming using Porter Stemmer
- What is the difference between stemming and lemmatization? Provide examples with the word “running.”
  Stemming is a crude, rule-based chop of word endings.
  ```plaintext
  from nltk.stem import PorterStemmer
PorterStemmer().stem("running")
```
Lemmatization uses a vocabulary and POS info to return a valid dictionary form.
```plaintext
  from nltk.stem import WordNetLemmatizer
  lemm = WordNetLemmatizer()
  lemm.lemmatize("running", pos="v")  # → "run"
  lemm.lemmatize("running", pos="n")
```
- Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
Useful when… you want to reduce noise and dimensionality in tasks focused on “what” rather than how. Harmful when… function words carry meaning or structure, e.g. sentiment analysis.
## Q3: Named Entity Recognition with SpaCy 
### Tasks completed
- Entity text, Entity label, start and end character positions.
string used: "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."
 **How does NER differ from POS tagging in NLP?**
- POS Tagging assigns every word a grammatical category to analyze sentence structure.
- NER detects and classifies multi-word spans as real-world entities focusing on semantic information rather than syntax.
 **Describe two applications that use NER in the real world (e.g., financial news, search engines).**
- Financial News Analysis: Automatically extracts entities like company names, stock symbols, dates, and monetary amounts from news articles to feed trading algorithms and generate market alerts.
- Search Engines & Voice Assistants: Identifies locations, dates and organizations in user queries to map intent—e.g., finding flights, weather forecasts, or booking appointments.
## Q4: Scaled Dot-Product Attention 
Implement the scaled dot-product attention mechanism.
**Why do we divide the attention score by √d in the scaled dot-product attention formula?**
The raw dot‐products of Q and K grow proportionally to the key/query dimension d. If these scores are too large, the softmax becomes extremely peaked and gradients vanish. Dividing by √d keeps the scores in a moderate range, ensuring stable gradients and better learning.
**How does self-attention help the model understand relationships between words in a sentence?**
In self‐attention, each token computes attention weights against all tokens’ “keys,” then takes a weighted sum of their “values.” This lets the model dynamically pool information from every other word—capturing long‐range dependencies and the specific relevance of each word to understanding the current one.

## Q5: Sentiment Analysis using HuggingFace Transformers
Implemented sentiment classifier by loading a pretrained sentiment analysis pipeline.

**What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?**
- BERT is built from the Transformer encoder blocks, processing all tokens bidirectionally to learn context from both left and right. It will use Encoder.
- GPT uses the Transformer decoder blocks, attending only to past tokens (unidirectional) for autoregressive text generation. It will use decoder.
**Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.**
- Data Efficiency: They’ve already learned language patterns from massive corpora, so you need far less task-specific data to fine-tune.
- Compute Savings: Training from scratch (billions of parameters) is prohibitively expensive; pre-trained checkpoints let you leverage that investment.
- Performance Boost: Transfer learning yields stronger baselines—better accuracy and faster convergence on downstream tasks (classification, QA, summarization, etc.).

