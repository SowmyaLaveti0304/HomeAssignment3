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
### Output:

**Explain the role of temperature scaling in text generation and its effect on randomness.**
- Lower T<1 makes the distribution “peaky,” so the highest‐scoring characters dominate and output is more deterministic.
- Higher T>1 flattens the distribution, giving lower‐scoring characters a bigger chance and producing more random, creative (but potentially incoherent) text.

## Q2: NLP Preprocessing Pipeline
### Tasks completed
- Tokenized input sentence
- Removed common English stopwords using NLTK
- Applied stemming using Porter Stemmer
### Output
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
