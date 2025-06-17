# nlp_preprocessing.py

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def nlp_preprocess(text: str):
    # 1. Tokenize using wordpunct (splits on punctuation)
    tokens = wordpunct_tokenize(text)
    print("Original Tokens:", tokens)

    # 2. Remove stopwords (we also filter out any non-alphabetic tokens here)
    stop_words = set(stopwords.words("english"))
    tokens_no_stop = [
        tok for tok in tokens
        if tok.isalpha() and tok.lower() not in stop_words
    ]
    print("Tokens Without Stopwords:", tokens_no_stop)

    # 3. Apply Porter stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(tok) for tok in tokens_no_stop]
    print("Stemmed Words:", stems)


if __name__ == "__main__":
    sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
    nlp_preprocess(sentence)
