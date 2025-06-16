# First, install the library (run in your shell):
# pip install transformers

from transformers import pipeline

def analyze_sentiment(text: str):
    # 1. Load a pre-trained sentiment-analysis pipeline
    classifier = pipeline("sentiment-analysis")
    
    # 2. Analyze the input text
    result = classifier(text)[0]
    
    # 3. Print label and confidence
    label = result["label"]
    score = result["score"]
    print(f"Sentiment: {label}")
    print(f"Confidence Score: {score:.4f}")

if __name__ == "__main__":
    sentence = (
        "Despite the high price, the performance of the new MacBook is outstanding."
    )
    analyze_sentiment(sentence)
