import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Process the text
doc = nlp(text)

# Loop over the entities and print the required info
for ent in doc.ents:
    print(f"Text: {ent.text}\tLabel: {ent.label_}\tStart: {ent.start_char}\tEnd: {ent.end_char}")
