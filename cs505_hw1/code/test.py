import spacy

nlp = spacy.load('en')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print(type(ent.text), type(ent.label_))