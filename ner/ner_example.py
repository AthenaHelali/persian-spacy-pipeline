import spacy

# Load the trained NER model
model_path = "/persian-spacy/persian_spacy/models/ner"
nlp = spacy.load(model_path)

sentences = ["ایران یک کشور بزرگ است.", "رئیس‌جمهور آمریکا به ایران سفر کرد."]


for sentence in sentences:
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.text, ent.label_)
