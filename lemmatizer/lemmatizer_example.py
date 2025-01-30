import spacy
from lemmatizer import load_lemma_dictionary, lemmatize

# Load the lemma dictionary
file_path = "persian_spacy/data/lemmatizer/lemma_dict.txt"
lemma_dict = load_lemma_dictionary(file_path)

# Test sentence
text = "او دیروز به مدرسه رفت و با دوستانش صحبت کرد. بعد از آن، او به خانه آمد و شام خورد."

# Lemmatize with SpaCy
nlp = spacy.blank("fa")


@spacy.Language.component("persian_lemmatizer")
def persian_lemmatizer(doc):
    for token in doc:
        token.lemma_ = lemmatize(token.text, lemma_dict)
    return doc


# Add lemmatizer to pipeline
nlp.add_pipe("persian_lemmatizer", last=True)

doc = nlp(text)
for token in doc:
    print(f"Original: {token.text}, Lemma: {token.lemma_}")
