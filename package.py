import spacy
from pathlib import Path
from spacy.language import Language

# Paths to your trained models and resources
pos_model_path = "/Users/atenahli/Documents/persian-spacy/persian_spacy/models/pos"
ner_model_path = "/Users/atenahli/Documents/persian-spacy/persian_spacy/models/ner"
lemma_dict_path = "lemma_dict.txt"
vectors_path = "/Users/atenahli/Documents/persian-spacy/persian_spacy/fasttext/vocab"


def load_lemma_dictionary(file_path: str) -> dict:
    """
    Load a lemma dictionary from a file.

    Args:
        file_path (str): Path to the lemma dictionary file.

    Returns:
        dict: A dictionary where keys are words and values are their lemmas.
    """
    lemma_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  # Assuming tab-separated values
            if len(parts) == 3:
                word, lemma, pos_tag = parts
                lemma_dict[word] = lemma
    return lemma_dict


def normalize_text(word: str) -> str:
    """
    Normalize Persian text to standard form.

    Args:
        word (str): Input word.

    Returns:
        str: Normalized word.
    """
    word = (
        word.replace("ئ", "ی").replace("ک", "ک").replace("گ", "گ")
    )  # Example normalization
    word = word.replace("ؤ", "و").replace("إ", "ا")  # More normalization rules
    word = word.replace("\u200C", " ")  # Replace Zero Width Non-Joiner
    return word


def remove_suffixes(word: str, dictionary: dict) -> str:
    """
    Try removing each suffix and check if it exists in the dictionary.

    Args:
        word (str): Input word.
        dictionary (dict): Lemma dictionary.

    Returns:
        str: Lemmatized word or the original word.
    """
    suffixes = [
        "ها",
        "ی",
        "تر",
        "ترین",
        "انه",
        "یی",
        "آسا",
        "آگین",
        "او",
        "اومند",
        "اور",
        "ا",
        "گین",
        "اده",
        "ار",
        "اک",
        "ال",
        "اله",
        "ُم",
        "ان",
        "انه",
        "یک",
        "ین",
        "ینه",
        "انی",
        "بان",
        "بد",
        "تر",
        "ترین",
        "چه",
        "دان",
        "دیس",
        "زار",
        "سار",
        "سان",
        "ِستان",
        "وش",
        "سیر",
        "ِش",
        "فام",
        "َک",
        "وند",
        "کده",
        "گار",
        "گاه",
        "گاه",
        "گر",
        "گری",
        "گون",
        "لاخ",
        "مان",
        "مند",
        "نا",
        "ناک",
        "ند",
        "نده",
        "وار",
        "وار",
        "واره",
        "واری",
        "ور",
        "ه",
        "ی",
        "گرا",
        "شده",
        "گوش",
        "مندی",
        "گر",
        "گین",
        "ری",
        "ور",
        "یده",
        "کار",
        "یابی",
        "یافته",
        "ده",
        "ش",
        "ساز",
        "نامه",
        "شده",
        "خوار",
        "بند",
        "ساز",
        "ساز",
        "جوی",
        "شناس",
        "خوار",
        "شناس",
        "ند",
        "آور",
        "طلب",
        "آورده",
        "آوری",
        "جویی",
        "گر",
        "ناکی",
        "گونه",
        "گون",
        "ای",
        "یی",
        "شان",
        "یگر",
        "یانه",
        "ه‌ای",
        "تار",
        "گره",
        "لگن",
        "گان",
        "پذیر",
        "کن",
        "پوی",
        "زن",
        "گون",
        "نی",
        "گانه",
        "شناس",
        "پذیر",
        "پرداز",
        "حس",
        "هایت",
        "هایم",
        "هایش",
        "م",
        "ن",
        "ی",
    ]

    for suffix in suffixes:
        if word.endswith(suffix):
            modified_word = word[: -len(suffix)].strip()
            lemma = dictionary.get(modified_word, None)
            if lemma:
                return lemma
    return word


def remove_prefixes(word: str, dictionary: dict) -> str:
    """
    Try removing each prefix and check if it exists in the dictionary.

    Args:
        word (str): Input word.
        dictionary (dict): Lemma dictionary.

    Returns:
        str: Lemmatized word or the original word.
    """
    prefixes = [
        "با",
        "بی",
        "نا",
        "دی",
        "به",
        "اندر",
        "ب",
        "باز",
        "بر",
        "بس",
        "بیش",
        "پاد",
        "پت",
        "پرا",
        "پس",
        "پسا",
        "پی",
        "پیرا",
        "پیش",
        "ترا",
        "تک",
        "در",
        "دژ",
        "دش",
        "می",
        "سر",
        "فر",
        "فرا",
        "فرو",
        "نا",
        "ن",
        "وا",
        "ور",
        "هم",
        "هو",
        "ی",
        "آ",
        "پیش",
        "پرا",
        "ده",
        "تا",
        "همه",
        "نیز",
        "نا",
        "ره",
        "به",
        "دگر",
        "در",
        "زیر",
    ]

    for prefix in prefixes:
        if word.startswith(prefix):
            modified_word = word[len(prefix) :].strip()
            lemma = dictionary.get(modified_word, None)
            if lemma:
                return lemma
    return word


def lemmatize(word: str, dictionary: dict) -> str:
    """
    Lemmatize the word based on POS and search in the dictionary.

    Args:
        word (str): Input word.
        dictionary (dict): Lemma dictionary.

    Returns:
        str: Lemmatized word or the original word.
    """
    normalized_word = normalize_text(word)
    lemma = dictionary.get(normalized_word, None)
    if lemma:
        return lemma
    lemma = remove_suffixes(normalized_word, dictionary)
    if lemma != normalized_word:
        return lemma
    lemma = remove_prefixes(normalized_word, dictionary)
    if lemma != normalized_word:
        return lemma
    return word


# Register the factory with spaCy
@Language.factory("rule_based_lemmatizer")
def create_lemmatizer(nlp, name, lemma_dict_path):
    class LemmatizerComponent:

        def __init__(self, dictionary_path):
            # Ensure the path works after packaging
            package_path = Path(
                __file__
            ).parent  # Get the directory of the current file
            full_path = (
                package_path / dictionary_path
            ).resolve()  # Combine and resolve the full path
            self.lemma_dict = load_lemma_dictionary(full_path.as_posix())

        def __call__(self, doc):
            for token in doc:
                token.lemma_ = lemmatize(token.text, self.lemma_dict)
            return doc

    return LemmatizerComponent(lemma_dict_path)


# Create a blank pipeline for Persian
nlp = spacy.blank("fa")

# Load FastText vectors into the pipeline
nlp.vocab.from_disk(vectors_path)

# Add pretrained POS tagger
pos_nlp = spacy.load(pos_model_path)
nlp.add_pipe("tagger", source=pos_nlp)

# Add pretrained NER
ner_nlp = spacy.load(ner_model_path)
nlp.add_pipe("ner", source=ner_nlp)

# Add rule-based lemmatizer
nlp.add_pipe("rule_based_lemmatizer", config={"lemma_dict_path": lemma_dict_path})

# Analyze the pipeline to confirm components
print(nlp.analyze_pipes())

# Save the combined pipeline
output_path = Path("./fa_core_web_sm")
output_path.mkdir(exist_ok=True, parents=True)
nlp.to_disk(output_path)
print(f"Pipeline saved to {output_path}")
