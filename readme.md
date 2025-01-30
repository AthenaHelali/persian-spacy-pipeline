# Persian-Spacy Project

This project provides tools for **Named Entity Recognition (NER)**, **Part-of-Speech (POS) Tagging**, and **Lemmatization** for the Persian language using **SpaCy**. It includes preprocessing datasets, training models, and evaluating their performance. The pipeline is packaged for easy installation and usage.

Due to the large size of the package, it is available for download at the following Google Drive link:
[Download Persian-Spacy Package](https://drive.google.com/file/d/1rOv33doSOIgoZopaXWOERUphEaq9CR_-/view?usp=drive_link)

---

## Installation

To install the Persian-Spacy pipeline, navigate to the project directory and run:

```bash
pip install .
```

---

## Directory Structure

```plaintext
persian_spacy/
│
├── data/                      # Contains datasets
│
├── fa_pipeline/               # Main package directory
│
├── fasttext/                  # FastText embeddings for Persian
│
├── lemmatizer/                # Rule-based lemmatizer implementation
│
├── models/                    # Trained models directory
│
├── ner/                       # Named Entity Recognition scripts
│
├── pos/                       # POS tagging scripts
│
├── spacy-env/                 # SpaCy virtual environment
│
├── setup.py                   # Installation script
│
├── requirements.txt            # Required dependencies
```

---

## Features

- **POS Tagging**: Assigns **grammatical categories** (e.g., noun, verb, adjective) to Persian words.
- **NER (Named Entity Recognition)**: Detects **names, locations, and organizations** in Persian text.
- **Lemmatization**: Extracts the **root form of words** using a **rule-based approach**.
- **FastText Embeddings**: Supports **word vector representations** for semantic processing.
- **Fully integrated with SpaCy**: Designed to work within the **SpaCy NLP pipeline**.

---

## Example Usage

The following is a simple example demonstrating the use of the **NER model**:

```python
import spacy

# Load the trained NER model
nlp = spacy.load("fa_pipeline")

# Example sentence
text = "علی به مدرسه رفت."
doc = nlp(text)

# Print named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## Datasets

The models were trained on **high-quality Persian datasets**, ensuring **optimal accuracy**. The following datasets were used:

- **POS Tagging**: Trained on the **Seraji dataset (Universal Dependencies)**.
- **NER**: Trained on the **PersianNER dataset**.
- **Lemmatization**: Built using a **custom rule-based approach** with Persian linguistic rules.

### Model Performance

| Task        | Accuracy |
| ----------- | -------- |
| POS Tagging | **97%**  |
| NER         | **72%**  |

---

## POS Tags

The project uses **Universal Dependencies (UD) POS tags**, ensuring consistency and compatibility with other NLP tools.

| Tag       | Description               |
| --------- | ------------------------- |
| **ADJ**   | Adjective                 | 
| **ADP**   | Adposition                |
| **ADV**   | Adverb                    | 
| **AUX**   | Auxiliary verb            |
| **CCONJ** | Coordinating conjunction  | 
| **DET**   | Determiner                |
| **INTJ**  | Interjection              | 
| **NOUN**  | Noun                      | 
| **NUM**   | Numeral                   | 
| **PART**  | Particle                  |
| **PRON**  | Pronoun                   |
| **PROPN** | Proper noun               |
| **PUNCT** | Punctuation               |
| **SCONJ** | Subordinating conjunction |
| **SYM**   | Symbol                    |
| **VERB**  | Verb                      | 
| **X**     | Other                     |

---

## NER Tags

The following entity tags are used in the NER model:

| Tag         | Description                      |
| ----------- | -------------------------------- | 
| **B-PER**   | Beginning of Person Name         | 
| **I-PER**   | Inside Person Name               | 
| **B-ORG**   | Beginning of Organization        | 
| **I-ORG**   | Inside Organization              | 
| **B-LOC**   | Beginning of Location            | 
| **I-LOC**   | Inside Location                  | 
| **B-GPE**   | Beginning of Geopolitical Entity | 
| **I-GPE**   | Inside Geopolitical Entity       | 
| **B-FAC**   | Beginning of Facility            | 
| **I-FAC**   | Inside Facility                  | 
| **B-TITLE** | Beginning of Title               | 
| **I-TITLE** | Inside Title                     |

---

## Future Work

- Improve **NER accuracy** by using **higher-quality datasets**.
- Expand the **lemmatization rules** to **better support Persian morphology**.
- Add **dependency parsing** and **text classification** components.

