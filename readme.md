# Persian-Spacy Project

This project provides tools for **Named Entity Recognition (NER)**, **Part-of-Speech (POS) Tagging**, and **Lemmatization** for the Persian language using **SpaCy**. It includes preprocessing datasets, training models, and evaluating their performance. The pipeline is packaged for easy installation and usage.

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
│
└── test2.py                    # Example test script
```

---

## Features

- **POS Tagging**: Assigns **grammatical categories** (e.g., noun, verb, adjective) to Persian words.
- **NER (Named Entity Recognition)**: Detects **names, locations, and organizations** in Persian text.
- **Lemmatization**: Extracts the **root form of words** using a **rule-based approach**.
- **FastText Embeddings**: Supports **word vector representations** for semantic processing.
- **Fully integrated with SpaCy**: Designed to work within the **SpaCy NLP pipeline**.

---

## Datasets

The models were trained on **high-quality Persian datasets**, ensuring **optimal accuracy**. The following datasets were used:

- **POS Tagging**: Trained on the **Seraji dataset (Universal Dependencies)**.
- **NER**: Uses a combination of **Peyma and Arman datasets**.
- **Lemmatization**: Built using a **custom rule-based approach** with Persian linguistic rules.

### Model Performance

| Task        | Accuracy |
| ----------- | -------- |
| POS Tagging | **97%**  |
| NER         | **72%**  |

The **POS model performs exceptionally well**, while the **NER model accuracy can be improved with better datasets**.

---

## POS Tags

The project uses **Universal Dependencies (UD) POS tags**, ensuring consistency and compatibility with other NLP tools.

| Tag       | Description               | Example                      |
| --------- | ------------------------- | ---------------------------- |
| **ADJ**   | Adjective                 | بزرگ (big)                   |
| **ADP**   | Adposition                | در (in), به (to)             |
| **ADV**   | Adverb                    | بسیار (very), سریع (quickly) |
| **AUX**   | Auxiliary verb            | است (is), بود (was)          |
| **CCONJ** | Coordinating conjunction  | و (and), یا (or)             |
| **DET**   | Determiner                | این (this), آن (that)        |
| **INTJ**  | Interjection              | آه (oh), وای (wow)           |
| **NOUN**  | Noun                      | کتاب (book), درخت (tree)     |
| **NUM**   | Numeral                   | سه (three), ده (ten)         |
| **PART**  | Particle                  | هم (also), فقط (only)        |
| **PRON**  | Pronoun                   | او (he/she), ما (we)         |
| **PROPN** | Proper noun               | ایران (Iran), علی (Ali)      |
| **PUNCT** | Punctuation               | . , ; ! ؟ (.,;!?)            |
| **SCONJ** | Subordinating conjunction | که (that), زیرا (because)    |
| **SYM**   | Symbol                    | % , + , -                    |
| **VERB**  | Verb                      | رفت (went), می‌خواند (reads)  |
| **X**     | Other                     | نامشخص (unknown words)       |

---

## NER Tags

The following entity tags are used in the NER model:

| Tag         | Description                      | Example                              |
| ----------- | -------------------------------- | ------------------------------------ |
| **B-PER**   | Beginning of Person Name         | علی (Ali)                            |
| **I-PER**   | Inside Person Name               | محمدی (Mohammadi)                    |
| **B-ORG**   | Beginning of Organization        | گوگل (Google)                        |
| **I-ORG**   | Inside Organization              | دانشگاه تهران (University of Tehran) |
| **B-LOC**   | Beginning of Location            | تهران (Tehran)                       |
| **I-LOC**   | Inside Location                  | شمال تهران (North Tehran)            |
| **B-GPE**   | Beginning of Geopolitical Entity | ایران (Iran)                         |
| **I-GPE**   | Inside Geopolitical Entity       | استان فارس (Fars Province)           |
| **B-FAC**   | Beginning of Facility            | برج میلاد (Milad Tower)              |
| **I-FAC**   | Inside Facility                  | ورزشگاه آزادی (Azadi Stadium)        |
| **B-TITLE** | Beginning of Title               | دکتر (Doctor)                        |
| **I-TITLE** | Inside Title                     | مهندس (Engineer)                     |

---

## Future Work

- Improve **NER accuracy** by using **higher-quality datasets**.
- Expand the **lemmatization rules** to **better support Persian morphology**.
- Add **dependency parsing** and **text classification** components.

