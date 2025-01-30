import random
from spacy.training import Example
from spacy.tokens import DocBin
from pathlib import Path
import spacy
import argparse
from gensim.models import KeyedVectors


# Hard-coded tag mapping
TAG_MAPPING = {
    0: "O",
    1: "B-ORG",
    2: "I-ORG",
    3: "B-PER",
    4: "I-PER",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-FAC",
    8: "I-FAC",
    9: "B-GPE",
    10: "I-GPE",
    11: "B-TITLE",
    12: "I-TITLE",
}


def train_ner_model(train_path, output_dir, iterations):
    """
    Train a Named Entity Recognition (NER) model using SpaCy.

    Args:
        train_path (str): Path to the training dataset in .spacy format.
        output_dir (str): Directory to save the trained model.
        iterations (int): Number of training iterations.
    """
    # Initialize a blank SpaCy model
    nlp = spacy.blank("fa")

    embedding_path = "persian_spacy/fasttext/cc.fa.300.vec"

    fasttext_model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    print("FastText embeddings loaded successfully!")


    # Add FastText vectors to spaCy's vocabulary
    for word in fasttext_model.index_to_key:
        vector = fasttext_model[word]
        nlp.vocab.set_vector(word, vector)

    ner = nlp.add_pipe("ner", last=True)

    for label in TAG_MAPPING.values():
        if label != "O":
            ner.add_label(label)

    train_doc_bin = DocBin().from_disk(train_path)

    # Train the model
    optimizer = nlp.begin_training()
    for iteration in range(iterations):
        print(f"Starting iteration {iteration + 1}")
        losses = {}
        # Load the training data
        train_examples = list(train_doc_bin.get_docs(nlp.vocab))
        random.shuffle(train_examples)
        # Update the model
        for doc in train_examples:
            example = Example.from_dict(
                doc,
                {
                    "entities": [
                        (ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
                    ]
                },
            )
            nlp.update([example], drop=0.3, losses=losses)
        print(f"Losses at iteration {iteration + 1}: {losses}")

    # Save the trained model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model using SpaCy.")
    parser.add_argument(
        "--train_path",
        required=True,
        help="Path to the training dataset in .spacy format.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the trained model."
    )
    parser.add_argument(
        "--iterations", type=int, default=25, help="Number of training iterations."
    )
    args = parser.parse_args()

    train_ner_model(args.train_path, args.output_dir, args.iterations)
