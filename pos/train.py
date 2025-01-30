import spacy
import random
from pathlib import Path
from spacy.training import Example
from spacy.tokens import DocBin
from gensim.models import KeyedVectors
from collections import Counter
import argparse


def balance_data(doc_bin, nlp, rare_tags, common_tag_threshold=15000):
    """
    Balance training data by oversampling rare tags and downsampling common tags.

    Args:
        doc_bin (DocBin): SpaCy DocBin containing training data.
        nlp (Language): SpaCy language object.
        rare_tags (set): Tags considered rare.
        common_tag_threshold (int): Maximum number of examples for common tags.

    Returns:
        DocBin: Balanced training data.
    """
    docs = list(doc_bin.get_docs(nlp.vocab))
    tag_counter = Counter()

    # Count tag occurrences before balancing
    for doc in docs:
        tag_counter.update([token.tag_ for token in doc])

    # Balance the dataset
    balanced_docs = []
    for doc in docs:
        doc_tags = [token.tag_ for token in doc]

        # Oversample rare tags
        if any(tag in rare_tags for tag in doc_tags):
            balanced_docs.extend([doc] * 5)  # Oversample rare tags

        # Downsample common tags
        elif any(tag_counter[tag] > common_tag_threshold for tag in doc_tags):
            if random.random() < 0.5:  # Randomly include half of the examples
                balanced_docs.append(doc)
        else:
            balanced_docs.append(doc)

    # Create a new DocBin with balanced docs
    balanced_doc_bin = DocBin()
    for doc in balanced_docs:
        balanced_doc_bin.add(doc)

    return balanced_doc_bin


def train_model(train_path, output_dir, rare_tags, common_tag_threshold=15000):
    """
    Train a POS tagging model without validation evaluation.

    Args:
        train_path (str): Path to the SpaCy training data file.
        output_dir (str): Directory to save the trained model.
        rare_tags (set): Tags considered rare for oversampling.
        common_tag_threshold (int): Threshold for downsampling common tags.
    """
    nlp = spacy.blank("fa")  # Load blank Persian SpaCy model

    embedding_path = "persian_spacy/fasttext/cc.fa.300.vec"

    # Load FastText embeddings
    fasttext_model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    print("FastText embeddings loaded successfully!")

    # Add FastText vectors to SpaCy's vocabulary
    for word in fasttext_model.index_to_key:
        vector = fasttext_model[word]
        nlp.vocab.set_vector(word, vector)

    tagger = nlp.add_pipe("tagger", last=True)

    # Load training data
    train_doc_bin = DocBin().from_disk(train_path)

    # Balance training data
    train_doc_bin = balance_data(train_doc_bin, nlp, rare_tags, common_tag_threshold)

    all_tags = set()
    for doc in train_doc_bin.get_docs(nlp.vocab):
        all_tags.update(token.tag_ for token in doc)

    for tag in all_tags:
        tagger.add_label(tag)

    optimizer = nlp.begin_training()

    for iteration in range(24):  # Train for 24 iterations
        print(f"Starting iteration {iteration + 1}")
        losses = {}
        train_examples = list(train_doc_bin.get_docs(nlp.vocab))
        random.shuffle(train_examples)

        for doc in train_examples:
            example = Example.from_dict(
                doc,
                {
                    "tags": [token.tag_ for token in doc],
                },
            )
            nlp.update([example], drop=0.3, losses=losses)

        # Print the training loss for this iteration
        print(f"Iteration {iteration + 1} - Training Loss: {losses['tagger']}")

    # Save the trained model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a POS tagging model.")
    parser.add_argument(
        "--train_path", required=True, help="Path to the training data."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the trained model."
    )
    args = parser.parse_args()

    # Rare tags to oversample
    rare_tags = {"INTJ", "X"}

    train_model(args.train_path, args.output_dir, rare_tags, common_tag_threshold=15000)
