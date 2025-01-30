"""
Converts datasets into SpaCy-compatible format for NER after downloading and saving locally.
"""

from datasets import load_dataset
from spacy.tokens import DocBin
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import spacy
import argparse


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


def convert_to_spacy_format(dataset_split, nlp, tag_mapping):
    """
    Convert a HuggingFace dataset split to SpaCy DocBin format.

    Args:
        dataset_split: An iterable HuggingFace dataset split (e.g., 'train', 'test').
        nlp: SpaCy NLP pipeline (blank model).
        tag_mapping (dict): Mapping from tag IDs to tag labels.

    Returns:
        DocBin: SpaCy-compatible binary data.
    """
    doc_bin = DocBin()
    for example in dataset_split:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        doc = nlp.make_doc(" ".join(tokens))
        entities = []
        start = 0
        for token, tag in zip(tokens, ner_tags):
            end = start + len(token)
            if tag != 0:  # Skip "O" tags
                label = tag_mapping[tag]
                entities.append((start, end, label))
            start = end + 1  # Account for space
        spans = [
            doc.char_span(start, end, label=label)
            for start, end, label in entities
            if doc.char_span(start, end)
        ]
        doc.ents = spans
        doc_bin.add(doc)
    return doc_bin


def explore_dataset(train_path):
    """
    Explore the training dataset to extract statistics and visualize label distribution.

    Args:
        train_path (str): Path to the training dataset in .spacy format.
    """
    # Load the .spacy dataset
    nlp = spacy.blank("fa")
    doc_bin = DocBin().from_disk(train_path)
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Extract statistics
    total_samples = len(docs)
    label_counts = Counter()

    for doc in docs:
        for ent in doc.ents:
            label_counts[ent.label_] += 1

    print(f"Total samples in dataset: {total_samples}")
    print("Entity counts by label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Plot the label distribution
    labels, counts = zip(*label_counts.items())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Entity Labels")
    plt.ylabel("Count")
    plt.title("Entity Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace NER dataset to SpaCy format."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the HuggingFace dataset to load.",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help="Path to save the training dataset in .spacy format.",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        required=True,
        help="Path to save the test dataset in .spacy format.",
    )
    args = parser.parse_args()

    # Load the HuggingFace dataset
    dataset = load_dataset(args.dataset_name)

    # Convert dataset splits to lists
    train_split = list(dataset["train"]) if "train" in dataset else []
    test_split = list(dataset["test"]) if "test" in dataset else []

    if not train_split or not test_split:
        raise ValueError("Dataset must contain 'train' and 'test' splits.")

    # Initialize SpaCy pipeline
    nlp = spacy.blank("fa")

    # Convert datasets
    train_doc_bin = convert_to_spacy_format(train_split, nlp, TAG_MAPPING)
    test_doc_bin = convert_to_spacy_format(test_split, nlp, TAG_MAPPING)

    # Save datasets
    train_path = Path(args.train_output)
    test_path = Path(args.test_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_doc_bin.to_disk(train_path)
    test_doc_bin.to_disk(test_path)
    print("Datasets converted and saved successfully.")
    explore_dataset(train_path)
