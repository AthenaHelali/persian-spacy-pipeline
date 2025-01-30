"""
Script for preprocessing datasets and converting CoNLL-U files to SpaCy format.
"""

import spacy
from spacy.tokens import Doc, DocBin
import pyconll
import argparse


def conllu_to_spacy_aligned(input_file, output_file):
    """
    Convert a CoNLL-U formatted file to SpaCy training format using pyconll.

    Args:
        input_file (str): Path to the input CoNLL-U file.
        output_file (str): Path to save the converted SpaCy binary file.
    """
    nlp = spacy.blank("fa")  # Create a blank SpaCy model for Persian
    doc_bin = DocBin()

    # Load the CoNLL-U data using pyconll
    try:
        conllu_data = pyconll.load.iter_from_file(input_file)
    except Exception as e:
        raise RuntimeError(f"Error reading CoNLL-U file: {e}")

    for sentence in conllu_data:
        words = []
        pos_tags = []

        # Iterate through tokens and extract words and POS tags
        for token in sentence:
            if token.form is None or token.upos is None:
                continue

            words.append(token.form)
            pos_tags.append(token.upos)

        if not words or not pos_tags:
            continue

        # Create a SpaCy Doc object
        doc = Doc(nlp.vocab, words=words)

        # Assign POS tags to tokens
        for i, token in enumerate(doc):
            token.tag_ = pos_tags[i]
            token.pos_ = pos_tags[i]

        # Add the processed Doc to the DocBin
        doc_bin.add(doc)

    # Save the processed data to disk
    doc_bin.to_disk(output_file)
    print(f"Successfully saved SpaCy binary file to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CoNLL-U files to SpaCy format."
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to the input CoNLL-U file."
    )
    parser.add_argument(
        "--output_file", required=True, help="Path to the output SpaCy file."
    )
    args = parser.parse_args()

    conllu_to_spacy_aligned(args.input_file, args.output_file)
