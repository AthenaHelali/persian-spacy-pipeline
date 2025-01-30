"""
Script for evaluating a trained POS tagging model using SpaCy's built-in evaluation function.
"""

import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training import Example
from tqdm import tqdm
import argparse


def evaluate_model(model_path, validation_path):
    """
    Evaluate the trained POS tagging model using SpaCy's built-in evaluation function.

    Args:
        model_path (str): Path to the trained SpaCy model.
        validation_path (str): Path to the SpaCy validation data file.
    """
    nlp = spacy.load(model_path)  # Load the trained SpaCy model
    validation_doc_bin = DocBin().from_disk(validation_path)  # Load validation data
    validation_docs = list(validation_doc_bin.get_docs(nlp.vocab))  # Extract Docs

    scorer = Scorer()
    examples = []

    print("Evaluating the model on validation data...")
    for doc in tqdm(validation_docs, total=len(validation_docs)):
        pred_doc = nlp(doc.text)  # Predict using the trained model
        examples.append(Example(doc, pred_doc))

    # Get overall evaluation results
    results = scorer.score(examples)
    print("\nEvaluation Results:")

    if "tag_acc" in results:
        print(f'Accuracy : {results["tag_acc"]:.5f}')
    else:
        print("Warning: 'tag_acc' not found in results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained POS tagging model using SpaCy's evaluation function."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "--validation_path", required=True, help="Path to the validation SpaCy file."
    )
    args = parser.parse_args()

    evaluate_model(args.model_path, args.validation_path)
