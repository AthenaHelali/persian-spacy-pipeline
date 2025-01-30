import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
from tqdm import tqdm
import argparse


def evaluate_ner_model(model_path, validation_path):
    """
    Args:
        model_path (str): Path to the trained SpaCy model.
        validation_path (str): Path to the validation dataset in `.spacy` format.
    """
    nlp = spacy.load(model_path)
    doc_bin = DocBin().from_disk(validation_path)
    validation_docs = list(doc_bin.get_docs(nlp.vocab))

    scorer = Scorer()
    examples = []
    total_tokens = 0
    correct_tokens = 0

    print("Evaluating the model...")
    for doc in tqdm(validation_docs):
        pred_doc = nlp(doc.text)
        example = Example(doc, pred_doc)
        examples.append(example)

        # Count token-level accuracy (exact match for entities)
        for token_true, token_pred in zip(doc, pred_doc):
            total_tokens += 1
            if token_true.ent_type_ == token_pred.ent_type_:
                correct_tokens += 1

    results = scorer.score(examples)


    print("\nEvaluation Results:")
    print(f'NER Precision: {results.get("ents_p", 0.0):.3f}')
    print(f'NER Recall   : {results.get("ents_r", 0.0):.3f}')
    print(f'NER F1-Score : {results.get("ents_f", 0.0):.3f}')

    if "ents_per_type" in results:
        print("\nDetailed Scores Per Entity Type:")
        for entity, scores in results["ents_per_type"].items():
            print(
                f'  {entity}: Precision: {scores.get("p", 0.0):.3f}, Recall: {scores.get("r", 0.0):.3f}, F1-Score: {scores.get("f", 0.0):.3f}'
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a NER model using SpaCy's evaluation function with accuracy calculation."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "--validation_path",
        required=True,
        help="Path to the validation dataset (.spacy format).",
    )
    args = parser.parse_args()

    evaluate_ner_model(args.model_path, args.validation_path)
