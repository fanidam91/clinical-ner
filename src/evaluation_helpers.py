import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_bert_metrics(predictions, labels, labels_list):
    """
    Computes sequence labeling metrics (Precision, Recall, F1, Accuracy) for BERT predictions.
    
    predictions: Logits from the model, shape (batch_size, seq_len, num_labels) or predictions (batch_size, seq_len)
    labels: Ground truth token labels, shape (batch_size, seq_len) with -100 for ignored tokens
    labels_list: List of tag names corresponding to indices (e.g. ['O', 'B-Disease', 'I-Disease'])
    """
    if len(predictions.shape) == 3:
        preds = np.argmax(predictions, axis=2)
    else:
        preds = predictions
        
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions)
    }
    
    report = seqeval_classification_report(true_labels, true_predictions)
    return results, report


def parse_llm_entities(text):
    """
    Parses LLM generated comma-separated list of entities.
    Example: "hypertension, type 2 diabetes" -> {"hypertension", "type 2 diabetes"}
    """
    if not text or text.strip().lower() == "none" or text.strip() == "":
        return set()
        
    # Split by comma and strip whitespaces
    entities = [ent.strip().lower() for ent in text.split(",")]
    # Remove empty strings and any 'none' tokens
    entities = {ent for ent in entities if ent and ent != "none"}
    return entities


def compute_generative_metrics(predictions, references):
    """
    Computes set-based precision, recall, and F1-score for generative NER.
    
    predictions: list of strings (generated responses)
    references: list of strings (ground truth responses)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, ref in zip(predictions, references):
        pred_set = parse_llm_entities(pred)
        ref_set = parse_llm_entities(ref)
        
        tp = len(pred_set.intersection(ref_set))
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn
    }
