import unittest
import torch
import numpy as np
from transformers import AutoTokenizer

from src.dataset_helpers import extract_entities_from_bio, format_instruction_sample, tokenize_and_align_labels, GenerativeDataset
from src.evaluation_helpers import compute_bert_metrics, compute_generative_metrics, parse_llm_entities

class TestDatabricksPipeline(unittest.TestCase):
    
    def setUp(self):
        # A tiny tokenizer for unit testing (fast download, fully functional)
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.labels_list = ["O", "B-Disease", "I-Disease"]
        
    def test_extract_entities_from_bio(self):
        tokens = ["The", "patient", "was", "diagnosed", "with", "breast", "cancer", "and", "hypertension"]
        ner_tags = [0, 0, 0, 0, 0, 1, 2, 0, 1]  # O, O, O, O, O, B-Dis, I-Dis, O, B-Dis
        
        entities = extract_entities_from_bio(tokens, ner_tags, self.labels_list)
        
        # Check extraction and uniqueness
        self.assertEqual(len(entities), 2)
        self.assertIn("breast cancer", entities)
        self.assertIn("hypertension", entities)
        
    def test_format_instruction_sample(self):
        tokens = ["breast", "cancer", "patient"]
        ner_tags = [1, 2, 0]
        
        formatted = format_instruction_sample(tokens, ner_tags, self.labels_list)
        
        self.assertIn("system", formatted)
        self.assertIn("user", formatted)
        self.assertIn("response", formatted)
        self.assertEqual(formatted["response"], "breast cancer")
        
    def test_parse_llm_entities(self):
        text = "breast cancer, hypertension, None, "
        entities = parse_llm_entities(text)
        
        self.assertEqual(len(entities), 2)
        self.assertIn("breast cancer", entities)
        self.assertIn("hypertension", entities)
        self.assertNotIn("none", entities)
        
    def test_compute_generative_metrics(self):
        predictions = ["breast cancer, diabetes", "hypertension", "None"]
        references = ["breast cancer", "hypertension, diabetes", "stroke"]
        
        metrics = compute_generative_metrics(predictions, references)
        
        # Sample 1: Pred={breast cancer, diabetes}, Ref={breast cancer} -> TP=1, FP=1, FN=0
        # Sample 2: Pred={hypertension}, Ref={hypertension, diabetes} -> TP=1, FP=0, FN=1
        # Sample 3: Pred={}, Ref={stroke} -> TP=0, FP=0, FN=1
        # Totals: TP=2, FP=1, FN=2
        # Precision = 2 / 3 = 0.6666
        # Recall = 2 / 4 = 0.50
        # F1 = 2 * 0.6666 * 0.5 / (1.1666) = 0.5714
        
        self.assertAlmostEqual(metrics["precision"], 2/3)
        self.assertAlmostEqual(metrics["recall"], 2/4)
        self.assertEqual(metrics["total_true_positives"], 2)
        self.assertEqual(metrics["total_false_positives"], 1)
        self.assertEqual(metrics["total_false_negatives"], 2)

    def test_compute_bert_metrics(self):
        # Shape: (batch_size, seq_len)
        predictions = np.array([
            [0, 1, 2, 0],
            [0, 0, 1, 0]
        ])
        labels = np.array([
            [0, 1, 2, -100],  # the last one is ignored
            [-100, 0, 1, 0]
        ])
        
        metrics, report = compute_bert_metrics(predictions, labels, self.labels_list)
        
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(report, str)
        
    def test_tokenize_and_align_labels(self):
        examples = {
            "tokens": [["breast", "cancer"], ["hypertension"]],
            "ner_tags": [[1, 2], [1]]
        }
        
        aligned = tokenize_and_align_labels(examples, self.tokenizer, max_length=10)
        
        self.assertIn("input_ids", aligned)
        self.assertIn("attention_mask", aligned)
        self.assertIn("labels", aligned)
        
        # First word should be labeled, subwords -100, special tokens -100
        self.assertEqual(len(aligned["labels"]), 2)
        # Verify the list format
        self.assertEqual(len(aligned["labels"][0]), 10)
        self.assertEqual(aligned["labels"][0][0], -100) # [CLS]
        
    def test_generative_dataset(self):
        mock_hf_dataset = [
            {"tokens": ["breast", "cancer"], "ner_tags": [1, 2]},
            {"tokens": ["hypertension"], "ner_tags": [1]}
        ]
        
        dataset = GenerativeDataset(mock_hf_dataset, self.tokenizer, self.labels_list, max_length=20)
        
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("labels", sample)
        
        self.assertEqual(sample["input_ids"].shape, (20,))
        self.assertEqual(sample["labels"].shape, (20,))
        
        # Verify prompt tokens are masked with -100
        # The prompt prefix starts with <|im_start|>system... and ends with <|im_start|>assistant
        # So the label value at index 0 should be -100
        self.assertEqual(sample["labels"][0].item(), -100)

if __name__ == "__main__":
    unittest.main()
