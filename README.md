# Clinical Text Entity Extraction in Databricks

This project implements a clinical Named Entity Recognition (NER) pipeline on the benchmark **NCBI Disease** dataset, specifically optimized for **Databricks**. 

It provides two distinct tracks for extraction:
1. **BERT Sequence Labeling (Deep Learning Classifier)**: Fine-tunes a clinical BERT model (`emilyalsentzer/Bio_ClinicalBERT`) using PyTorch and Hugging Face Transformers.
2. **Generative LLM PEFT/LoRA (GenAI)**: Fine-tunes a Causal Language Model (such as `Qwen/Qwen2.5-1.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with parameter-efficient fine-tuning (LoRA) and QLoRA.

Both tracks are integrated with **MLflow** for tracking learning rates, loss curves, and evaluation metrics, and registering the resulting models to the Databricks Model Registry / Unity Catalog.

---

## Project Structure

```
FraudDetection/
├── config.yaml                     # Shared training configurations
├── requirements.txt                # Package dependencies
├── src/
│   ├── dataset_helpers.py         # Wordpiece alignment & ChatML formatting helpers
│   └── evaluation_helpers.py      # seqeval metrics & set-based generative metrics
├── notebooks/
│   ├── databricks_bert_ner.py      # Databricks Notebook: BERT Token Classification
│   └── databricks_llm_lora_ner.py  # Databricks Notebook: Causal LLM LoRA training
├── tests/
│   └── test_databricks_pipeline.py # Unit tests for data prep & metrics
└── README.md                       # Documentation
```

---

## Getting Started in Databricks

### 1. Provision a Cluster
For the best experience, create a Databricks Cluster with:
*   **Databricks Runtime Version**: `14.3 LTS ML` (Machine Learning) or higher. This version pre-installs PyTorch, Transformers, and MLflow.
*   **Worker Type**: A GPU-enabled node (e.g. Azure `Standard_NC4as_T4_v3` or `Standard_NV36ads_A10_v5` with NVIDIA T4/A10 GPUs) is highly recommended for the Generative LLM LoRA track to enable QLoRA 4-bit quantization. If running on a CPU-only cluster, both notebooks will run, but LLM training will be slower.

### 2. Import the Notebooks
To import the notebooks into your Databricks workspace:
1.  In Databricks, navigate to your workspace folder.
2.  Click **Import** and select **File**.
3.  Upload `notebooks/databricks_bert_ner.py` and `notebooks/databricks_llm_lora_ner.py` (Databricks automatically detects the `# Databricks notebook source` format and converts them into interactive notebooks).

### 3. Hyperparameter Configuration (Widgets)
Both notebooks have top-level **Databricks Widgets** that allow you to configure training parameters interactively:
*   `model_name`: The Hugging Face model identifier (e.g. `emilyalsentzer/Bio_ClinicalBERT` or `Qwen/Qwen2.5-1.5B-Instruct`).
*   `learning_rate`: Adjust training step sizes (e.g., `2e-5` for BERT, `1e-4` for LoRA).
*   `batch_size`: Batch size per training step.
*   `epochs`: Total training cycles.
*   `mlflow_experiment`: Path to the Databricks MLflow experiment where runs are tracked (e.g. `/Shared/clinical_bert_ner`).

---

## Pipeline Highlights

### Track A: BERT Token Classification (`databricks_bert_ner.py`)
*   **How it works**: Preprocesses the NCBI dataset, tokenizing words and mapping the ground truth BIO tags (`O`, `B-Disease`, `I-Disease`) to subwords, assigning `-100` to ignored wordpiece tokens.
*   **Training Loop**: A custom PyTorch loop that evaluates validation loss and logs sequence-labeling precision, recall, and F1-score to MLflow at the end of each epoch.
*   **Model Registration**: Saves and registers the model in the Databricks Model Registry under the name `clinical_bert_ner_model`.

### Track B: Causal LLM Fine-Tuning with LoRA (`databricks_llm_lora_ner.py`)
*   **How it works**: Formulates NER as an instruction-following task. It converts token-label pairs into system, user, and assistant prompts formatted in ChatML style.
*   **Quantization & PEFT**: Loads the model in 4-bit using `BitsAndBytesConfig` (QLoRA) to minimize GPU memory consumption. It applies LoRA adapters (`peft`) targeting the projection layers of the attention blocks.
*   **MLflow autologging**: Automatically reports validation metrics and learning curves to Databricks MLflow using the Hugging Face `Trainer`'s native integration.
*   **Set-based Entity Evaluation**: Evaluates the generative extraction accuracy using set-based Precision, Recall, and F1 metrics by comparing predicted comma-separated lists of diseases against the ground truth.

---

## Local Development and Verification

If you want to run unit tests locally to verify any code modifications:

1.  Create and activate a virtual environment:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```
3.  Run the tests:
    ```powershell
    python -m unittest tests/test_databricks_pipeline.py
    ```
