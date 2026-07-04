# Databricks notebook source
# MAGIC %md
# MAGIC # Clinical Text Entity Extraction - BERT Sequence Labeling
# MAGIC This notebook fine-tunes a clinical BERT model (e.g. `emilyalsentzer/Bio_ClinicalBERT`) for Named Entity Recognition (NER) on the `ncbi/ncbi_disease` dataset.
# MAGIC It tracks training progress, loss curves, and evaluation metrics using MLflow, and registers the final model.

# COMMAND ----------
# MAGIC %pip install mlflow seqeval transformers datasets accelerate pyyaml

# COMMAND ----------
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import mlflow
import mlflow.pytorch

# Set sys.path to be able to import helpers from src if notebooks is run in dbfs
# In Databricks, when importing files from repositories, they are automatically placed on path
sys.path.append(os.path.abspath(".."))
from src.dataset_helpers import tokenize_and_align_labels
from src.evaluation_helpers import compute_bert_metrics

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Create Databricks Widgets for Parameters

# COMMAND ----------
dbutils.widgets.text("model_name", "emilyalsentzer/Bio_ClinicalBERT", "Hugging Face Model")
dbutils.widgets.text("learning_rate", "2e-5", "Learning Rate")
dbutils.widgets.text("batch_size", "16", "Batch Size")
dbutils.widgets.text("epochs", "3", "Epochs")
dbutils.widgets.text("mlflow_experiment", "/Shared/clinical_bert_ner", "MLflow Experiment Name")
dbutils.widgets.dropdown("use_unity_catalog", "false", ["true", "false"], "Use Unity Catalog Model Registry")
dbutils.widgets.text("unity_catalog_name", "main", "Unity Catalog Name")
dbutils.widgets.text("unity_catalog_schema", "default", "Unity Catalog Schema")

# COMMAND ----------
# Load variables from widgets
model_name = dbutils.widgets.get("model_name")
learning_rate = float(dbutils.widgets.get("learning_rate"))
batch_size = int(dbutils.widgets.get("batch_size"))
epochs = int(dbutils.widgets.get("epochs"))
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
use_unity_catalog = dbutils.widgets.get("use_unity_catalog") == "true"
unity_catalog_name = dbutils.widgets.get("unity_catalog_name")
unity_catalog_schema = dbutils.widgets.get("unity_catalog_schema")

print(f"Model Name: {model_name}")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"MLflow Experiment: {mlflow_experiment}")
print(f"Use Unity Catalog: {use_unity_catalog}")
if use_unity_catalog:
    print(f"Unity Catalog Path: {unity_catalog_name}.{unity_catalog_schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Load Dataset and Preprocess

# COMMAND ----------
# Load NCBI Disease Dataset from Hugging Face Parquet conversion branch (script-less)
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset
import os

print("Downloading Parquet dataset files from Hugging Face...")
api = HfApi()
repo_files = api.list_repo_files(repo_id="ncbi/ncbi_disease", repo_type="dataset", revision="refs/convert/parquet")
parquet_files = [f for f in repo_files if f.endswith(".parquet")]

local_paths = {}
for f in parquet_files:
    local_path = hf_hub_download(
        repo_id="ncbi/ncbi_disease",
        filename=f,
        repo_type="dataset",
        revision="refs/convert/parquet"
    )
    split = f.split("/")[1]
    if split not in local_paths:
        local_paths[split] = []
    local_paths[split].append(local_path)

raw_datasets = load_dataset("parquet", data_files=local_paths)

# Auto-scale dataset for fast CPU runs if GPU is not available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("CPU detected. Downscaling dataset size and epochs for a fast run...")
    raw_datasets["train"] = raw_datasets["train"].select(range(min(800, len(raw_datasets["train"]))))
    raw_datasets["validation"] = raw_datasets["validation"].select(range(min(200, len(raw_datasets["validation"]))))
    epochs = 1  # Force 1 epoch
    print_interval = 5
else:
    print_interval = 20

labels_list = ["O", "B-Disease", "I-Disease"]
num_labels = len(labels_list)
print("Labels list:", labels_list)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize datasets
tokenized_datasets = raw_datasets.map(
    lambda x: tokenize_and_align_labels(x, tokenizer),
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Prepare PyTorch DataLoaders

# COMMAND ----------
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 4: Initialize Model & Optimizer

# COMMAND ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 5: Start Custom PyTorch Training Loop tracked with MLflow

# COMMAND ----------
mlflow.set_experiment(mlflow_experiment)

with mlflow.start_run(run_name="bert_ner_run") as run:
    # Log Hyperparameters
    mlflow.log_params({
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "device": str(device)
    })
    
    # Custom training loop
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1} ---")
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            train_loss += loss.item()
            
            if step % print_interval == 0:
                print(f"  Step {step}/{len(train_dataloader)} | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_dataloader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        print(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to("cpu").numpy()
                
                val_preds.append(logits)
                val_labels.append(label_ids)
                
        avg_val_loss = val_loss / len(val_dataloader)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        
        # Compute val metrics
        all_preds = np.concatenate(val_preds, axis=0)
        all_labels = np.concatenate(val_labels, axis=0)
        val_metrics, _ = compute_bert_metrics(all_preds, all_labels, labels_list)
        
        mlflow.log_metrics({
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"]
        }, step=epoch)
        
        print(f"Validation Loss: {avg_val_loss:.4f} | F1: {val_metrics['f1']:.4f}")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ### Step 6: Test Evaluation and Model Registration

    # COMMAND ----------
    print("\n--- Running Final Evaluation on Test Set ---")
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()
            
            test_preds.append(logits)
            test_labels.append(label_ids)
            
    all_test_preds = np.concatenate(test_preds, axis=0)
    all_test_labels = np.concatenate(test_labels, axis=0)
    
    test_metrics, classification_report = compute_bert_metrics(all_test_preds, all_test_labels, labels_list)
    print("\nClassification Report:\n", classification_report)
    
    # Log test metrics
    mlflow.log_metrics({
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"]
    })
    
    # Save the classification report as a text artifact
    os.makedirs("/tmp/artifacts", exist_ok=True)
    report_path = "/tmp/artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report)
    mlflow.log_artifact(report_path)
    
    # Configure Unity Catalog Registry if enabled
    if use_unity_catalog:
        mlflow.set_registry_uri("databricks-uc")
        registered_name = f"{unity_catalog_name}.{unity_catalog_schema}.clinical_bert_ner_model"
        print(f"Registering model to Unity Catalog: {registered_name}")
    else:
        registered_name = "clinical_bert_ner_model"
        print(f"Registering model to Workspace Registry: {registered_name}")

    # Infer model signature for Unity Catalog compatibility
    from mlflow.models import infer_signature
    dummy_input = {
        "input_ids": np.zeros((1, 128), dtype=np.int32),
        "attention_mask": np.zeros((1, 128), dtype=np.int32)
    }
    with torch.no_grad():
        dummy_in_ids = torch.tensor(dummy_input["input_ids"]).to(device)
        dummy_att_mask = torch.tensor(dummy_input["attention_mask"]).to(device)
        dummy_out = model(input_ids=dummy_in_ids, attention_mask=dummy_att_mask).logits.cpu().numpy()
    signature = infer_signature(dummy_input, dummy_out)

    # Log and register the PyTorch model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=registered_name,
        signature=signature
    )
    print(f"Training finished. Model successfully registered as: {registered_name}")
