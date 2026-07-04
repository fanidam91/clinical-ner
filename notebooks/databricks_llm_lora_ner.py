# Databricks notebook source
# MAGIC %md
# MAGIC # Clinical Text Entity Extraction - Generative LLM Fine-Tuning with LoRA
# MAGIC This notebook fine-tunes a Causal Small Language Model (SLM) (e.g. `Qwen/Qwen2.5-1.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
# MAGIC for Generative Named Entity Recognition (NER) on the `ncbi/ncbi_disease` dataset.
# MAGIC It uses **LoRA** and **QLoRA** (if GPU is available) and logs all training details directly to MLflow.

# COMMAND ----------
# MAGIC %pip install mlflow seqeval transformers peft datasets accelerate bitsandbytes pyyaml jinja2

# COMMAND ----------
import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import mlflow
import mlflow.pytorch

# Set sys.path to be able to import helpers from src if notebooks is run in dbfs
sys.path.append(os.path.abspath(".."))
from src.dataset_helpers import GenerativeDataset, format_instruction_sample
from src.evaluation_helpers import compute_generative_metrics

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Create Databricks Widgets for Parameters

# COMMAND ----------
dbutils.widgets.text("model_name", "Qwen/Qwen2.5-0.5B-Instruct", "Hugging Face Model")
dbutils.widgets.text("learning_rate", "1e-4", "Learning Rate")
dbutils.widgets.text("batch_size", "4", "Batch Size")
dbutils.widgets.text("epochs", "3", "Epochs")
dbutils.widgets.text("lora_r", "8", "LoRA Rank (r)")
dbutils.widgets.text("lora_alpha", "16", "LoRA Alpha")
dbutils.widgets.text("mlflow_experiment", "/Shared/clinical_llm_lora_ner", "MLflow Experiment Name")
dbutils.widgets.dropdown("use_unity_catalog", "false", ["true", "false"], "Use Unity Catalog Model Registry")
dbutils.widgets.text("unity_catalog_name", "main", "Unity Catalog Name")
dbutils.widgets.text("unity_catalog_schema", "default", "Unity Catalog Schema")

# COMMAND ----------
# Load variables from widgets
model_name = dbutils.widgets.get("model_name")
learning_rate = float(dbutils.widgets.get("learning_rate"))
batch_size = int(dbutils.widgets.get("batch_size"))
epochs = int(dbutils.widgets.get("epochs"))
lora_r = int(dbutils.widgets.get("lora_r"))
lora_alpha = int(dbutils.widgets.get("lora_alpha"))
mlflow_experiment = dbutils.widgets.get("mlflow_experiment")
use_unity_catalog = dbutils.widgets.get("use_unity_catalog") == "true"
unity_catalog_name = dbutils.widgets.get("unity_catalog_name")
unity_catalog_schema = dbutils.widgets.get("unity_catalog_schema")

# CPU Safeguard: Auto-override model and batch sizing to fit in 16GB RAM
import torch
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("CPU environment detected. Applying automatic safeguards to prevent OOM...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    batch_size = 1
    epochs = 1

print(f"Model Name: {model_name}")
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"LoRA Rank (r): {lora_r}")
print(f"LoRA Alpha: {lora_alpha}")
print(f"MLflow Experiment: {mlflow_experiment}")
print(f"Use Unity Catalog: {use_unity_catalog}")
if use_unity_catalog:
    print(f"Unity Catalog Path: {unity_catalog_name}.{unity_catalog_schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Load Dataset and Initialize Tokenizer

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
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("CPU detected. Downscaling dataset size and epochs for a fast run...")
    raw_datasets["train"] = raw_datasets["train"].select(range(min(400, len(raw_datasets["train"]))))
    raw_datasets["validation"] = raw_datasets["validation"].select(range(min(100, len(raw_datasets["validation"]))))
    raw_datasets["test"] = raw_datasets["test"].select(range(min(50, len(raw_datasets["test"]))))
    epochs = 1  # Force 1 epoch

labels_list = ["O", "B-Disease", "I-Disease"]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format data into custom GenerativeDataset
train_dataset = GenerativeDataset(raw_datasets["train"], tokenizer, labels_list)
val_dataset = GenerativeDataset(raw_datasets["validation"], tokenizer, labels_list)

print(f"Train Dataset size: {len(train_dataset)}")
print(f"Validation Dataset size: {len(val_dataset)}")
print("\nFormatted Example Text:\n", train_dataset.samples[0]["text"])

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Load Model with QLoRA Quantization

# COMMAND ----------
# Configure QLoRA if GPU is available. Fallback to CPU-friendly load if not.
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Enable gradient checkpointing to save GPU memory
    model.gradient_checkpointing_enable()
else:
    # CPU fallback
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 4: Configure and Apply LoRA Adapters

# COMMAND ----------
# Identify common module names for LoRA targeting based on model family
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules
)

model = get_peft_model(model, peft_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 5: Start Training via HF Trainer and MLflow

# COMMAND ----------
mlflow.set_experiment(mlflow_experiment)

training_args = TrainingArguments(
    output_dir="/tmp/llm_lora_ner",
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8 if not cuda_available else 2,  # Compensate for batch_size=1
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="mlflow",
    dataloader_drop_last=False,
    optim="adafactor",
    gradient_checkpointing=True if not cuda_available else False,  # Save activation memory!
    # Use CPU/FP32 if CUDA is not available
    fp16=cuda_available,
    bf16=False
)

# Collate pads inputs dynamically to max batch sequence length (saving memory)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Start MLflow run to log hyperparameters and the model card
with mlflow.start_run(run_name="llm_lora_run") as run:
    # Run HuggingFace Trainer
    trainer.train()
    
    # MAGIC %md
    # MAGIC ### Step 6: Generative Evaluation on the Test Set

    # COMMAND ----------
    print("\n--- Running Generative Evaluation on Test Subset ---")
    model.eval()
    
    # Evaluate on a subset of the test set to stay within reasonable time limits on CPU
    # If on a high-end GPU cluster, feel free to run on the entire test dataset (raw_datasets["test"])
    test_subset = raw_datasets["test"].select(range(min(100, len(raw_datasets["test"]))))
    
    predictions = []
    references = []
    
    for idx, sample in enumerate(test_subset):
        formatted = format_instruction_sample(sample["tokens"], sample["ner_tags"], labels_list)
        
        prompt_text = (
            f"<|im_start|>system\n{formatted['system']}<|im_end|>\n"
            f"<|im_start|>user\n{formatted['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>")
            )
            
        # Decode only the generated text (ignoring prompt)
        generated_tokens = output_tokens[0][input_ids.shape[1]:]
        decoded_out = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # Clean up any residual markers
        decoded_out = decoded_out.replace("<|im_end|>", "").strip()
        
        predictions.append(decoded_out)
        references.append(formatted["response"])
        
        if idx < 5:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Text: {' '.join(sample['tokens'])}")
            print(f"Ground Truth: {formatted['response']}")
            print(f"Predicted:    {decoded_out}")

    # Compute precision, recall, and F1-score
    gen_metrics = compute_generative_metrics(predictions, references)
    print("\nGenerative Metrics:")
    for k, v in gen_metrics.items():
        print(f"  {k}: {v}")
        
    # Log test metrics manually to MLflow
    mlflow.log_metrics({
        "test_gen_precision": gen_metrics["precision"],
        "test_gen_recall": gen_metrics["recall"],
        "test_gen_f1": gen_metrics["f1"],
        "test_gen_tp": gen_metrics["total_true_positives"],
        "test_gen_fp": gen_metrics["total_false_positives"],
        "test_gen_fn": gen_metrics["total_false_negatives"]
    })
    
    # Save a sample predictions file
    os.makedirs("/tmp/artifacts", exist_ok=True)
    predictions_path = "/tmp/artifacts/generative_predictions.txt"
    with open(predictions_path, "w") as f:
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Sample {idx + 1}\nRef: {ref}\nPred: {pred}\n\n")
    mlflow.log_artifact(predictions_path)
    
    # Configure Unity Catalog Registry if enabled
    if use_unity_catalog:
        mlflow.set_registry_uri("databricks-uc")
        registered_name = f"{unity_catalog_name}.{unity_catalog_schema}.clinical_llm_lora_model"
        print(f"Registering model adapter to Unity Catalog: {registered_name}")
    else:
        registered_name = "clinical_llm_lora_model"
        print(f"Registering model adapter to Workspace Registry: {registered_name}")

    # Infer model signature for Unity Catalog compatibility
    from mlflow.models import infer_signature
    dummy_input = ["Extract diseases from this text: The patient has breast cancer and diabetes."]
    dummy_output = ["breast cancer, diabetes"]
    signature = infer_signature(dummy_input, dummy_output)

    # Save/register the LoRA model adapter
    mlflow.pytorch.log_model(
        pytorch_model=trainer.model,
        artifact_path="lora_adapter",
        registered_model_name=registered_name,
        signature=signature,
        serialization_format="pickle"
    )
    print(f"Training finished. LoRA adapter successfully registered as: {registered_name}")
