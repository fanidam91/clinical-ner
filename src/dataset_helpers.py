import torch
from datasets import load_dataset

def extract_entities_from_bio(tokens, ner_tags, labels_list):
    """
    Extracts continuous entity strings from tokens and their corresponding BIO tags.
    Example:
      tokens = ["A", "patient", "with", "breast", "cancer"]
      ner_tags = [0, 0, 0, 1, 2]  # O, O, O, B-Disease, I-Disease
      Returns: ["breast cancer"]
    """
    entities = []
    current_entity = []
    
    for token, tag_id in zip(tokens, ner_tags):
        tag_name = labels_list[tag_id]
        if tag_name == "B-Disease":
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)
        elif tag_name == "I-Disease":
            if current_entity:  # Only append if we started an entity
                current_entity.append(token)
        else:  # "O" tag
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
                
    if current_entity:
        entities.append(" ".join(current_entity))
        
    # Clean up token boundaries (e.g. remove spacing for punctuations if necessary)
    # Since CoNLL splits tokens by whitespace, joining them by space is standard.
    # We clean up multiple spaces and filter out empty strings.
    cleaned_entities = []
    for ent in entities:
        ent_clean = " ".join(ent.split()).strip()
        if ent_clean:
            cleaned_entities.append(ent_clean)
            
    # Return unique entities to avoid duplicates in target output
    return list(set(cleaned_entities))


def format_instruction_sample(tokens, ner_tags, labels_list):
    """
    Converts a token/tag sample into a system/user/assistant instruction format.
    """
    sentence = " ".join(tokens)
    diseases = extract_entities_from_bio(tokens, ner_tags, labels_list)
    
    target_output = ", ".join(diseases) if diseases else "None"
    
    system_prompt = (
        "You are an expert clinical NLP assistant. Your task is to extract all disease names "
        "mentioned in the clinical text. Output the extracted diseases as a comma-separated list. "
        "If no diseases are mentioned, output 'None'."
    )
    user_prompt = f"Extract diseases from this text: {sentence}"
    
    return {
        "system": system_prompt,
        "user": user_prompt,
        "response": target_output
    }


def tokenize_and_align_labels(examples, tokenizer, max_length=256):
    """
    Tokenizes text and aligns token classification labels (BIO) with subwords.
    Wordpiece tokenization splits tokens, so subwords should be ignored (labeled -100).
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens are mapped to None. Set their label to -100 so they are ignored.
            if word_idx is None:
                label_ids.append(-100)
            # Set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subwords, set the label to -100.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class GenerativeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Causal LLM fine-tuning.
    Formats inputs using ChatML or standard instruction prompt format.
    """
    def __init__(self, hf_dataset, tokenizer, labels_list, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for sample in hf_dataset:
            formatted = format_instruction_sample(sample["tokens"], sample["ner_tags"], labels_list)
            
            # Format text into ChatML-like structure:
            # <|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
            text = (
                f"<|im_start|>system\n{formatted['system']}<|im_end|>\n"
                f"<|im_start|>user\n{formatted['user']}<|im_end|>\n"
                f"<|im_start|>assistant\n{formatted['response']}<|im_end|>"
            )
            
            self.samples.append({
                "text": text,
                "input_text": f"<|im_start|>system\n{formatted['system']}<|im_end|>\n<|im_start|>user\n{formatted['user']}<|im_end|>\n<|im_start|>assistant\n",
                "target_text": formatted["response"]
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize full prompt for training
        tokenized = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize only prompt to identify where the label masks should start
        # During training, we mask the loss on prompt tokens (setting label to -100)
        prompt_tokenized = self.tokenizer(
            sample["input_text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids, but prompt tokens masked with -100
        labels = input_ids.clone()
        prompt_len = prompt_tokenized["input_ids"].shape[1]
        
        # Mask prompt tokens
        labels[:prompt_len] = -100
        # Mask padding tokens (assuming pad_token_id is defined and is padded on right)
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
