import os

import pandas as pd
import torch
from adapters import AdapterConfig
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.model.hybridbyt5 import HybridByT5PCAPTokenizer


# Load your QA dataset - replace this with your actual dataset loading code
# Example format: { "pcap_file": "path/to/file.pcap", "question": "What is...", "answer": "The answer is..." }
def load_pcap_qa_dataset(data_dir):
    # Replace this with your actual dataset loading logic
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))

    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return {"train": train_dataset, "validation": val_dataset}


# Initialize the custom tokenizer and model
tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=277)
model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

# Add PCAP-specific adapters
adapter_config = AdapterConfig(
    reduction_factor=16,  # Controls adapter size (higher = smaller)
    non_linearity="relu",
    adapter_residual_before_ln=True
)

AdapterConfig()

# Add adapters to all encoder and decoder layers
model.add_adapter("pcap_adapter", config=adapter_config)
model.train_adapter("pcap_adapter")  # Set this adapter for training


# Preprocess function to handle mixed PCAP and text data
def preprocess_function(examples):
    inputs = []
    targets = []

    for question, pcap_file, answer in zip(examples['question'], examples['pcap_file'], examples['answer']):
        # Create input format: question + PCAP
        # For PCAP files, we use the tokenizer's special method to handle mixed content
        if os.path.exists(pcap_file):
            # Text with PCAP attachment
            encoded_input = tokenizer.tokenize_text_with_pcap(f"Question: {question}", pcap_file)
            inputs.append(encoded_input)
        else:
            # Fallback to text-only if PCAP file not found
            encoded_input = tokenizer.encode_mixed_input(text=f"Question: {question} (PCAP unavailable)")
            inputs.append(encoded_input)

        # Target is just the text answer
        targets.append(answer)

    # Tokenize inputs (already tokenized) and targets
    batch_inputs = {"input_ids": inputs}
    batch_labels = tokenizer(targets, max_length=100, truncation=True, padding='max_length')

    # Set up the labels for the model
    batch_inputs["labels"] = batch_labels["input_ids"]

    return batch_inputs


# Custom data collator to handle varying length inputs
def data_collator(features):
    # Determine max length in this batch
    max_length = max([len(feature["input_ids"]) for feature in features])

    # Create padded batch
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for feature in features:
        # Pad input_ids
        padded_inputs = feature["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(feature["input_ids"]))
        batch["input_ids"].append(padded_inputs)

        # Create attention mask
        attention_mask = [1] * len(feature["input_ids"]) + [0] * (max_length - len(feature["input_ids"]))
        batch["attention_mask"].append(attention_mask)

        # Copy labels
        batch["labels"].append(feature["labels"])

    # Convert to tensors
    batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
    batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
    batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)

    return batch


# Load your dataset
data_dir = "./data"  # Replace with your data directory
dataset = load_pcap_qa_dataset(data_dir)

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments for Phase 1 (adapter training)
phase1_args = TrainingArguments(
    output_dir="./results/phase1",
    evaluation_strategy="epoch",
    learning_rate=5e-4,  # Higher learning rate for adapter training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,  # Shorter training for phase 1
    weight_decay=0.01,
    logging_dir="./logs/phase1",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Phase 1 Trainer (adapter training)
phase1_trainer = Trainer(
    model=model,
    args=phase1_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Train Phase 1
print("Starting Phase 1: Adapter Training")
phase1_trainer.train()
phase1_trainer.save_model("./results/phase1_final")

# Phase 2: Full model fine-tuning
print("Starting Phase 2: Full Model Fine-tuning")

# Load the adapter-trained model
model = T5ForConditionalGeneration.from_pretrained("./results/phase1_final")

# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Training arguments for Phase 2 (full fine-tuning)
phase2_args = TrainingArguments(
    output_dir="./results/phase2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,  # Lower learning rate for full fine-tuning
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs/phase2",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Phase 2 Trainer (full fine-tuning)
phase2_trainer = Trainer(
    model=model,
    args=phase2_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# Train Phase 2
phase2_trainer.train()
phase2_trainer.save_model("./results/final_model")

# Evaluation
print("Evaluating final model on validation set")
eval_results = phase2_trainer.evaluate()
print(f"Evaluation results: {eval_results}")


# Example inference
def process_pcap_query(question, pcap_file):
    # Tokenize the input
    input_ids = tokenizer.tokenize_text_with_pcap(f"Question: {question}", pcap_file)
    input_ids = torch.tensor([input_ids])

    # Generate the answer
    output_ids = model.generate(input_ids, max_length=150)[0].tolist()

    # Decode the output
    output_string = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_string


# Example usage
print("Example inference:")
example_question = "What protocol is used in this capture?"
example_pcap = "./data/example.pcap"  # Replace with an actual PCAP file
if os.path.exists(example_pcap):
    answer = process_pcap_query(example_question, example_pcap)
    print(f"Q: {example_question}")
    print(f"A: {answer}")