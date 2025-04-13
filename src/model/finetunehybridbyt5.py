import binascii
import json
import os
import torch
import pandas as pd
from adapters import SeqBnConfig
from adapters.models.t5 import T5AdapterModel
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
)

from src.model.hybridbyt5 import HybridByT5PCAPTokenizer

def main():
    # Define data directory
    data_dir = "./data"  # Replace with your data directory


    # Load your QA dataset from JSON files
    def load_pcap_qa_dataset(data_dir):
        """Load and parse the QA datasets from JSON files"""
        try:
            # Load train data
            with open("../../data/train.json", 'r') as f:
                train_data = json.load(f)

            # Load validation data
            with open("../../data/test.json", 'r') as f:
                val_data = json.load(f)

            print(f"Successfully loaded {len(train_data)} training and {len(val_data)} validation examples")

            # Convert to HuggingFace Dataset format
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)

            return {"train": train_dataset, "validation": val_dataset}
        except Exception as e:
            print(f"Error loading dataset from {data_dir}: {e}")
            raise


    # Initialize the custom tokenizer and model
    tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=277)
    model = T5AdapterModel.from_pretrained("google/byt5-small")

    # Add PCAP-specific adapters
    adapter_config = SeqBnConfig(
        reduction_factor=16,
        non_linearity="relu",
        adapter_residual_before_ln=True
    )

    # Add adapters to all encoder and decoder layers
    model.add_adapter("pcap_adapter", config=adapter_config)
    model.train_adapter("pcap_adapter")  # Set this adapter for training


    # Preprocess function to handle mixed PCAP and text data
    def preprocess_function(examples):
        inputs = []
        targets = []

        for item in examples:
            question = item['question']
            pcap_content = item['pcap']
            answer = item['answer']

            # Check if pcap_content is a file path or inline content
            if os.path.exists(pcap_content):
                # Text with PCAP file
                encoded_input = tokenizer.tokenize_text_with_pcap(f"Question: {question}", pcap_content)
            else:
                # Inline PCAP content (hex format)
                encoded_input = tokenizer.encode_mixed_input(
                    text = f"Question: {question}",
                    pcap_bytes=binascii.unhexlify(pcap_content)
                )

            inputs.append(encoded_input)
            targets.append(answer)

        # Tokenize inputs (already tokenized) and targets
        batch_inputs = {"input_ids": inputs}
        batch_labels = tokenizer(targets)

        # Set up the labels for the model
        batch_inputs["labels"] = batch_labels["input_ids"]

        return batch_inputs


    def data_collator(features):
        """
        Custom data collator for mixed text and PCAP token data.
        Handles potential deeply nested structures in the dataset.
        """
        # First, do some error checking and logging
        if not features:
            return {}

        # Debug info - can be removed in production
        print(f"Processing batch with {len(features)} features")

        # Extract and normalize input_ids and labels
        input_ids = []
        labels = []

        for feature in features:
            # Handle potential nesting in input_ids
            ids = feature["input_ids"]

            # Recursively flatten any nested structures
            def flatten_recursive(data):
                flattened = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list):
                            flattened.extend(flatten_recursive(item))
                        else:
                            flattened.append(item)
                else:
                    flattened.append(data)
                return flattened

            # Flatten input_ids completely
            ids_flattened = flatten_recursive(ids)
            input_ids.append(ids_flattened)

            # Handle labels
            lbl = feature["labels"]
            lbl_flattened = flatten_recursive(lbl)
            labels.append(lbl_flattened)

        # Determine max length for this batch
        max_input_length = max(len(ids) for ids in input_ids)
        max_label_length = max(len(lbl) for lbl in labels)

        # Get padding token ID from tokenizer configuration
        padding_token_id = tokenizer.pad_token_id

        # Pad input_ids
        padded_input_ids = []
        attention_mask = []

        for ids in input_ids:
            # Create padding
            padding_length = max_input_length - len(ids)
            padded_ids = ids + [padding_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_mask.append(mask)

        # Pad labels (with -100 as padding to ignore these tokens in loss calculation)
        padded_labels = []

        for lbl in labels:
            padding_length = max_label_length - len(lbl)
            padded_lbl = lbl + [-100] * padding_length
            padded_labels.append(padded_lbl)

        # Create batch with error handling
        try:
            batch = {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
            }
            return batch
        except Exception as e:
            print(f"Error creating tensor batch: {e}")
            print(f"Input shape info: {[len(ids) for ids in input_ids]}")
            print(f"Label shape info: {[len(lbl) for lbl in labels]}")
            raise

    # Load your dataset
    dataset = load_pcap_qa_dataset(data_dir)

    # Preprocess the dataset
    tokenized_dataset = {
        split: dataset[split].map(
            lambda examples: preprocess_function([examples]),
            batched=False  # Process one example at a time
        )
        for split in dataset
    }

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

    # Load the adapter-trained model - IMPORTANT: use T5AdapterModel here too, not T5ForConditionalGeneration
    model = T5AdapterModel.from_pretrained("./results/phase1_final")

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


    # Example inference function
    def process_pcap_query(question, pcap_file_or_content):
        """Process a query with either a PCAP file path or inline content"""
        # Check if input is a file path or inline content
        if os.path.exists(pcap_file_or_content):
            # Tokenize from file
            input_ids = tokenizer.tokenize_text_with_pcap(f"Question: {question}", pcap_file_or_content)
        else:
            # Tokenize from inline content
            input_ids = tokenizer.encode_mixed_input(
                text=f"Question: {question}",
                pcap_bytes=binascii.unhexlify(pcap_file_or_content)
            )

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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()