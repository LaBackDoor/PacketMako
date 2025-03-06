import glob
import math
import os

import torch
import torch.nn as nn
from scapy.all import rdpcap
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

#############################
# Custom PCAP Tokenization
#############################

# Special token IDs (byte values: 0-255; special tokens: 256+)
SOS = 256  # Start-of-sequence token
EOS = 257  # End-of-sequence token
PACKET_SEP = 258  # Packet separator token (marks end of a packet)
COL_SEP = 259  # Column separator token (marks end of a layer)
VOCAB_SIZE = 260  # 256 + 4 special tokens


def get_packet_layers(packet):
    """
    Recursively extracts layers from a packet.
    Returns a list of layers.
    """
    layers = []
    current_layer = packet
    while True:
        layers.append(current_layer)
        if not hasattr(current_layer, 'payload') or current_layer.payload is None:
            break
        if bytes(current_layer.payload) == b'' or current_layer.payload == current_layer:
            break
        current_layer = current_layer.payload
    return layers


def tokenize_pcap(file_path, max_layer_length=128):
    """
    Tokenizes a PCAP file into a flat sequence of tokens and builds 2D indices.

    For each packet:
      - Split into layers.
      - For each layer, tokenize each byte (truncated to max_layer_length) and insert a COL_SEP after.
      - Append a PACKET_SEP after all layers of the packet.

    Inserts SOS at the beginning and EOS at the end.

    Returns:
      token_list: List[int] of tokens.
      row_list: List[int] of row indices.
      col_list: List[int] of column indices.
      max_rows: int (total rows in this file).
      max_cols: int (max column index encountered + 1).
    """
    packets = rdpcap(file_path)
    token_list = []
    row_list = []
    col_list = []

    # Insert SOS token at row 0, col 0
    token_list.append(SOS)
    row_list.append(0)
    col_list.append(0)

    current_row = 1  # Each packet gets its own row
    max_cols = 0
    for packet in packets:
        layers = get_packet_layers(packet)
        layer_index = 0
        for layer in layers:
            layer_bytes = list(bytes(layer))
            if max_layer_length is not None:
                layer_bytes = layer_bytes[:max_layer_length]
            for offset, byte in enumerate(layer_bytes):
                token_list.append(byte)
                row_list.append(current_row)
                col_index = layer_index * max_layer_length + offset
                col_list.append(col_index)
            # Insert COL_SEP after each layer
            token_list.append(COL_SEP)
            row_list.append(current_row)
            col_index = layer_index * max_layer_length + len(layer_bytes)
            col_list.append(col_index)
            layer_index += 1
        # Append PACKET_SEP at the end of packet
        token_list.append(PACKET_SEP)
        row_list.append(current_row)
        col_list.append(layer_index * max_layer_length)
        max_cols = max(max_cols, layer_index * max_layer_length + 1)
        current_row += 1

    # Append EOS token on its own row.
    token_list.append(EOS)
    row_list.append(current_row)
    col_list.append(0)
    max_rows = current_row + 1
    return token_list, row_list, col_list, max_rows, max_cols


#############################
# 2D Positional Encoding & Embedding
#############################

class TwoDPositionalEncoding(nn.Module):
    """
    Basic 2D positional encoding: learns separate embeddings for row and column.
    """

    def __init__(self, max_rows, max_cols, d_model):
        super(TwoDPositionalEncoding, self).__init__()
        self.row_embedding = nn.Embedding(max_rows, d_model)
        self.col_embedding = nn.Embedding(max_cols, d_model)

    def forward(self, row_indices, col_indices):
        row_pos = self.row_embedding(row_indices)
        col_pos = self.col_embedding(col_indices)
        return row_pos + col_pos


class TokenEmbeddingWith2DPos(nn.Module):
    """
    Combines token embeddings with 2D positional encodings.
    """

    def __init__(self, vocab_size, d_model, max_rows, max_cols):
        super(TokenEmbeddingWith2DPos, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = TwoDPositionalEncoding(max_rows, max_cols, d_model)

    def forward(self, tokens, row_indices, col_indices):
        token_embeddings = self.token_embed(tokens)
        pos_embeddings = self.pos_encoder(row_indices, col_indices)
        return token_embeddings + pos_embeddings


#############################
# File Matching Utilities
#############################

def find_matching_files(pcap_dir, expl_dir):
    """
    Find matching pairs of PCAP and explanation files.

    Args:
        pcap_dir: Directory containing PCAP files
        expl_dir: Directory containing explanation files

    Returns:
        list of tuples: (pcap_path, explanation_path) for matching files
    """
    # Get lists of all files
    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap"))
    expl_files = glob.glob(os.path.join(expl_dir, "*.txt"))

    # Create dictionaries with basenames as keys
    pcap_dict = {}
    for pcap_path in pcap_files:
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(pcap_path))[0]
        pcap_dict[base_name] = pcap_path

    expl_dict = {}
    for expl_path in expl_files:
        # Handle both direct matches and _explanation suffix
        base_name = os.path.splitext(os.path.basename(expl_path))[0]
        if base_name.endswith("_explanation"):
            base_name = base_name[:-12]  # Remove _explanation suffix
        expl_dict[base_name] = expl_path

    # Find matches
    matched_pairs = []
    for base_name, pcap_path in pcap_dict.items():
        if base_name in expl_dict:
            matched_pairs.append((pcap_path, expl_dict[base_name]))

    print(f"Found {len(matched_pairs)} matching PCAP and explanation file pairs")
    print(f"Ignored {len(pcap_files) - len(matched_pairs)} PCAP files without matching explanations")
    print(f"Ignored {len(expl_files) - len(matched_pairs)} explanation files without matching PCAPs")

    return matched_pairs


#############################
# Dataset Class: PCAP & Explanation with File Matching
#############################

class PCAPExplanationDataset(torch.utils.data.Dataset):
    """
    Dataset that reads matched PCAP files and corresponding explanation text files.
    Only processes files that have both PCAP and explanation available.
    """

    def __init__(self, pcap_dir, expl_dir, text_tokenizer, max_layer_length=128):
        super().__init__()
        # Find matched files instead of assuming matching indices
        self.file_pairs = find_matching_files(pcap_dir, expl_dir)
        self.text_tokenizer = text_tokenizer
        self.max_layer_length = max_layer_length

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        # Get the matched file pair
        pcap_path, expl_path = self.file_pairs[idx]

        # Tokenize PCAP file
        tokens_list, row_list, col_list, pcap_max_rows, pcap_max_cols = tokenize_pcap(pcap_path, self.max_layer_length)

        # Tokenize explanation text (target)
        try:
            with open(expl_path, "r", encoding="utf-8") as f:
                expl_text = f.read().strip()
            expl_encoding = self.text_tokenizer.encode(expl_text, add_special_tokens=False)
        except Exception as e:
            print(f"Error processing explanation file {expl_path}: {e}")
            # Return empty encoding if there's an error
            expl_encoding = []

        # For explanation tokens, assign a fixed row value (e.g. pcap_max_rows) and sequential columns.
        expl_row = [pcap_max_rows] * len(expl_encoding)
        expl_col = list(range(len(expl_encoding)))

        # Combine: we use the EOS token from the pcap part as a separator between input and target.
        combined_tokens = tokens_list + expl_encoding + [EOS]
        combined_rows = row_list + expl_row + [pcap_max_rows]
        combined_cols = col_list + expl_col + [0]

        # Labels: for causal LM training, we use the entire sequence (shifted by one)
        sample = {
            "input_ids": torch.tensor(combined_tokens, dtype=torch.long),
            "row_ids": torch.tensor(combined_rows, dtype=torch.long),
            "col_ids": torch.tensor(combined_cols, dtype=torch.long),
            "labels": torch.tensor(combined_tokens, dtype=torch.long),
            "pcap_path": pcap_path,
            "expl_path": expl_path
        }
        return sample


#############################
# Data Collator for Padding
#############################

def data_collator(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    row_ids = pad_sequence([item["row_ids"] for item in batch], batch_first=True, padding_value=0)
    col_ids = pad_sequence([item["col_ids"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(0)
    return {
        "input_ids": input_ids,
        "row_ids": row_ids,
        "col_ids": col_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


#############################
# Custom Model Wrapper: Use Custom Embeddings
#############################

class CustomEmbeddingWrapper(nn.Module):
    """
    Wraps a causal LM model so that it uses our custom token+2D positional embedding.
    Instead of passing input_ids, we compute inputs_embeds and pass them to the model.
    """

    def __init__(self, model, custom_embed):
        super().__init__()
        self.model = model
        self.custom_embed = custom_embed

    def forward(self, input_ids, row_ids, col_ids, **kwargs):
        embeddings = self.custom_embed(input_ids, row_ids, col_ids)
        return self.model(inputs_embeds=embeddings, **kwargs)


#############################
# Main Training Script
#############################

def main():
    # Directories for PCAP and explanation text files.
    pcap_dir = "/home/resbears/PycharmProjects/Packet Analysis Data/data_streams/pcap_streams"
    expl_dir = "/home/resbears/PycharmProjects/Packet Analysis Data/data_streams/explaination_streams"

    # Load text tokenizer for the target (Llama 1B text tokenizer).
    model_name = "meta-llama/Llama-3.2-1B"
    text_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Create dataset with proper file matching
    dataset = PCAPExplanationDataset(pcap_dir, expl_dir, text_tokenizer, max_layer_length=128)

    # Check if we have any valid pairs
    if len(dataset) == 0:
        print("No matching PCAP and explanation file pairs found. Exiting.")
        return

    # 80/20 split on matched files
    indices = list(range(len(dataset)))
    train_indices, eval_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    print(f"Split dataset into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples")

    # For our custom 2D embedding, we need global max_rows and max_cols.
    all_max_rows = []
    all_max_cols = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        all_max_rows.append(item["row_ids"].max().item() + 1)
        all_max_cols.append(item["col_ids"].max().item() + 1)
    global_max_rows = max(all_max_rows)
    global_max_cols = max(all_max_cols)
    print(f"Global max_rows: {global_max_rows}, Global max_cols: {global_max_cols}")

    # Load pre-trained Llama-1B causal LM.
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False)

    d_model = 128  # Embedding dimension (choose based on your fine-tuning requirements)
    # Create our custom embedding module.
    custom_embed = TokenEmbeddingWith2DPos(VOCAB_SIZE, d_model, global_max_rows, global_max_cols)
    # Replace the model's input embedding layer by wrapping it.
    model = CustomEmbeddingWrapper(base_model, custom_embed)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./llama_pcap_finetune",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=False,
    )

    # Define a simple compute_metrics function that reports perplexity.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Only consider non -100 labels.
        mask = labels != -100
        loss = nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction="none")
        loss = loss[mask].mean().item()
        perplexity = math.exp(loss)
        return {"perplexity": perplexity}

    # Create Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training.
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()