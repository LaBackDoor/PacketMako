import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from src.tokenization.pcap_tokenizer import PCAPTokenizer


class MultiModalTokenizer:
    def __init__(self, text_tokenizer_name="bert-base-uncased", pcap_vocab_size=260):
        # Initialize text tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)

        # Initialize PCAP tokenizer
        self.pcap_tokenizer = PCAPTokenizer(vocab_size=pcap_vocab_size)

        # Add special tokens for identifying modality
        self.modality_identifiers = {
            'text': 0,
            'pcap': 1
        }

    def detect_input_type(self, input_data):
        """Determine if input is text or PCAP"""
        if isinstance(input_data, str):
            if input_data.lower().endswith('.pcap') and os.path.isfile(input_data):
                return 'pcap'
            else:
                return 'text'
        elif isinstance(input_data, (bytes, bytearray)) or hasattr(input_data, 'read'):
            # Assume it's a PCAP file or file-like object
            return 'pcap'
        else:
            raise ValueError("Unsupported input type")

    def tokenize(self, input_data, max_length=512):
        """Tokenize input based on its type"""
        input_type = self.detect_input_type(input_data)

        if input_type == 'text':
            # Tokenize text using standard text tokenizer
            encoding = self.text_tokenizer(
                input_data,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'modality_type': torch.tensor([self.modality_identifiers['text']])
            }

        elif input_type == 'pcap':
            # For PCAP files, extract flows and tokenize
            if isinstance(input_data, (bytes, bytearray)):
                # Write bytes to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pcap') as temp_file:
                    temp_file.write(input_data)
                    temp_file.flush()
                    tokenized_flows = self.pcap_tokenizer.tokenize_pcap(temp_file.name)
            else:
                # Assume it's a file path
                tokenized_flows = self.pcap_tokenizer.tokenize_pcap(input_data)

            # Process tokenized flows
            all_flow_tokens = []
            for flow_id, tokens in tokenized_flows.items():
                # Truncate if needed
                tokens = tokens[:max_length]
                # Pad to max_length
                if len(tokens) < max_length:
                    tokens = tokens + [0] * (max_length - len(tokens))
                all_flow_tokens.append(tokens)

            if not all_flow_tokens:
                # No valid flows found, return empty tensor
                return {
                    'input_ids': torch.zeros((1, max_length), dtype=torch.long),
                    'attention_mask': torch.zeros((1, max_length), dtype=torch.long),
                    'modality_type': torch.tensor([self.modality_identifiers['pcap']])
                }

            # Convert to tensor
            input_ids = torch.tensor(all_flow_tokens, dtype=torch.long)
            attention_mask = (input_ids != 0).long()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'modality_type': torch.tensor([self.modality_identifiers['pcap']])
            }


class MultiModalModel(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", pcap_embedding_dim=256, hidden_size=768):
        super().__init__()

        # Text model
        self.text_model = AutoModel.from_pretrained(text_model_name)

        # PCAP embedding layer
        self.pcap_embedding = nn.Embedding(260, pcap_embedding_dim)  # 260 tokens as per TrafficGPT

        # Projection layer to align dimensions
        self.pcap_projector = nn.Linear(pcap_embedding_dim, hidden_size)

        # Modality type embedding
        self.modality_embedding = nn.Embedding(2, hidden_size)  # 2 modalities: text and PCAP

        # Transformer layers for joint processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True),
            num_layers=4
        )

        # Output layer
        self.classifier = nn.Linear(hidden_size, 2)  # Example: binary classification

    def forward(self, input_ids, attention_mask, modality_type):
        batch_size = input_ids.size(0)

        if modality_type[0] == 0:  # Text modality
            # Process with text model
            outputs = self.text_model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        else:  # PCAP modality
            # Get embeddings from PCAP tokens
            embeddings = self.pcap_embedding(input_ids)
            # Project to the same dimension as text embeddings
            embeddings = self.pcap_projector(embeddings)

        # Add modality embedding
        modality_embed = self.modality_embedding(modality_type)
        # Expand modality embedding to add to all token positions
        modality_embed = modality_embed.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        embeddings = embeddings + modality_embed

        # Process through transformer layers
        transformer_output = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())

        # Use [CLS] token or mean pooling for classification
        cls_output = transformer_output[:, 0]  # Use first token

        # Final classification
        logits = self.classifier(cls_output)

        return logits