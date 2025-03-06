import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import glob
from tokenizers import ByteLevelBPETokenizer
from torch import nn


# Load and process labels
def create_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['Class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    return df, class_to_idx


def train_tokenizer(files, vocab_size=8192, min_frequency=2):
    """Custom tokenizer training function for binary data"""
    tokenizer = ByteLevelBPETokenizer()

    # Pre-process binary files into a format the tokenizer can handle
    processed_files = []
    for file_path in files:
        temp_path = f"{file_path}.processed"
        with open(file_path, 'rb') as f_in, open(temp_path, 'w', encoding='utf-8') as f_out:
            bytes_data = f_in.read()
            # Convert bytes to space-separated hex strings
            hex_str = ' '.join(f'{b:02x}' for b in bytes_data)
            f_out.write(hex_str)
        processed_files.append(temp_path)

    # Train the tokenizer on processed files
    tokenizer.train(
        files=processed_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<eos>"]
    )

    # Clean up temporary files
    for temp_file in processed_files:
        Path(temp_file).unlink()

    return tokenizer


def prepare_data(byte_sequence, tokenizer, max_length=1024):
    """Modified prepare_data function with proper attention mask typing"""
    # Truncate or pad the sequence
    if len(byte_sequence) > max_length:
        byte_sequence = byte_sequence[:max_length]
    else:
        byte_sequence = byte_sequence + b'\x00' * (max_length - len(byte_sequence))

    # Convert bytes to hex string format
    hex_str = ' '.join(f'{b:02x}' for b in byte_sequence)

    # Encode using tokenizer
    encoding = tokenizer.encode(hex_str)

    # Convert to tensor and handle padding
    input_ids = torch.tensor(encoding.ids[:max_length], dtype=torch.long)
    attention_mask = torch.ones(max_length, dtype=torch.bool)  # Change to boolean type

    # Pad if necessary
    if len(input_ids) < max_length:
        padding_length = max_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
        attention_mask[len(input_ids) - padding_length:] = False

    return input_ids, attention_mask


class ByteDataset(Dataset):
    def __init__(self, data_dir, df, tokenizer, class_to_idx, max_length=1024):
        self.data_dir = Path(data_dir)
        self.df = df
        self.tokenizer = tokenizer
        self.class_to_idx = class_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bin_file = self.df.iloc[idx]['Bin']
        label = self.df.iloc[idx]['Class']

        # Read binary file
        with open(self.data_dir / bin_file, 'rb') as f:
            byte_sequence = f.read()

        # Prepare input data
        input_ids, attention_mask = prepare_data(byte_sequence, self.tokenizer, self.max_length)
        label_idx = self.class_to_idx[label]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_idx, dtype=torch.long)
        }


class ByteLevelTransformerClassifier(nn.Module):
    def __init__(self, vocab_size=8192, embed_dim=512, num_heads=8, num_layers=6,
                 dim_feedforward=2048, max_seq_length=1024, num_classes=2,
                 dropout=0.1):
        super().__init__()

        self.pad_idx = 0
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Modified transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # This can help with training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x, attention_mask=None):
        B, L = x.shape

        # Embedding and position encoding
        x = self.token_embedding(x) + self.position_embedding[:, :L, :]
        x = self.layer_norm(x)

        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.bool, device=x.device)

        # Convert boolean mask to float mask for compatibility
        float_mask = attention_mask.float().masked_fill(
            attention_mask == 0, float('-inf')).masked_fill(
            attention_mask == 1, float(0.0))

        # Apply transformer with float mask
        x = self.transformer_encoder(x, src_key_padding_mask=float_mask)

        # Global average pooling with mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        # Classification
        logits = self.classifier(x)

        return logits, x


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits, _ = model(input_ids, attention_mask)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print('----------------------------------------')


def setup_device():
    """Configure the appropriate device for training with fallback handling"""
    import os

    # Enable MPS fallback for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    return device


def main():
    device = setup_device()
    print(f"Using device: {device}")

    try:
        # Load and process data
        df, class_to_idx = create_label_mapping('data/Label.csv')
        num_classes = len(class_to_idx)
        print(f"Found {num_classes} classes: {class_to_idx}")

        # Get binary files
        files = glob.glob("data/bytes/*.bin")
        if not files:
            raise FileNotFoundError("No .bin files found in data/bytes/")
        print(f"Found {len(files)} binary files")

        # Train tokenizer with the new approach
        tokenizer = train_tokenizer(files)
        print("Tokenizer training completed")

        # Save tokenizer for future use
        tokenizer.save_model("data/tokenized")
        print("Tokenizer saved to data/tokenized")

        # Create datasets with extra error handling
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)

        train_dataset = ByteDataset('data/bytes', train_df, tokenizer,
                                    class_to_idx)
        val_dataset = ByteDataset('data/bytes', val_df, tokenizer,
                                  class_to_idx)
        print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create dataloaders with smaller batch size for memory efficiency
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        # Initialize and train model
        model = ByteLevelTransformerClassifier(
            vocab_size=8192,
            num_classes=num_classes,
            embed_dim=512,
            num_heads=8,
            num_layers=6
        )

        train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
