import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from transformers import T5PreTrainedModel, T5Config, T5EncoderModel
from torch import nn


def create_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['Class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    return df, class_to_idx


class ByteDataset(Dataset):
    def __init__(self, data_dir, df, class_to_idx, max_length=1024):
        self.data_dir = Path(data_dir)
        self.df = df
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

        # Truncate or pad sequence
        if len(byte_sequence) > self.max_length:
            byte_sequence = byte_sequence[:self.max_length]
        else:
            byte_sequence = byte_sequence + b'\x00' * (self.max_length - len(byte_sequence))

        # Convert to tensor
        input_ids = torch.tensor([b for b in byte_sequence], dtype=torch.long)
        attention_mask = torch.ones(self.max_length, dtype=torch.bool)
        label_idx = self.class_to_idx[label]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_idx, dtype=torch.long)
        }


class ByT5BinaryClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()

        # Load pre-trained ByT5 encoder
        self.byt5 = T5EncoderModel.from_pretrained('google/byt5-small')

        # Get the correct hidden size from the model config
        self.hidden_size = self.byt5.config.hidden_size

        # Freeze some of the encoder layers
        for param in self.byt5.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers for fine-tuning
        for param in self.byt5.encoder.block[-2:].parameters():
            param.requires_grad = True

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),  # Use the actual hidden size from ByT5
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        # Get encoder outputs from ByT5
        outputs = self.byt5(input_ids=input_ids, attention_mask=attention_mask)

        # Global average pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sequence_output = outputs.last_hidden_state
        pooled_output = (sequence_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        # Classification
        logits = self.classifier(pooled_output)

        return logits, pooled_output


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)

    # Separate parameter groups for different learning rates
    encoder_params = list(model.byt5.encoder.block[-2:].parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},
        {'params': classifier_params, 'lr': 2e-4}
    ], weight_decay=0.01)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'models/pretrained_best_model.pt')

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print('----------------------------------------')


def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def main():
    device = setup_device()
    print(f"Using device: {device}")

    try:
        # Load and process data
        df, class_to_idx = create_label_mapping('data/Label.csv')
        num_classes = len(class_to_idx)
        print(f"Found {num_classes} classes: {class_to_idx}")

        # Create datasets
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
        train_dataset = ByteDataset('data/bytes', train_df, class_to_idx)
        val_dataset = ByteDataset('data/bytes', val_df, class_to_idx)
        print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create dataloaders with smaller batch size for memory efficiency
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)

        # Initialize model
        model = ByT5BinaryClassifier(num_classes=num_classes)
        print(f"Model initialized with hidden size: {model.hidden_size}")

        # Train model
        train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()