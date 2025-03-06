from pathlib import Path

import torch
from torch import nn
from transformers import T5EncoderModel, AutoTokenizer, AutoModelForCausalLM


class Byt5LlamaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 data_dir,
                 class_to_idx,
                 explanations_dir,
                 byt5_tokenizer,
                 llama_tokenizer,
                 max_length=1024,
                 max_prompt_len=256,
                 max_target_len=256):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.explanations_dir = Path(explanations_dir)
        self.byt5_tokenizer = byt5_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.max_length = max_length
        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bin_file = row['Bin']
        label_str = row['Class']
        explanation_path = self.explanations_dir / bin_file.replace('.bin', '.txt')

        with open(self.data_dir / bin_file, 'rb') as f:
            byte_data = f.read()
        if len(byte_data) > self.max_length:
            byte_data = byte_data[:self.max_length]
        else:
            byte_data = byte_data + b'\x00' * (self.max_length - len(byte_data))

        byte_input_ids = torch.tensor(list(byte_data), dtype=torch.long)
        byte_attention_mask = torch.ones(self.max_length, dtype=torch.bool)

        prompt_text = f"Classification: {label_str}\nExplain why this traffic is {label_str}:\n"
        prompt_tokens = self.llama_tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt"
        )

        with open(explanation_path, 'r', encoding='utf-8') as f:
            explanation_text = f.read().strip()
        target_tokens = self.llama_tokenizer(
            explanation_text,
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt"
        )

        return {
            "byte_input_ids": byte_input_ids,
            "byte_attention_mask": byte_attention_mask,
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "explanation_input_ids": target_tokens["input_ids"].squeeze(0),
            "explanation_attention_mask": target_tokens["attention_mask"].squeeze(0),
            "labels": target_tokens["input_ids"].squeeze(0),
        }


class ByT5Embedder(nn.Module):
    def __init__(self, byt5_model_name='google/byt5-small', trainable_layers=2):
        super().__init__()
        self.byt5 = T5EncoderModel.from_pretrained(byt5_model_name)
        for param in self.byt5.parameters():
            param.requires_grad = False
        if trainable_layers > 0:
            for param in self.byt5.encoder.block[-trainable_layers:].parameters():
                param.requires_grad = True
        self.hidden_size = self.byt5.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.byt5(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_output = (last_hidden_state * mask_expanded).sum(1)
        pooled_output = pooled_output / mask_expanded.sum(1).clamp(min=1e-9)
        return pooled_output


class EmbeddingBridging(nn.Module):
    def __init__(self, byt5_hidden_size, llama_hidden_size, num_virtual_tokens=4):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.proj = nn.Linear(byt5_hidden_size, num_virtual_tokens * llama_hidden_size)

    def forward(self, pooled_embeds):
        x = self.proj(pooled_embeds)
        bsz = x.size(0)
        x = x.view(bsz, self.num_virtual_tokens, -1)
        return x


class FullMultimodalModel(nn.Module):
    def __init__(self,
                 byt5_model_name='google/byt5-small',
                 llama_model_name='meta-llama/Llama-7B-Instruct',
                 num_virtual_tokens=4,
                 trainable_layers=2):
        super().__init__()

        self.byt5_embedder = ByT5Embedder(byt5_model_name=byt5_model_name,
                                          trainable_layers=trainable_layers)
        byt5_hidden_size = self.byt5_embedder.hidden_size

        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            load_in_8bit=True,  # Enable 8-bit loading
            device_map='auto'  # Automatically map model to available devices
        )
        self.llama_model.config.use_cache = False
        self.llama_model.gradient_checkpointing_enable()
        llama_hidden_size = self.llama_model.config.hidden_size

        self.bridging = EmbeddingBridging(byt5_hidden_size, llama_hidden_size, num_virtual_tokens)
        self.num_virtual_tokens = num_virtual_tokens

    def forward(self,
                byte_input_ids,
                byte_attention_mask,
                prompt_input_ids,
                prompt_attention_mask,
                explanation_input_ids,
                explanation_attention_mask,
                labels=None):
        with torch.set_grad_enabled(self.byt5_embedder.training):
            pooled_output = self.byt5_embedder(byte_input_ids, byte_attention_mask)

        bridging_tokens = self.bridging(pooled_output)

        with torch.set_grad_enabled(self.llama_model.training):
            token_embedder = self.llama_model.get_input_embeddings()
            prompt_embeds = token_embedder(prompt_input_ids)

        explanation_embeds = token_embedder(explanation_input_ids)

        combined_embeds = torch.cat([bridging_tokens, prompt_embeds, explanation_embeds], dim=1)

        bsz = bridging_tokens.size(0)
        bridging_attention = torch.ones((bsz, self.num_virtual_tokens), dtype=prompt_attention_mask.dtype,
                                        device=prompt_attention_mask.device)
        combined_attention_mask = torch.cat([
            bridging_attention,
            prompt_attention_mask,
            explanation_attention_mask
        ], dim=1)

        if labels is not None:
            pad_len = self.num_virtual_tokens + prompt_input_ids.size(1)
            pad = torch.full((labels.size(0), pad_len), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([pad, labels], dim=1)

        combined_embeds = combined_embeds.half()  # Cast to FP16
        output = self.llama_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        return output


def collate_fn(batch):
    byte_input_ids = [x["byte_input_ids"] for x in batch]
    byte_attention_mask = [x["byte_attention_mask"] for x in batch]
    prompt_input_ids = [x["prompt_input_ids"] for x in batch]
    prompt_attention_mask = [x["prompt_attention_mask"] for x in batch]
    explanation_input_ids = [x["explanation_input_ids"] for x in batch]
    explanation_attention_mask = [x["explanation_attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    byte_input_ids = nn.utils.rnn.pad_sequence(byte_input_ids, batch_first=True, padding_value=0)
    byte_attention_mask = nn.utils.rnn.pad_sequence(byte_attention_mask, batch_first=True, padding_value=0)
    prompt_input_ids = nn.utils.rnn.pad_sequence(prompt_input_ids, batch_first=True, padding_value=0)
    prompt_attention_mask = nn.utils.rnn.pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
    explanation_input_ids = nn.utils.rnn.pad_sequence(explanation_input_ids, batch_first=True, padding_value=0)
    explanation_attention_mask = nn.utils.rnn.pad_sequence(explanation_attention_mask, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "byte_input_ids": byte_input_ids,
        "byte_attention_mask": byte_attention_mask,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "explanation_input_ids": explanation_input_ids,
        "explanation_attention_mask": explanation_attention_mask,
        "labels": labels
    }


def train_multimodal(model, dataloader, optimizer, device="cuda", epochs=1):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            byte_input_ids = batch["byte_input_ids"].to(device)
            byte_attention_mask = batch["byte_attention_mask"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            explanation_input_ids = batch["explanation_input_ids"].to(device)
            explanation_attention_mask = batch["explanation_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Directly call the model without autocast
            outputs = model(
                byte_input_ids=byte_input_ids,
                byte_attention_mask=byte_attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                explanation_input_ids=explanation_input_ids,
                explanation_attention_mask=explanation_attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward and step without scaling
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


# ================= Evaluation Functions =================

def evaluate_model(model, dataset, device="cuda"):
    model.eval()
    model.to(device)

    # Take a single sample for demonstration
    sample = dataset[0]
    # Add batch dimension and move tensors to device
    for key in sample:
        sample[key] = sample[key].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            byte_input_ids=sample["byte_input_ids"],
            byte_attention_mask=sample["byte_attention_mask"],
            prompt_input_ids=sample["prompt_input_ids"],
            prompt_attention_mask=sample["prompt_attention_mask"],
            explanation_input_ids=sample["explanation_input_ids"],
            explanation_attention_mask=sample["explanation_attention_mask"],
            labels=None
        )

    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded_output = model.llama_tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print("Decoded Output:", decoded_output)


def evaluate_on_dataloader(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)
    all_decoded_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(device)

            outputs = model(
                byte_input_ids=batch["byte_input_ids"],
                byte_attention_mask=batch["byte_attention_mask"],
                prompt_input_ids=batch["prompt_input_ids"],
                prompt_attention_mask=batch["prompt_attention_mask"],
                explanation_input_ids=batch["explanation_input_ids"],
                explanation_attention_mask=batch["explanation_attention_mask"],
                labels=None
            )

            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            decoded_texts = [
                model.llama_tokenizer.decode(ids, skip_special_tokens=True)
                for ids in predicted_ids
            ]
            all_decoded_outputs.extend(decoded_texts)

    return all_decoded_outputs


def main_finetune():
    import pandas as pd
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AdamW

    df = pd.read_csv('data/Label.csv')
    class_names = sorted(df['Class'].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    llama_model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)

    dataset = Byt5LlamaDataset(
        df=df,
        data_dir="data/bytes",
        class_to_idx=class_to_idx,
        explanations_dir="data/explanations",
        byt5_tokenizer=None,
        llama_tokenizer=llama_tokenizer,
        max_length=1024
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = FullMultimodalModel(
        byt5_model_name="google/byt5-small",
        llama_model_name=llama_model_name,
        num_virtual_tokens=4,
        trainable_layers=2
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_multimodal(model, dataloader, optimizer, device="cuda", epochs=3)

    torch.save(model.state_dict(), "multimodal_model.pt")

    # Evaluate the model on a single sample
    evaluate_model(model, dataset, device="cuda")

    # Optionally, evaluate on the entire dataloader
    decoded_outputs = evaluate_on_dataloader(model, dataloader, device="cuda")
    for text in decoded_outputs[:5]:
        print(text)


if __name__ == "__main__":
    main_finetune()
