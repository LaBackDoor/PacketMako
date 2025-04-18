import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from evaluate import load as load_evaluator
from rouge_score import rouge_scorer

# ── USER CONFIG ────────────────────────────────────────────────────────────────

# Paths to your train and test JSONL files:
TRAIN_JSON = "/path/to/your/train.jsonl"
TEST_JSON = "/path/to/your/test.jsonl"

# Output directory base
OUTPUT_BASE = "./byt5_finetuned"

# List all BYT5 variants you want to try:
MODEL_NAMES = [
    "google/byt5-small",
    "google/byt5-base",
    "google/byt5-large",
    "google/byt5-xl",
    "google/byt5-xxl"
]

# RAM you can devote to shuffling, in bytes:
RAM_SIZE_BYTES = 4 * 1024 ** 3  # e.g. 4 GB

# Approximate average size of one example when loaded:
AVG_EXAMPLE_SIZE_BYTES = 2_000

# Training hyperparams:
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5


# ── END USER CONFIG ───────────────────────────────────────────────────────────

def preprocess_batch(examples, tokenizer):
    # build input and target strings
    inputs = [f"question: {q}  context: {c}"
              for q, c in zip(examples["question"], examples["pcap"])]
    targets = examples["answer"]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace pad tokens in the labels
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in batch] for batch in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # SQuAD-style exact match & F1
    squad = load_evaluator("squad")
    pred_list = [{"id": str(i), "prediction_text": p}
                 for i, p in enumerate(decoded_preds)]
    ref_list = [{"id": str(i),
                 "answers": {"text": [ref], "answer_start": [0]}}
                for i, ref in enumerate(decoded_labels)]
    squad_res = squad.compute(predictions=pred_list, references=ref_list)

    # ROUGE‑L
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(r, p)["rougeL"].fmeasure
                    for p, r in zip(decoded_preds, decoded_labels)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "exact_match": squad_res["exact_match"],
        "f1": squad_res["f1"],
        "rougeL": avg_rouge,
    }


def main():
    # compute buffer size in number of examples
    buffer_size = max(1_000,
                      int(RAM_SIZE_BYTES // AVG_EXAMPLE_SIZE_BYTES))
    print(f"→ shuffle buffer size: {buffer_size:,} examples")

    # stream‑load and shuffle train set
    train_ds = load_dataset(
        "json",
        data_files=TRAIN_JSON,
        split="train",
        streaming=True
    ).shuffle(buffer_size=buffer_size, seed=42)

    # load test set normally (small enough)
    test_ds = load_dataset(
        "json",
        data_files=TEST_JSON,
        split="train"
    )

    # Prepare test set
    tokenizer_for_prep = AutoTokenizer.from_pretrained(MODEL_NAMES[0])
    test_ds = test_ds.map(
        lambda ex: preprocess_batch(ex, tokenizer_for_prep),
        batched=True,
        remove_columns=test_ds.column_names
    )

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_metrics = {}

    for model_name in MODEL_NAMES:
        print(f"\n\n=== Fine‑tuning {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # tokenize train as IterableDataset
        tokenized_train = train_ds.map(
            lambda ex: preprocess_batch(ex, tokenizer),
            batched=True,
            remove_columns=train_ds.column_names
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        )

        output_dir = os.path.join(
            OUTPUT_BASE, model_name.replace("/", "_")
        )
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            save_total_limit=2,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,  # IterableDataset OK
            eval_dataset=test_ds,  # in‑memory Dataset
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, tokenizer)
        )

        trainer.train()
        metrics = trainer.evaluate()
        all_metrics[model_name] = metrics
        # save model
        trainer.save_model(output_dir)

    # dump all metrics
    with open(os.path.join(OUTPUT_BASE, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\n✅ Done. Metrics for each model written to {OUTPUT_BASE}/all_metrics.json")


if __name__ == "__main__":
    main()
