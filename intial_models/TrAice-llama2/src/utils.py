import matplotlib.pyplot as plt

# Formatting function for prompts
def formatting_func(example):
    return f"### Question: {example['input']}\n ### Answer: {example['output']}"

# Tokenization and dataset preparation
def generate_and_tokenize_prompt(tokenizer, prompt):
    return tokenizer(formatting_func(prompt))

def generate_and_tokenize_prompt2(tokenizer, prompt, max_length=512):
    result = tokenizer(formatting_func(prompt), truncation=True, max_length=max_length, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    # print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.savefig("data_lengths.png")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )