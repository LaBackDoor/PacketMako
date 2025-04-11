import logging
from multimodaltokenizer import MultiModalTokenizer, MultiModalModel

logging.basicConfig(level=logging.INFO)

# Initialize tokenizer and model
tokenizer = MultiModalTokenizer()
model = MultiModalModel()

# Example with text input
text_input = "This is a sample text message for analysis."
text_tokens = tokenizer.tokenize(text_input)
print(f"Text input tokenized to shape: {text_tokens['input_ids'].shape}")

# Example with PCAP input (using a file path)
pcap_file = "sample.pcap"  # Replace with your PCAP file
try:
    pcap_tokens = tokenizer.tokenize(pcap_file)
    print(f"PCAP input tokenized to shape: {pcap_tokens['input_ids'].shape}")

    # Run through the model
    with torch.no_grad():
        text_output = model(**text_tokens)
        pcap_output = model(**pcap_tokens)

    print(f"Text classification logits: {text_output}")
    print(f"PCAP classification logits: {pcap_output}")

except Exception as e:
    print(f"Error processing PCAP: {e}")