import os
import tempfile
import time

import matplotlib.pyplot as plt
from scapy.all import Raw, wrpcap
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether

from src.model.hybridbyt5 import HybridByT5PCAPTokenizer


# Helper function to create mock PCAP data for testing
def create_mock_pcap_file(filename, num_packets=5):
    """Create a simple PCAP file with mock packets for testing."""
    packets = []
    for i in range(num_packets):
        # Create a simple packet with Ethernet, IP, TCP, and some payload
        payload = f"Test packet {i + 1}".encode()
        pkt = Ether(src="00:11:22:33:44:55", dst="AA:BB:CC:DD:EE:FF") / \
              IP(src="192.168.1.1", dst="192.168.1.2") / \
              TCP(sport=1234, dport=80) / \
              Raw(load=payload)
        packets.append(pkt)

    # Write packets to a PCAP file
    wrpcap(filename, packets)
    return filename


# Helper function to visualize token IDs
def visualize_token_ids(token_ids, title, save_path=None):
    """Visualize token IDs with color coding and labels for special tokens."""
    plt.figure(figsize=(30, 6))

    # Define colors for different token types
    colors = []
    labels = []
    token_names = []  # New list to store actual token text

    # Create reverse mapping for pcap special tokens
    rev_pcap_map = {v: k for k, v in tokenizer.pcap_to_byt5_token_map.items()}

    # Add colors and labels based on token types
    for token_id in token_ids:
        if token_id == tokenizer.text_start_token_id:
            colors.append('green')
            labels.append('text_start')
            token_names.append("<text_start>")
        elif token_id == tokenizer.text_end_token_id:
            colors.append('darkgreen')
            labels.append('text_end')
            token_names.append("<text_end>")
        elif token_id == tokenizer.pcap_start_token_id:
            colors.append('blue')
            labels.append('pcap_start')
            token_names.append("<pcap_start>")
        elif token_id == tokenizer.pcap_end_token_id:
            colors.append('darkblue')
            labels.append('pcap_end')
            token_names.append("<pcap_end>")
        elif token_id == tokenizer.pcap_attachment_token_id:
            colors.append('purple')
            labels.append('pcap_attachment')
            token_names.append("<pcap_attachment>")
        elif token_id in tokenizer.pcap_to_byt5_token_map.values():
            # Get the original PCAP token ID
            pcap_token_id = rev_pcap_map[token_id]

            # Determine what kind of PCAP token it is
            if pcap_token_id == tokenizer.pcap_tokenizer.special_tokens['packet_start']:
                colors.append('orange')
                labels.append('packet_start')
                token_names.append("<packet_start>")
            elif pcap_token_id == tokenizer.pcap_tokenizer.special_tokens['end']:
                colors.append('darkorange')
                labels.append('packet_end')
                token_names.append("<packet_end>")
            else:
                # Must be a link type token - find which one
                for link_type, link_token_id in tokenizer.pcap_tokenizer.link_types.items():
                    if link_token_id == pcap_token_id:
                        colors.append('yellow')
                        labels.append(f'link_type_{link_type}')
                        token_names.append(f"<link_type_{link_type}>")
                        break
                else:
                    # If we didn't find a match
                    colors.append('orange')
                    labels.append('pcap_special')
                    token_names.append("unknown_pcap_special")
        else:
            colors.append('lightgray')
            labels.append(str(token_id))

            # Try to decode regular tokens (bytes)
            if tokenizer.offset <= token_id < tokenizer.offset + 256:
                byte_val = token_id - tokenizer.offset
                if 32 <= byte_val <= 126:  # Printable ASCII
                    token_names.append(f"'{chr(byte_val)}'")
                else:
                    token_names.append(f"byte_{byte_val}")
            else:
                token_names.append(str(token_id))

    # Create bar plot of token IDs
    plt.bar(range(len(token_ids)), token_ids, color=colors)

    # Add token ID values on top of bars
    for i, v in enumerate(token_ids):
        plt.text(i, v + 5, str(v), ha='center', fontsize=8)

    # Add labels below bars (not just for special tokens)
    for i, (label, name) in enumerate(zip(labels, token_names)):
        if label in ['text_start', 'text_end', 'pcap_start', 'pcap_end',
                     'pcap_attachment', 'packet_start', 'packet_end'] or label.startswith('link_type_'):
            plt.text(i, -20, name, ha='center', rotation=90, fontsize=8)

    plt.title(title)
    plt.xlabel('Token Position')
    plt.ylabel('Token ID')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Print the first few tokens with their IDs and names
    print("\nFirst 20 tokens with names:")
    for i, (tid, name) in enumerate(zip(token_ids[:20], token_names[:20])):
        print(f"Position {i}: ID {tid} = {name}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Helper function to print and visualize token sequences
def analyze_tokens(tokens, title, visualize=True, save_path=None):
    """Print token details and optionally visualize them."""
    print(f"\n=== {title} ===")
    print(f"Total tokens: {len(tokens)}")

    # Print the first 20 tokens (or all if less than 20)
    print("First tokens:", tokens[:min(20, len(tokens))])

    # Count special tokens
    special_tokens = {
        'text_start': tokenizer.text_start_token_id,
        'text_end': tokenizer.text_end_token_id,
        'pcap_start': tokenizer.pcap_start_token_id,
        'pcap_end': tokenizer.pcap_end_token_id,
        'pcap_attachment': tokenizer.pcap_attachment_token_id
    }

    for name, token_id in special_tokens.items():
        count = tokens.count(token_id)
        if count > 0:
            print(f"  {name} tokens: {count}")

    if visualize:
        visualize_token_ids(tokens, title, save_path)


# Test cases
def run_test_cases():
    # Initialize tokenizer
    print("Initializing HybridByT5PCAPTokenizer...")
    global tokenizer
    tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=277)

    # Test Case 1: Text-only input
    text_sample = "This is a sample text for tokenization testing."
    text_tokens = tokenizer.encode_mixed_input(text=text_sample)
    analyze_tokens(text_tokens, "Text-only Tokenization", save_path="text_only_tokens.png")

    # Create mock PCAP data
    temp_dir = tempfile.mkdtemp()
    mock_pcap_path = os.path.join(temp_dir, "mock_sample.pcap")

    try:
        # Create mock PCAP file
        print("\nCreating mock PCAP file...")
        create_mock_pcap_file(mock_pcap_path)

        # Test Case 2: PCAP-only input (file path)
        pcap_tokens = tokenizer.encode_mixed_input(pcap_file_path=mock_pcap_path)
        analyze_tokens(pcap_tokens, "PCAP-only Tokenization (file)", save_path="pcap_file_tokens.png")

        # Read PCAP bytes for byte-based tests
        with open(mock_pcap_path, 'rb') as f:
            pcap_bytes = f.read()

        # Test Case 3: PCAP-only input (bytes)
        pcap_bytes_tokens = tokenizer.encode_mixed_input(pcap_bytes=pcap_bytes)
        analyze_tokens(pcap_bytes_tokens, "PCAP-only Tokenization (bytes)", save_path="pcap_bytes_tokens.png")

        # Test Case 4: Text with PCAP file attachment
        text_with_pcap_tokens = tokenizer.tokenize_text_with_pcap(text_sample, mock_pcap_path)
        analyze_tokens(text_with_pcap_tokens, "Text with PCAP File", save_path="text_with_pcap_tokens.png")

        # Test Case 5: Text followed by PCAP bytes
        text_then_pcap_tokens = tokenizer.tokenize_text_followed_by_pcap(text_sample, pcap_bytes)
        analyze_tokens(text_then_pcap_tokens, "Text followed by PCAP Bytes", save_path="text_then_pcap_tokens.png")

        # Test Case 6: PCAP bytes followed by text
        pcap_then_text_tokens = tokenizer.tokenize_pcap_followed_by_text(pcap_bytes, text_sample)
        analyze_tokens(pcap_then_text_tokens, "PCAP Bytes followed by Text", save_path="pcap_then_text_tokens.png")

        # Test Case 7: Decoding tokens back to mixed format
        print("\n=== Decoding Test ===")
        decoded_data = tokenizer.decode_mixed_input(text_with_pcap_tokens)
        print("Decoded content types:", list(decoded_data.keys()))
        if 'text' in decoded_data:
            print("Decoded text:", decoded_data['text'])
        if 'pcap_data' in decoded_data:
            print("PCAP data retrieved (showing first 50 bytes):", decoded_data['pcap_data'][:50])

        # Test Case 8: More complex text for visualization
        complex_text = """This is a longer sample with multiple sentences. 
        It contains various special characters: !@#$%^&*()_+
        And numbers: 1234567890
        To better visualize the tokenization process."""

        complex_tokens = tokenizer.encode_mixed_input(text=complex_text)
        analyze_tokens(complex_tokens, "Complex Text Tokenization", save_path="complex_text_tokens.png")

        # Test Case 9: Distribution of token IDs
        print("\n=== Token ID Distribution Analysis ===")
        all_tokens = text_with_pcap_tokens  # Use one of the more complex examples

        # Create a histogram of token IDs
        plt.figure(figsize=(30, 6))
        plt.hist(all_tokens, bins=50, alpha=0.7, color='blue')
        plt.title('Distribution of Token IDs')
        plt.xlabel('Token ID')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig("token_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Count non-special tokens vs special tokens
        special_token_ids = set([
                                    tokenizer.text_start_token_id,
                                    tokenizer.text_end_token_id,
                                    tokenizer.pcap_start_token_id,
                                    tokenizer.pcap_end_token_id,
                                    tokenizer.pcap_attachment_token_id
                                ] + list(tokenizer.pcap_to_byt5_token_map.values()))

        special_count = sum(1 for t in all_tokens if t in special_token_ids)
        normal_count = len(all_tokens) - special_count

        print(f"Special tokens: {special_count} ({special_count / len(all_tokens) * 100:.1f}%)")
        print(f"Regular tokens: {normal_count} ({normal_count / len(all_tokens) * 100:.1f}%)")

    finally:
        # Clean up the temporary files
        if os.path.exists(mock_pcap_path):
            os.remove(mock_pcap_path)
        try:
            os.rmdir(temp_dir)
        except:
            pass


# For real-world testing with actual PCAP files
def test_with_real_pcap(pcap_file_path, sample_text="Analyzing network traffic:"):
    """Test the tokenizer with a real PCAP file."""
    print(f"\n=== Testing with real PCAP file: {pcap_file_path} ===")

    # Ensure the file exists
    if not os.path.exists(pcap_file_path):
        print(f"Error: PCAP file not found at {pcap_file_path}")
        return

    # Get file size
    file_size = os.path.getsize(pcap_file_path) / (1024 * 1024)  # Size in MB
    print(f"PCAP file size: {file_size:.2f} MB")

    # Test tokenization
    start_time = time.time()
    tokens = tokenizer.tokenize_text_with_pcap(sample_text, pcap_file_path)
    elapsed = time.time() - start_time

    print(f"Tokenization completed in {elapsed:.2f} seconds")
    print(f"Generated {len(tokens)} tokens")

    # Visualize a sample of the tokens (first 200)
    visualize_token_ids(tokens[:200], "Real PCAP Sample (first 200 tokens)",
                        save_path="real_pcap_sample.png")

    # Create a full token distribution chart
    plt.figure(figsize=(30, 6))
    plt.hist(tokens, bins=100, alpha=0.7, color='green')
    plt.title(f'Token Distribution for {os.path.basename(pcap_file_path)}')
    plt.xlabel('Token ID')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig("real_pcap_distribution.png", dpi=300, bbox_inches='tight')

    return tokens


if __name__ == "__main__":
    # Run the test cases
    run_test_cases()

    # Uncomment to test with a real PCAP file
    # real_pcap_path = "/path/to/your/pcap/file.pcap"
    # test_with_real_pcap(real_pcap_path)