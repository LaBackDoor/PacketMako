def main():
    """manual test script"""
    from src.model.hybridbyt5 import HybridByT5PCAPTokenizer
    from scapy.all import Ether, IP, TCP, Raw, wrpcap
    import tempfile

    # Create a tokenizer
    tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=277)

    # Create a simple PCAP file
    pkt = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=1234, dport=80) / Raw(b"GET / HTTP/1.1\r\n")

    with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
        pcap_file = tmp.name
        wrpcap(pcap_file, [pkt])

    # Test text tokenization
    text_tokens = tokenizer.encode_mixed_input(text="Query: What protocol is used?")
    print(f"Text tokens: {text_tokens[:10]}...{text_tokens[-10:]} (length: {len(text_tokens)})")

    # Test PCAP tokenization
    pcap_tokens = tokenizer.tokenize_text_with_pcap("", pcap_file)
    print(f"PCAP tokens: {pcap_tokens[:10]}...{pcap_tokens[-10:]} (length: {len(pcap_tokens)})")

    # Test mixed tokenization
    mixed_tokens = tokenizer.tokenize_text_with_pcap("Query: What protocol is used?", pcap_file)
    print(f"Mixed tokens: {mixed_tokens[:10]}...{mixed_tokens[-10:]} (length: {len(mixed_tokens)})")

    # Test decoding
    decoded = tokenizer.decode_mixed_input(mixed_tokens)
    print("Decoded content:")
    print(f"- Text: {decoded.get('text')}")
    print(f"- PCAP data: {decoded.get('pcap_data')}")

    # Clean up
    import os
    os.unlink(pcap_file)


if __name__ == "__main__":
    main()