import os
import tempfile

import pytest
from scapy.all import Raw, wrpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.layers.dns import DNS, DNSQR

from src.tokenization.hybrid_byt5_tokenizer import HybridByT5PCAPTokenizer

class TestHybridByT5PCAPTokenizer:

    @pytest.fixture
    def tokenizer(self):
        """Create and return a tokenizer instance for testing"""
        return HybridByT5PCAPTokenizer(pcap_vocab_size=277)

    @pytest.fixture
    def sample_pcap_file(self):
        """Create a temporary PCAP file with some test packets"""
        # Create a simple packet
        pkt1 = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=1234, dport=80) / Raw(
            b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        pkt2 = Ether() / IP(src="192.168.1.2", dst="192.168.1.1") / TCP(sport=80, dport=1234) / Raw(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body>Hello</body></html>")
        pkt3 = Ether() / IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=5678, dport=53) / DNS(rd=1, qd=DNSQR(
            qname="example.com"))

        # Create a temporary file and write the packets to it
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
            temp_file_path = tmp.name
            wrpcap(temp_file_path, [pkt1, pkt2, pkt3])

        yield temp_file_path

        # Clean up the temporary file
        os.unlink(temp_file_path)

    @pytest.fixture
    def sample_pcap_bytes(self):
        """Create sample PCAP bytes for testing"""
        pkt = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=1234, dport=80) / Raw(b"TEST")
        return bytes(pkt)

    def test_initialization(self, tokenizer):
        """Test tokenizer initialization and special tokens"""
        # Verify special tokens are correctly registered
        assert tokenizer.pcap_start_token_id is not None
        assert tokenizer.pcap_end_token_id is not None
        assert tokenizer.text_start_token_id is not None
        assert tokenizer.text_end_token_id is not None
        assert tokenizer.pcap_attachment_token_id is not None

        # Verify basic properties
        assert tokenizer.vocab_size == 2 ** 8  # ByT5 uses a byte-level vocabulary
        assert len(tokenizer.pcap_to_byt5_token_map) > 0  # PCAP token mapping exists

    def test_text_tokenization(self, tokenizer):
        """Test text-only tokenization and decoding"""
        text = "This is a test message"
        token_ids = tokenizer.encode_mixed_input(text=text)

        # Verify token structure
        assert token_ids[0] == tokenizer.text_start_token_id
        assert token_ids[-1] == tokenizer.text_end_token_id

        # Verify decoding works
        decoded = tokenizer.decode_mixed_input(token_ids)
        assert 'text' in decoded
        assert decoded['text'] == text

    def test_pcap_file_tokenization(self, tokenizer, sample_pcap_file):
        """Test PCAP file tokenization and decoding"""
        token_ids = tokenizer.tokenize_text_with_pcap("", sample_pcap_file)

        # Verify token structure
        assert token_ids[0] == tokenizer.pcap_attachment_token_id
        assert token_ids[-1] == tokenizer.pcap_end_token_id

        # Verify token count is reasonable
        assert len(token_ids) > 10  # Should have a substantial number of tokens

        # Verify packet start tokens exist
        packet_start_tokens = [tid for tid in token_ids
                               if tid == tokenizer.pcap_to_byt5_token_map.get(
                tokenizer.pcap_tokenizer.special_tokens['packet_start'], None)]
        assert len(packet_start_tokens) >= 3  # We created 3 packets

    def test_pcap_bytes_tokenization(self, tokenizer, sample_pcap_bytes):
        """Test PCAP bytes tokenization and decoding"""
        token_ids = tokenizer.encode_mixed_input(pcap_bytes=sample_pcap_bytes)

        # Verify token structure
        assert token_ids[0] == tokenizer.pcap_start_token_id
        assert token_ids[-1] == tokenizer.pcap_end_token_id

        # Verify token count is reasonable
        assert len(token_ids) > 10  # Should have a substantial number of tokens

    def test_mixed_tokenization(self, tokenizer, sample_pcap_file):
        """Test mixed text and PCAP tokenization"""
        text = "Analyze this traffic"
        token_ids = tokenizer.tokenize_text_with_pcap(text, sample_pcap_file)

        # Verify text tokens exist
        assert tokenizer.text_start_token_id in token_ids
        assert tokenizer.text_end_token_id in token_ids

        # Verify PCAP tokens exist
        assert tokenizer.pcap_attachment_token_id in token_ids
        assert tokenizer.pcap_end_token_id in token_ids

        # Verify decoding works for mixed content
        decoded = tokenizer.decode_mixed_input(token_ids)
        assert 'text' in decoded
        assert decoded['text'] == text
        assert 'pcap_data' in decoded
        assert len(decoded['pcap_data']) > 0

    def test_bidirectional_conversion(self, tokenizer, sample_pcap_bytes):
        """Test bidirectional conversion between tokens and PCAP"""
        # Convert PCAP to tokens
        token_ids = tokenizer.encode_mixed_input(pcap_bytes=sample_pcap_bytes)

        # Convert tokens back to PCAP
        decoded = tokenizer.decode_mixed_input(token_ids)

        # Verify PCAP data was recovered
        assert 'pcap_data' in decoded
        assert len(decoded['pcap_data']) > 0

        # Verify we can reconstruct similar PCAP bytes
        # Note: Exact matching is difficult due to potential differences in representation
        # This test verifies flow integrity by checking packet count
        assert len(decoded['pcap_data']) == 1  # We should have one packet in our flow

    def test_edge_cases(self, tokenizer):
        """Test edge cases like empty inputs"""
        # Empty text
        token_ids_empty_text = tokenizer.encode_mixed_input(text="")
        assert len(token_ids_empty_text) == 2  # Just start and end tokens

        # Very short text
        token_ids_short = tokenizer.encode_mixed_input(text="a")
        assert len(token_ids_short) > 2  # Start, content, and end tokens

        # Very long text (test for potential memory issues)
        long_text = "a" * 1000
        token_ids_long = tokenizer.encode_mixed_input(text=long_text)
        assert len(token_ids_long) > 1000  # Should have many tokens

        # Empty PCAP bytes
        token_ids_empty_pcap = tokenizer.encode_mixed_input(pcap_bytes=b"")
        assert len(token_ids_empty_pcap) == 2  # Just start and end tokens

    def test_error_handling(self, tokenizer):
        """Test error handling for invalid inputs"""
        # Non-existent PCAP file
        with pytest.raises(Exception):
            tokenizer.tokenize_text_with_pcap("test", "non_existent_file.pcap")

        # Invalid PCAP bytes
        with pytest.raises(Exception):
            tokenizer.encode_mixed_input(pcap_bytes=b"not_valid_pcap_data")

    def test_tokenize_pcap_followed_by_text(self, tokenizer, sample_pcap_bytes):
        """Test PCAP bytes followed by text tokenization"""
        text = "This is the analysis"
        token_ids = tokenizer.tokenize_pcap_followed_by_text(sample_pcap_bytes, text)

        # Verify PCAP tokens come before text tokens
        pcap_start_idx = token_ids.index(tokenizer.pcap_start_token_id)
        text_start_idx = token_ids.index(tokenizer.text_start_token_id)
        assert pcap_start_idx < text_start_idx

        # Verify decoding works
        decoded = tokenizer.decode_mixed_input(token_ids)
        assert 'text' in decoded
        assert decoded['text'] == text
        assert 'pcap_data' in decoded