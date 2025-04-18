import os
import tempfile
from collections import defaultdict

import pytest
from scapy.all import Raw, wrpcap, rdpcap
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether

from src.tokenization.pcap_tokenizer import PCAPTokenizer


class TestPCAPTokenizer:

    @pytest.fixture
    def tokenizer(self):
        """Create a standard tokenizer for testing"""
        return PCAPTokenizer(vocab_size=277, offset=3)

    @pytest.fixture
    def sample_pcap_file(self):
        """Create a temporary PCAP file with test traffic"""
        # Create packets representing different protocols and flow types
        pkt1 = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=1234, dport=80) / Raw(
            b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        pkt2 = Ether() / IP(src="192.168.1.2", dst="192.168.1.1") / TCP(sport=80, dport=1234) / Raw(
            b"HTTP/1.1 200 OK\r\n")
        pkt3 = Ether() / IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=5678, dport=53) / DNS(rd=1, qd=DNSQR(
            qname="example.com"))

        # Create a temporary file for our PCAP
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
            temp_file_path = tmp.name

        # Write the packets to the file
        wrpcap(temp_file_path, [pkt1, pkt2, pkt3])

        yield temp_file_path

        # Clean up after the test
        try:
            os.unlink(temp_file_path)
        except:
            pass

    def test_initialization(self, tokenizer):
        """Test that the tokenizer initializes correctly with proper values"""
        assert tokenizer.vocab_size == 277
        assert tokenizer.offset == 3
        assert 'packet_start' in tokenizer.special_tokens
        assert 'end' in tokenizer.special_tokens
        assert len(tokenizer.hex_to_token) == 256
        assert len(tokenizer.link_types) > 0
        assert isinstance(tokenizer.flows, defaultdict)

    def test_allocate_token(self, tokenizer):
        """Test token allocation mechanism"""
        # Test normal allocation
        initial_allocation_count = len(tokenizer.allocated_tokens)
        new_token = tokenizer._allocate_token()

        # Verify token was allocated
        assert new_token not in tokenizer.allocated_tokens - {new_token}
        assert len(tokenizer.allocated_tokens) == initial_allocation_count + 1

        # Test allocation limit
        # Create a new tokenizer with a reasonable vocab size
        test_tokenizer = PCAPTokenizer(vocab_size=277, offset=3)

        # Keep allocating tokens until we hit the limit
        try:
            # Allocate tokens until we hit the limit
            while True:
                test_tokenizer._allocate_token()
        except ValueError:
            pass

        # Now verify that trying to allocate one more token raises ValueError
        with pytest.raises(ValueError):
            test_tokenizer._allocate_token()

    def test_tokenize_pcap(self, tokenizer, sample_pcap_file):
        """Test the tokenization of a PCAP file into flows"""
        tokenized_flows = tokenizer.tokenize_pcap(sample_pcap_file)

        # Verify we got flows
        assert len(tokenized_flows) > 0

        # Verify token structure in each flow
        for flow_id, tokens in tokenized_flows.items():
            # Flow should begin with a packet start token
            assert tokenizer.special_tokens['packet_start'] in tokens
            # Flow should end with an end token
            assert tokens[-1] == tokenizer.special_tokens['end']

            # Count packet start tokens to verify packet count
            packet_starts = tokens.count(tokenizer.special_tokens['packet_start'])
            assert packet_starts > 0

    def test_encode_time_interval(self, tokenizer):
        """Test encoding of time intervals to tokens"""
        # Test with zero interval
        tokens = tokenizer._encode_time_interval(0.0)
        assert len(tokens) == 8  # IEEE 754 double precision takes 8 bytes

        # Test with non-zero interval
        tokens = tokenizer._encode_time_interval(0.5)
        assert len(tokens) == 8

        # Test with a large interval
        tokens = tokenizer._encode_time_interval(3600.0)  # 1 hour
        assert len(tokens) == 8

    def test_encode_packet_data(self, tokenizer):
        """Test encoding of packet data to hex tokens"""
        data = b"HTTP/1.1"
        tokens = tokenizer._encode_packet_data(data)

        # Verify token count matches byte count
        assert len(tokens) == len(data)

        # Verify token values
        for i, byte_val in enumerate(data):
            assert tokens[i] == tokenizer.hex_to_token[byte_val]

    def test_get_link_type_token(self, tokenizer):
        """Test link type token determination"""
        # Test with Ethernet packet
        eth_pkt = Ether()
        link_token = tokenizer._get_link_type_token(eth_pkt)
        assert link_token == tokenizer.link_types[1]  # 1 is Ethernet

        # Test with unknown link type
        class CustomPacket:
            def __init__(self):
                self.linktype = 999  # Custom link type

            def __contains__(self, item):
                # Return False to simulate that this packet doesn't contain Ether layer
                return False

        custom_pkt = CustomPacket()
        link_token = tokenizer._get_link_type_token(custom_pkt)

        # After allocation, the link type should be in the mapping
        assert 999 in tokenizer.link_types

        # The token returned should match the allocated token
        assert link_token == tokenizer.link_types[999]

    def test_tokenize_flow(self, tokenizer, sample_pcap_file):
        """Test tokenizing a flow of packets"""
        # First get some flows from a PCAP
        tokenizer.tokenize_pcap(sample_pcap_file)
        assert len(tokenizer.flows) > 0

        # Get the first flow
        flow_id = next(iter(tokenizer.flows))
        flow_packets = tokenizer.flows[flow_id]

        # Tokenize the flow
        tokens = tokenizer._tokenize_flow(flow_packets)

        # Check token structure
        assert tokens[0] == tokenizer.special_tokens['packet_start']
        assert tokens[-1] == tokenizer.special_tokens['end']

        # Count packet start tokens
        packet_starts = tokens.count(tokenizer.special_tokens['packet_start'])
        assert packet_starts == len(flow_packets)

    def test_decode_flow(self, tokenizer):
        """Test decoding a token list back to a flow of packets"""
        # Create a simple token sequence
        tokens = [
            tokenizer.special_tokens['packet_start'],
            tokenizer.link_types[1],  # Ethernet
        ]

        # Add time tokens (8 bytes)
        tokens.extend(tokenizer._encode_time_interval(0.0))

        # Add some data tokens
        test_data = b"TESTDATA"
        tokens.extend(tokenizer._encode_packet_data(test_data))

        # Add end token
        tokens.append(tokenizer.special_tokens['end'])

        # Decode the flow
        decoded_packets = tokenizer.decode_flow(tokens)

        # Verify we got a packet
        assert len(decoded_packets) == 1
        link_type, time_interval, packet_bytes = decoded_packets[0]

        # Verify the packet has the expected values
        assert link_type == 1  # Ethernet
        assert time_interval == 0.0
        assert isinstance(packet_bytes, bytes)

    def test_flows_to_pcap(self, tokenizer, sample_pcap_file, tmp_path):
        """Test reconstructing a PCAP file from tokenized flows"""
        # First tokenize a PCAP file
        tokenized_flows = tokenizer.tokenize_pcap(sample_pcap_file)

        # Create an output file path
        output_path = os.path.join(tmp_path, "output.pcap")

        # Reconstruct the PCAP
        packet_count = tokenizer.flows_to_pcap(tokenized_flows, output_path)

        # Verify the file was created
        assert os.path.exists(output_path)
        assert packet_count > 0

        # Verify the file has content
        packets = rdpcap(output_path)
        assert len(packets) > 0

    def test_error_handling(self, tokenizer, tmp_path):
        """Test error handling for various scenarios"""
        # Test with non-existent file
        nonexistent_file = os.path.join(tmp_path, "nonexistent.pcap")
        result = tokenizer.tokenize_pcap(nonexistent_file)
        assert result == {}  # Should return empty dict for missing file

        # Test with overly small vocabulary
        with pytest.raises(ValueError):
            PCAPTokenizer(vocab_size=100)  # Too small for all tokens needed

        # Test decode_flow with empty token list
        result = tokenizer.decode_flow([])
        assert result == []  # Should handle gracefully