from scapy.all import rdpcap
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.contrib.igmp import IGMP
import struct
from collections import defaultdict
import logging


class PCAPTokenizer:
    def __init__(self, vocab_size=260):
        """
        Initialize the PCAP tokenizer based on TrafficGPT approach.

        Args:
            vocab_size: Size of the token vocabulary (default 260 as per paper)
        """
        self.vocab_size = vocab_size

        # Reserve first 256 tokens for byte values (0x00-0xFF)
        self.special_tokens = {
            'packet_start': 0x100,  # 256
            'end': 0x101,  # 257
        }

        # Verify we have enough space for special tokens
        if vocab_size < 258:  # 256 hex values + at least 2 special tokens
            raise ValueError(f"Vocab size {vocab_size} is too small. Minimum is 258.")

        # Create a mapping for hex values to token IDs
        self.hex_to_token = {i: i for i in range(256)}

        # Track allocated tokens
        self.allocated_tokens = set(range(256))  # Hex values
        self.allocated_tokens.update(self.special_tokens.values())

        # Link type token mapping
        self.link_types = {
            0: self._allocate_token(),  # Loopback (DLT_NULL)
            1: self._allocate_token(),  # Ethernet (DLT_EN10MB)
            8: self._allocate_token(),  # SLIP
            9: self._allocate_token(),  # PPP
            10: self._allocate_token(),  # FDDI
            101: self._allocate_token(),  # Raw IP
            105: self._allocate_token(),  # IEEE 802.11 (Wireless LAN)
            113: self._allocate_token(),  # Linux cooked capture
            127: self._allocate_token(),  # IEEE 802.11 with radiotap header
        }

        # Initialize flow data structure
        self.flows = defaultdict(list)

        # Logging
        self.logger = logging.getLogger('PCAPTokenizer')

    def _allocate_token(self):
        """
        Allocate a new token ID from the available pool.

        Returns:
            A new token ID

        Raises:
            ValueError: If the vocabulary is full
        """
        # Find first available token ID
        for token_id in range(self.vocab_size):
            if token_id not in self.allocated_tokens:
                self.allocated_tokens.add(token_id)
                return token_id

        raise ValueError(f"Token vocabulary limit of {self.vocab_size} exceeded")

    def tokenize_pcap(self, pcap_file):
        """
        Tokenize a PCAP file into flows of tokens.

        Args:
            pcap_file: Path to the PCAP file

        Returns:
            Dictionary mapping flow identifiers to token lists
        """
        # Read the PCAP file
        try:
            packets = rdpcap(pcap_file)
        except Exception as e:
            self.logger.error(f"Error reading PCAP file: {e}")
            return {}

        # Reset flows for new tokenization
        self.flows = defaultdict(list)

        # Extract flows
        self._extract_flows(packets)

        # Tokenize each flow
        tokenized_flows = {}
        for flow_id, flow_packets in self.flows.items():
            tokenized_flows[flow_id] = self._tokenize_flow(flow_packets)

        return tokenized_flows

    def _extract_flows(self, packets):
        """
        Extract flows from a list of packets.
        A flow is identified by (src_ip, dst_ip, src_port, dst_port, protocol)
        or by the reverse tuple for bidirectional flows.
        """
        for packet in packets:
            # Check if packet has IP layer
            if IP in packet:
                ip_layer = packet[IP]
                protocol = ip_layer.proto

                # For TCP and UDP, create flow identifier using 5-tuple
                if TCP in packet or UDP in packet:
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst

                    if TCP in packet:
                        transport_layer = packet[TCP]
                    else:
                        transport_layer = packet[UDP]

                    src_port = transport_layer.sport
                    dst_port = transport_layer.dport

                    # Create forward and reverse flow IDs
                    forward_id = (src_ip, dst_ip, src_port, dst_port, protocol)
                    reverse_id = (dst_ip, src_ip, dst_port, src_port, protocol)

                    # Check if reverse flow exists
                    if reverse_id in self.flows:
                        flow_id = reverse_id
                    else:
                        flow_id = forward_id

                    # Store packet in its flow with timestamp
                    self.flows[flow_id].append((float(packet.time), packet))
                    # Handling ICMP packets

                elif ICMP in packet:
                    icmp_layer = packet[ICMP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    icmp_type = icmp_layer.type
                    icmp_code = icmp_layer.code

                    # Use a tuple that includes ICMP-specific fields
                    flow_id = (src_ip, dst_ip, icmp_type, icmp_code, protocol)
                    self.flows[flow_id].append((float(packet.time), packet))

                # Handling IGMP packets
                elif IGMP in packet:
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst

                    # For IGMP, you might simply use the IP addresses and protocol number
                    flow_id = (src_ip, dst_ip, protocol)
                    self.flows[flow_id].append((float(packet.time), packet))

                else:
                    # Skip non-TCP/UDP/ICMP/IGMP packets as per paper approach
                    continue

        # Sort packets in each flow by timestamp
        for flow_id in self.flows:
            self.flows[flow_id].sort(key=lambda x: x[0])
            # Extract just the packets, removing timestamps used for sorting
            self.flows[flow_id] = [packet for _, packet in self.flows[flow_id]]

    def _tokenize_flow(self, packets):
        """
        Tokenize a flow of packets.

        Args:
            packets: List of packets in the flow

        Returns:
            List of tokens representing the flow
        """
        tokens = []
        prev_time = None

        for packet in packets:
            # 1. Add Packet Start Token
            tokens.append(self.special_tokens['packet_start'])

            # 2. Add Link Type Token
            link_type_token = self._get_link_type_token(packet)
            tokens.append(link_type_token)

            # 3. Add Time Interval Tokens
            curr_time = float(packet.time)
            if prev_time is None:
                time_interval = 0
            else:
                time_interval = curr_time - prev_time

            time_tokens = self._encode_time_interval(time_interval)
            tokens.extend(time_tokens)

            prev_time = curr_time

            # 4. Add Hex Tokens for packet data
            raw_data = bytes(packet)
            hex_tokens = self._encode_packet_data(raw_data)
            tokens.extend(hex_tokens)

        # Add End Token
        tokens.append(self.special_tokens['end'])

        return tokens

    def _get_link_type_token(self, packet):
        """
        Determine the link type token of a packet.

        Args:
            packet: A scapy packet

        Returns:
            Link type token
        """
        # Try to get the link type from different packet attributes
        link_type = None

        # Get link type from Scapy packet
        if Ether in packet:
            link_type = 1  # Ethernet
        elif hasattr(packet, 'linktype'):
            link_type = packet.linktype

        # Return token for the link type
        if link_type in self.link_types:
            return self.link_types[link_type]

        # If unknown link type, add it to the map
        if link_type is not None and link_type not in self.link_types:
            try:
                self.link_types[link_type] = self._allocate_token()
                return self.link_types[link_type]
            except ValueError:
                # Vocabulary limit exceeded, use default
                self.logger.warning(f"Vocabulary limit exceeded for link type {link_type}, using default")
                return self.link_types.get(1, 0)

        # Default to Ethernet
        return self.link_types.get(1, 0)

    def _encode_time_interval(self, time_interval):
        """
        Convert a time interval to 8 byte tokens.

        Args:
            time_interval: Time interval in seconds

        Returns:
            List of 8 tokens representing the time interval
        """
        # Convert to exponential form as mentioned in the paper
        # Using IEEE 754 double precision format (8 bytes)
        packed = struct.pack('!d', time_interval)

        # Convert each byte to a token
        time_tokens = [self.hex_to_token[byte] for byte in packed]

        return time_tokens

    def _encode_packet_data(self, raw_data):
        """
        Convert packet data to hex tokens.

        Args:
            raw_data: Raw bytes of the packet

        Returns:
            List of tokens representing the packet data
        """
        # Convert each byte to a token
        hex_tokens = [self.hex_to_token[byte] for byte in raw_data]

        return hex_tokens

    def decode_flow(self, tokens):
        """
        Decode a list of tokens back to a flow of packets.

        Args:
            tokens: List of tokens representing a flow

        Returns:
            List of reconstructed packets
        """
        # Create reverse mappings
        token_to_link_type = {v: k for k, v in self.link_types.items()}

        packets = []
        i = 0

        while i < len(tokens):
            if tokens[i] == self.special_tokens['packet_start']:
                # Found packet start
                i += 1

                if i >= len(tokens):
                    self.logger.warning("Unexpected end of tokens after packet start")
                    break

                # Extract link type
                link_type_token = tokens[i]
                link_type = token_to_link_type.get(link_type_token, 1)  # Default to Ethernet
                i += 1

                if i + 8 > len(tokens):
                    self.logger.warning("Not enough tokens for time interval")
                    break

                # Extract time interval
                time_tokens = tokens[i:i + 8]
                time_bytes = bytes([t if t < 256 else 0 for t in time_tokens])
                time_interval = struct.unpack('!d', time_bytes)[0]
                i += 8

                # Collect packet data until next packet start or end
                packet_data = []
                while (i < len(tokens) and
                       tokens[i] != self.special_tokens['packet_start'] and
                       tokens[i] != self.special_tokens['end']):
                    # Map token to byte value, for non-hex tokens use 0
                    byte_value = tokens[i] if tokens[i] < 256 else 0
                    packet_data.append(byte_value)
                    i += 1

                # Convert to bytes
                packet_bytes = bytes(packet_data)

                # Store reconstructed packet info
                packets.append((link_type, time_interval, packet_bytes))
            elif tokens[i] == self.special_tokens['end']:
                # End of flow
                break
            else:
                # Unexpected token
                self.logger.warning(f"Unexpected token {tokens[i]} at position {i}")
                i += 1

        return packets

    def flows_to_pcap(self, tokenized_flows, output_file):
        """
        Reconstruct a PCAP file from tokenized flows.

        Args:
            tokenized_flows: Dictionary mapping flow IDs to token lists
            output_file: Path to save the reconstructed PCAP

        Returns:
            Number of packets written to the output file
        """
        from scapy.utils import wrpcap

        # Decode flows back to packets
        all_packets = []

        for flow_id, tokens in tokenized_flows.items():
            decoded_packets = self.decode_flow(tokens)

            for link_type, time_interval, packet_bytes in decoded_packets:
                try:
                    # Parse packet from bytes
                    packet = Ether(packet_bytes)
                    all_packets.append(packet)
                except Exception as e:
                    self.logger.warning(f"Failed to reconstruct packet: {e}")

        # Write packets to PCAP
        if all_packets:
            wrpcap(output_file, all_packets)

        return len(all_packets)