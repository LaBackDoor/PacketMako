import ipaddress
import os
import re
import pyshark


def is_private_ip(ip_str):
    """Check if an IP address is private."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private
    except ValueError:
        return False


def generate_flow_description(packets):
    """
    Generate a high-level description of the entire network flow,
    following the format shown in the example.
    """
    if not packets:
        return "No packets found in the flow."

    # Initialize basic flow information
    protocol = "Unknown"
    src_ip = "None"
    src_port = "None"
    dst_ip = "None"
    dst_port = "None"
    application_layer = "Unknown"
    service = "Unknown"

    # Packet counts
    packets_sent = 0
    packets_received = 0

    # Data size metrics
    data_sent = 0
    data_received = 0

    # Packet sizes for statistics
    packet_sizes_sent = []

    # First pass: extract basic protocol information from the first relevant packet
    for packet in packets:
        if hasattr(packet, 'ip'):
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst

            if hasattr(packet, 'tcp'):
                protocol = "TCP"
                src_port = packet.tcp.srcport
                dst_port = packet.tcp.dstport
                break
            elif hasattr(packet, 'udp'):
                protocol = "UDP"
                src_port = packet.udp.srcport
                dst_port = packet.udp.dstport

                # Check for DNS application layer
                if hasattr(packet, 'dns'):
                    application_layer = "DNS"
                    # Check for query type to determine service
                    if hasattr(packet.dns, 'qry_type'):
                        if packet.dns.qry_type == '1':  # A record
                            service = "domain"
                        elif packet.dns.qry_type == '28':  # AAAA record
                            service = "domain"
                        # Add other DNS query types as needed
                        else:
                            service = f"DNS type {packet.dns.qry_type}"
                break
            elif hasattr(packet, 'icmp'):
                protocol = "ICMP"
                break
            elif hasattr(packet, 'igmp'):
                protocol = "IGMP"
                if hasattr(packet.igmp, 'version'):
                    protocol = f"IGMPv{packet.igmp.version}"
                break

    # Second pass: calculate metrics
    start_time = None
    end_time = None

    for packet in packets:
        # Calculate flow times
        if hasattr(packet, 'frame_info') and hasattr(packet.frame_info, 'time_epoch'):
            timestamp = float(packet.frame_info.time_epoch)
            if start_time is None or timestamp < start_time:
                start_time = timestamp
            if end_time is None or timestamp > end_time:
                end_time = timestamp

        # Determine if packet is sent or received
        if hasattr(packet, 'ip'):
            if packet.ip.src == src_ip:
                packets_sent += 1
                if hasattr(packet, 'length'):
                    data_sent += int(packet.length)
                    packet_sizes_sent.append(int(packet.length))
            else:
                packets_received += 1
                if hasattr(packet, 'length'):
                    data_received += int(packet.length)

    # Calculate statistics for sent packets
    mean_size_sent = 0
    std_sent = 0
    if packet_sizes_sent:
        mean_size_sent = sum(packet_sizes_sent) / len(packet_sizes_sent)
        # Calculate standard deviation
        if len(packet_sizes_sent) > 1:
            variance = sum((x - mean_size_sent) ** 2 for x in packet_sizes_sent) / len(packet_sizes_sent)
            std_sent = variance ** 0.5

    # Determine if source and destination are internal networks
    src_internal = is_private_ip(src_ip)
    dst_internal = is_private_ip(dst_ip)

    # Format timestamps
    start_time_str = str(int(start_time)) if start_time else "Unknown"
    end_time_str = str(int(end_time)) if end_time else "Unknown"

    # Now create a paragraph version similar to the example
    paragraph = f"This network flow description represents a {protocol} flow. "
    paragraph += f"The flow consists of {packets_sent} packets sent from the source IP address {src_ip} "
    paragraph += f"to the destination IP address {dst_ip}. "

    if packets_received > 0:
        paragraph += f"{packets_received} packets were received in this flow. "
    else:
        paragraph += f"No data was received in this flow. "

    paragraph += f"The source port is {src_port} and the destination port is {dst_port}. "

    if application_layer != "Unknown":
        paragraph += f"The application layer is {application_layer}. "
        if service != "Unknown":
            paragraph += f"The service is {service}. "

    paragraph += f"The source and destination hostnames are not provided. "
    paragraph += f"The flow is {'internal' if src_internal and dst_internal else 'not internal'}. "

    if packet_sizes_sent:
        paragraph += f"The mean size of the sent packets is {int(mean_size_sent)} bytes, "
        paragraph += f"with standard deviation {int(std_sent)}. "

    paragraph += f"The flow started at timestamp {start_time_str} and ended at timestamp {end_time_str}."

    return paragraph


def convert_packet_to_paragraph(packet):
    """
    Convert a single packet's fields to a paragraph format.
    Dynamically extracts all available fields without assumptions about structure.
    Skips empty fields and does not interpret packet behavior.
    """
    paragraph = ""

    # Process each layer in the packet
    for layer in packet.layers:
        layer_name = layer.layer_name

        # Get all field names for this layer
        try:
            field_names = layer.field_names
        except AttributeError:
            # Some layers might not have field_names attribute
            continue

        layer_fields = []
        for field in field_names:
            try:
                # Get the field value
                value = layer.get_field_value(field)

                # Skip empty values
                if value:
                    layer_fields.append(f"{layer_name}.{field} is {value}")
            except AttributeError:
                continue

        if layer_fields:
            paragraph += f"The {layer_name} layer contains " + ", ".join(layer_fields) + ". "

    return paragraph


def extract_classification_from_filename(filename):
    """
    Extract attack classification information from the filename.
    Assumes format like: 0_0_0_0_0_224_0_0_1_0_2_attack_DDoS_ICMP_5.pcap
    """
    # Extract attack type, category and subcategory using regex
    match = re.search(r'attack_([^_]+)_([^_\.]+)', filename)
    if match:
        attack_type = match.group(1)  # e.g., DDoS
        category = match.group(2)  # e.g., ICMP_5
        return attack_type, category
    return "Unknown", "Unknown"


def process_pcap_with_flow_summary(pcap_file, output_file=None):
    """
    Process a PCAP file, generating:
    1. A high-level flow description similar to Figure 1 in the research paper
    2. Detailed packet-by-packet descriptions
    Does not interpret or imply anything about behavior.
    """
    print(f"Processing {pcap_file}...")

    # Extract classification from filename
    attack_type, category = extract_classification_from_filename(os.path.basename(pcap_file))
    classification_header = f"Classification - Attack Type - {attack_type} Category - {category}\n\n"

    capture = pyshark.FileCapture(pcap_file)

    # Read all packets into memory
    packets = []
    for packet in capture:
        packets.append(packet)

    # Generate flow-level description
    flow_description = generate_flow_description(packets)
    print("\nFlow Description:")
    print(flow_description)
    print("\n" + "-" * 50 + "\n")

    # Generate packet-by-packet descriptions
    packet_paragraphs = []
    for i, packet in enumerate(packets):
        paragraph = convert_packet_to_paragraph(packet)
        packet_info = f"Packet {i + 1}:\n{paragraph}\n"
        packet_paragraphs.append(packet_info)
        print(packet_info)

    # Write to output file if specified
    if output_file and packet_paragraphs:
        with open(output_file, 'w') as f:
            f.write(classification_header)  # Add classification header at the top
            f.write("FLOW DESCRIPTION:\n")
            f.write(flow_description)
            f.write("\n\n" + "-" * 50 + "\n\n")
            f.write("PACKET DESCRIPTIONS:\n\n")
            f.write("\n".join(packet_paragraphs))
        print(f"\nDescriptions saved to {output_file}")

    capture.close()
    return flow_description, packet_paragraphs


def process_pcap_directory(pcap_dir):
    """
    Process all PCAP files in a directory.
    """
    # Ensure the directory exists
    if not os.path.isdir(pcap_dir):
        print(f"Error: Directory '{pcap_dir}' does not exist.")
        return

    # Get all PCAP files in the directory
    pcap_files = [f for f in os.listdir(pcap_dir) if f.lower().endswith('.pcap')]

    if not pcap_files:
        print(f"No PCAP files found in '{pcap_dir}'.")
        return

    print(f"Found {len(pcap_files)} PCAP files to process.")

    # Process each PCAP file
    for pcap_file in pcap_files:
        pcap_path = os.path.join(pcap_dir, pcap_file)

        # Generate output filename - same as input but with .txt extension
        output_filename = os.path.splitext(pcap_file)[0] + "_explanation.txt"
        output_path = os.path.join(pcap_dir, output_filename)

        # Process the file
        try:
            process_pcap_with_flow_summary(pcap_path, output_path)
        except Exception as e:
            print(f"Error processing {pcap_file}: {str(e)}")


def process_pcap_directory_with_output(pcap_dir, output_directory=None):
    """
    Process all PCAP files in a directory and save output to specified directory.
    If output_dir is None, output files will be saved in the same directory as the PCAP files.
    """
    # Ensure the input directory exists
    if not os.path.isdir(pcap_dir):
        print(f"Error: Directory '{pcap_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Get all PCAP files in the directory
    pcap_files = [f for f in os.listdir(pcap_dir) if f.lower().endswith('.pcap')]

    if not pcap_files:
        print(f"No PCAP files found in '{pcap_dir}'.")
        return

    print(f"Found {len(pcap_files)} PCAP files to process.")

    # Process each PCAP file
    for pcap_file in pcap_files:
        pcap_path = os.path.join(pcap_dir, pcap_file)

        # Generate output filename
        output_filename = os.path.splitext(pcap_file)[0] + "_explanation.txt"

        if output_directory:
            output_path = os.path.join(output_directory, output_filename)
        else:
            output_path = os.path.join(pcap_dir, output_filename)

        # Process the file
        try:
            process_pcap_with_flow_summary(pcap_path, output_path)
        except Exception as e:
            print(f"Error processing {pcap_file}: {str(e)}")


if __name__ == "__main__":
    # Hardcoded parameters - modify these as needed
    input_dir = "../pcap_test/"
    output_dir = "../explanations_dir/"

    # Process all PCAP files in the directory
    process_pcap_directory_with_output(input_dir, output_dir)