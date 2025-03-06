#!/usr/bin/env python3
"""
This script processes PCAP files and segments them into flows.
Resume functionality is added using a checkpoint file. Each PCAP file that is fully processed
is recorded in 'processed_pcaps.txt'. On subsequent runs, the script will skip already processed PCAPs.
"""

import os
import re
from scapy.all import PcapReader, PcapWriter
from scapy.layers.inet import IP, TCP, UDP

# Checkpoint file for resume functionality
CHECKPOINT_FILE = "processed_pcaps.txt"

def sanitize_filename(name):
    """
    Replace any character that is not alphanumeric, dash, or underscore with an underscore.
    This ensures the generated filenames are safe.
    """
    return re.sub(r'[^\w\-]', '_', name)

def parse_labels_from_path(pcap_path, root_dir):
    """
    Infer (label, category, type) based on subdirectory structure.
    Adjust this logic to match your naming conventions.

    Example structure:
      PCAP/
        NORMAL/
          normal.pcap
        ATTACK/
          DDoS/
            ICMP/
              somefile.pcap
          Injection/
            SQL/
              ...
    """
    rel_path = os.path.relpath(pcap_path, start=root_dir)
    parts = rel_path.split(os.path.sep)

    # Example logic based on MU-IoT structure
    if len(parts) >= 2 and parts[0].upper() == "NORMAL":
        label = "normal"
        category = "normal"
        rtype = "normal"
    elif len(parts) >= 3 and parts[0].upper() == "ATTACK":
        label = "attack"
        category = parts[1]
        rtype = parts[2]
    else:
        label = "unknown"
        category = "unknown"
        rtype = "unknown"

    return label, category, rtype

def process_pcap_file(pcap_path, label, category, rtype, output_dir, tint=1.0):
    """
    Read packets from a PCAP, segment them into flows by time, and write each flow to its own PCAP.
    A 'flow' is defined by:
      - (SIP, SPort) and (DIP, DPort) sorted for bidirectionality
      - IP proto (TCP=6, UDP=17, ICMP=1, etc.)
      - label, category, type
      - time gap <= tint to stay in the same flow

    If the script crashes, processing for this PCAP will start over the next time.
    """
    print(f"  Opening {pcap_path} ...")
    flow_info = {}   # base_key -> (last_timestamp, flow_count)
    flow_writers = {}  # full_key -> PcapWriter object
    packet_count = 0

    # Use streaming PcapReader
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            packet_count += 1
            if packet_count % 100000 == 0:
                print(f"    Processed {packet_count} packets from {os.path.basename(pcap_path)}")

            if IP not in pkt:
                continue  # Non-IP packets are ignored

            timestamp = float(pkt.time)
            sip = pkt[IP].src
            dip = pkt[IP].dst
            proto_num = pkt[IP].proto

            # Default ports
            sport = 0
            dport = 0

            # If TCP or UDP, extract ports
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport

            # Sort endpoints for bidirectionality
            endpoint1 = (sip, sport)
            endpoint2 = (dip, dport)
            sorted_endpoints = sorted([endpoint1, endpoint2])

            # Base flow key ignoring time
            base_key = (sorted_endpoints[0], sorted_endpoints[1],
                        proto_num, label, category, rtype)

            # Check flow segmentation
            if base_key in flow_info:
                last_ts, flow_count = flow_info[base_key]
                if timestamp - last_ts <= tint:
                    new_flow_count = flow_count
                else:
                    new_flow_count = flow_count + 1
                flow_info[base_key] = (timestamp, new_flow_count)
            else:
                flow_info[base_key] = (timestamp, 1)
                new_flow_count = 1

            # Full key (with flow counter)
            full_key = base_key + (new_flow_count,)

            # If we don't have a PcapWriter for this flow yet, create one
            if full_key not in flow_writers:
                # Construct output PCAP filename
                # e.g., "192_168_1_10_443_192_168_1_20_80_6_attack_DDoS_ICMP_1.pcap"
                filename_key = (
                    f"{sorted_endpoints[0][0]}_{sorted_endpoints[0][1]}_"
                    f"{sorted_endpoints[1][0]}_{sorted_endpoints[1][1]}_"
                    f"{proto_num}_{label}_{category}_{rtype}_{new_flow_count}"
                )
                safe_filename = sanitize_filename(filename_key) + ".pcap"
                out_path = os.path.join(output_dir, safe_filename)

                # Create a new PcapWriter
                flow_writers[full_key] = PcapWriter(out_path, append=False)

            # Write this packet to the correct flow PCAP
            flow_writers[full_key].write(pkt)

    # Close all PcapWriter objects
    for writer in flow_writers.values():
        writer.close()

    print(f"  Done processing {pcap_path}. Total packets: {packet_count}")

def main():
    # Root directory with subfolders: NORMAL, ATTACK/<Category>/<Type>
    pcap_root = "../../../../Downloads/MU-IOT/PCAP"
    output_dir = "../../../Packet Analysis Data/data_streams/pcap_streams"
    tint = 1.0  # 1 second flow gap

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint file for resume functionality
    processed_pcaps = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as cp:
            processed_pcaps = {line.strip() for line in cp if line.strip()}

    # Walk the pcap_root directory
    for root, dirs, files in os.walk(pcap_root):
        for file in files:
            if file.endswith(".pcap"):
                pcap_path = os.path.join(root, file)
                # Skip PCAPs that have already been processed
                if pcap_path in processed_pcaps:
                    print(f"Skipping already processed pcap: {pcap_path}")
                    continue

                # Parse labels from path
                label, category, rtype = parse_labels_from_path(pcap_path, pcap_root)

                print(f"Processing pcap: {pcap_path}")
                try:
                    process_pcap_file(
                        pcap_path=pcap_path,
                        label=label,
                        category=category,
                        rtype=rtype,
                        output_dir=output_dir,
                        tint=tint
                    )
                    # Mark this PCAP as processed
                    with open(CHECKPOINT_FILE, "a") as cp:
                        cp.write(pcap_path + "\n")
                except Exception as e:
                    print(f"Error processing {pcap_path}: {e}")
                    # Do not mark the file as processed so it can be retried
                    continue

    print("All PCAPs processed.")

if __name__ == "__main__":
    main()
