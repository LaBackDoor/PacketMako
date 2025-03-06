#!/usr/bin/env python3
"""
This script creates a balanced dataset from processed PCAP stream files.
It identifies all attack categories and types from the filenames of the processed streams,
then selects approximately 5 examples of each type to create a balanced dataset.
All files are placed in a flat structure at the same directory level.
"""

import os
import random
import shutil
from collections import defaultdict

# Configuration
PCAP_STREAMS_DIR = "../../../Packet Analysis Data/data_streams/pcap_streams"
OUTPUT_DIR = "../../../Packet Analysis Data/balanced_dataset"
EXAMPLES_PER_TYPE = 5


def parse_labels_from_filename(filename):
    """
    Extract (label, category, type) from the filename of a processed PCAP stream.

    Example filename format:
    "192_168_1_10_443_192_168_1_20_80_6_attack_DDoS_ICMP_1.pcap"
    """
    # Remove .pcap extension
    basename = filename.replace('.pcap', '')

    # Split by underscore
    parts = basename.split('_')

    # Processed PCAP streams should have the label, category, and type near the end
    if len(parts) >= 14:  # Basic check that we have enough parts
        # Format is typically: [IPs]_[ports]_[proto]_label_category_type_[flow#]
        label_index = -4
        category_index = -3
        type_index = -2

        label = parts[label_index]
        category = parts[category_index]
        rtype = parts[type_index]

        return label, category, rtype

    # If format doesn't match, return unknown
    return "unknown", "unknown", "unknown"


def analyze_pcap_streams():
    """
    Analyze the processed PCAP stream files to identify all categories and types.
    Returns a dictionary of all categories and their types, and a dictionary mapping
    each (category, type) to a list of corresponding stream files.
    """
    categories_and_types = defaultdict(set)
    pcap_files_by_type = defaultdict(list)

    # Check if the stream directory exists
    if not os.path.exists(PCAP_STREAMS_DIR):
        print(f"Error: PCAP streams directory not found: {PCAP_STREAMS_DIR}")
        return categories_and_types, pcap_files_by_type

    # List all files in the streams directory
    for filename in os.listdir(PCAP_STREAMS_DIR):
        if filename.endswith(".pcap"):
            pcap_path = os.path.join(PCAP_STREAMS_DIR, filename)

            # Parse labels from filename
            label, category, rtype = parse_labels_from_filename(filename)

            # Skip unknown categories
            if category == "unknown" or rtype == "unknown":
                continue

            # Store category and type
            categories_and_types[category].add(rtype)

            # Store PCAP file path
            pcap_files_by_type[(category, rtype)].append(pcap_path)

    return categories_and_types, pcap_files_by_type


def clear_output_directory():
    """
    Clear the output directory if it exists.
    This ensures a fresh dataset each time the script is run.
    """
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Creating fresh output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_balanced_dataset(categories_and_types, pcap_files_by_type):
    """
    Create a balanced dataset by selecting approximately EXAMPLES_PER_TYPE examples
    of each attack type. All files are placed in the same output directory without subdirectories.
    """
    # Select and copy files
    selected_files = []
    for (category, rtype), pcap_files in pcap_files_by_type.items():
        print(f"Processing {category}/{rtype}: {len(pcap_files)} files available")

        # Determine how many files to select
        num_to_select = min(EXAMPLES_PER_TYPE, len(pcap_files))

        # Randomly select files
        selected = random.sample(pcap_files, num_to_select)
        selected_files.extend(selected)

        # Copy selected files to the output directory
        for pcap_path in selected:
            # Get the original filename
            filename = os.path.basename(pcap_path)

            # Create destination path (all at same level)
            dest_path = os.path.join(OUTPUT_DIR, filename)

            # Copy the file
            shutil.copy2(pcap_path, dest_path)
            print(f"  Selected: {filename}")

    return selected_files


def generate_report(categories_and_types, pcap_files_by_type, selected_files):
    """
    Generate a report of the balanced dataset.
    """
    report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")

    with open(report_path, 'w') as f:
        f.write("Balanced Dataset Report\n")
        f.write("======================\n\n")

        f.write("Categories and Types:\n")
        total_by_category = defaultdict(int)

        for category, types in sorted(categories_and_types.items()):
            f.write(f"  {category}:\n")
            category_count = 0

            for rtype in sorted(types):
                total_files = len(pcap_files_by_type[(category, rtype)])
                selected = sum(1 for path in selected_files if os.path.basename(path) in
                               [os.path.basename(p) for p in pcap_files_by_type[(category, rtype)]])
                f.write(f"    - {rtype}: {selected} selected out of {total_files} available\n")
                category_count += selected

            total_by_category[category] = category_count

        f.write("\nSummary by Category:\n")
        for category, count in sorted(total_by_category.items()):
            f.write(f"  {category}: {count} files\n")

        f.write("\nTotal Files in Balanced Dataset: {}\n".format(len(selected_files)))

    print(f"Report generated at {report_path}")


def list_categories_and_types(categories_and_types):
    """
    Print a list of all categories and types found in the PCAP streams.
    """
    if not categories_and_types:
        print("No categories and types found. Please check the PCAP streams directory.")
        return

    print("Found the following attack categories and types:")
    for category, types in sorted(categories_and_types.items()):
        print(f"  {category}:")
        for rtype in sorted(types):
            print(f"    - {rtype}")


def main():
    print(f"Analyzing PCAP streams directory: {PCAP_STREAMS_DIR}")

    # Clear the output directory
    clear_output_directory()

    # Analyze the processed PCAP streams
    categories_and_types, pcap_files_by_type = analyze_pcap_streams()

    # Check if we found any categories and types
    if not categories_and_types:
        print("No valid PCAP streams found. Please check the directory path.")
        return

    # List categories and types
    list_categories_and_types(categories_and_types)

    # Create balanced dataset
    print(f"\nCreating balanced dataset with ~{EXAMPLES_PER_TYPE} examples per type...")
    selected_files = create_balanced_dataset(categories_and_types, pcap_files_by_type)

    # Generate report
    generate_report(categories_and_types, pcap_files_by_type, selected_files)

    print(f"\nBalanced dataset created at: {OUTPUT_DIR}")
    print(f"Selected {len(selected_files)} files in total.")


if __name__ == "__main__":
    main()