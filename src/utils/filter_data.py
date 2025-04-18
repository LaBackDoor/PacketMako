import os
import json
import time
import sqlite3
import multiprocessing as mp
import subprocess
import argparse
from scapy.all import PcapReader
from scapy.layers.inet import TCP, UDP, ICMP

# Hard-coded variables
INPUT_DIR = "../../../Packet Analysis Data/data_streams/pcap_streams"
DB_FILE = "processed_files.db"

# Shared counters for reporting
kept_count = mp.Value('i', 0)
deleted_count = mp.Value('i', 0)
error_count = mp.Value('i', 0)


def setup_database(db_file):
    """Set up SQLite database for tracking processed files"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS processed_files
                   (
                       file_path
                       TEXT
                       PRIMARY
                       KEY,
                       status
                       TEXT,
                       protocol_counts
                       TEXT,
                       processed_time
                       TIMESTAMP
                   )
                   ''')
    conn.commit()
    return conn


def load_processed_files(conn):
    """Load the set of already processed files from database"""
    cursor = conn.cursor()
    cursor.execute('SELECT file_path FROM processed_files')
    return set(row[0] for row in cursor.fetchall())


def update_log_batch(conn, results):
    """Update the database with a batch of processed files"""
    if not results:
        return

    cursor = conn.cursor()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    data = []
    for result in results:
        if len(result) >= 2:
            status, file_path = result[0], result[1]
            protocol_counts = ""

            if len(result) >= 3:
                protocol_counts = result[2]

            data.append((file_path, status, protocol_counts, timestamp))

    if data:
        cursor.executemany(
            'INSERT OR REPLACE INTO processed_files VALUES (?, ?, ?, ?)',
            data
        )
        conn.commit()


def fast_packet_analysis(file_path):
    """Use tshark for faster packet counting"""
    try:
        # First check total packets with a single command
        total_cmd = f"tshark -r '{file_path}' -c 14 2>/dev/null | wc -l"
        total_count = int(subprocess.check_output(total_cmd, shell=True).decode().strip())

        # Early termination if total exceeds all thresholds
        if total_count > 13:
            return 0, 0, 0, "delete", "Exceeds all thresholds"

        # Count packets by protocol
        tcp_cmd = f"tshark -r '{file_path}' -Y 'tcp' 2>/dev/null | wc -l"
        tcp_count = int(subprocess.check_output(tcp_cmd, shell=True).decode().strip())

        udp_cmd = f"tshark -r '{file_path}' -Y 'udp' 2>/dev/null | wc -l"
        udp_count = int(subprocess.check_output(udp_cmd, shell=True).decode().strip())

        icmp_cmd = f"tshark -r '{file_path}' -Y 'icmp' 2>/dev/null | wc -l"
        icmp_count = int(subprocess.check_output(icmp_cmd, shell=True).decode().strip())

        # Check if file meets requirements
        if ((0 < tcp_count <= 10) or (0 < udp_count <= 13) or (0 < icmp_count <= 10)):
            return tcp_count, udp_count, icmp_count, "keep", "Meets criteria"
        else:
            return tcp_count, udp_count, icmp_count, "delete", "Does not meet criteria"

    except Exception as e:
        return 0, 0, 0, "error", f"Error in tshark analysis: {str(e)}"


def analyze_pcap_with_scapy(file_path):
    """Analyze a pcap file using Scapy's PcapReader for memory efficiency"""
    try:
        # Use PcapReader instead of rdpcap for memory efficiency
        tcp_count = udp_count = icmp_count = 0
        packet_count = 0

        with PcapReader(file_path) as pcap_reader:
            for pkt in pcap_reader:
                packet_count += 1

                # Early termination if we exceed all thresholds
                if packet_count > 13:
                    return 0, 0, 0, "delete", "Exceeds all thresholds"

                if TCP in pkt:
                    tcp_count += 1
                    if tcp_count > 10:  # Early exit if tcp exceeds threshold
                        return tcp_count, udp_count, icmp_count, "delete", "TCP count exceeds threshold"
                elif UDP in pkt:
                    udp_count += 1
                    if udp_count > 13:  # Early exit if udp exceeds threshold
                        return tcp_count, udp_count, icmp_count, "delete", "UDP count exceeds threshold"
                elif ICMP in pkt:
                    icmp_count += 1
                    if icmp_count > 10:  # Early exit if icmp exceeds threshold
                        return tcp_count, udp_count, icmp_count, "delete", "ICMP count exceeds threshold"

        # Check if file meets requirements
        if ((0 < tcp_count <= 10) or (0 < udp_count <= 13) or (0 < icmp_count <= 10)):
            return tcp_count, udp_count, icmp_count, "keep", "Meets criteria"
        else:
            return tcp_count, udp_count, icmp_count, "delete", "Does not meet criteria"

    except Exception as e:
        return 0, 0, 0, "error", f"Error in Scapy analysis: {str(e)}"


def analyze_pcap(args):
    """
    Analyze a single pcap file and delete it if it doesn't meet requirements.
    Returns tuple of (status, file_path, protocol_counts_json)
    """
    file_path, use_tshark = args

    # Validate file_path
    if not file_path or not isinstance(file_path, str) or file_path.strip() == '':
        with error_count.get_lock():
            error_count.value += 1
        return "error", file_path, ""

    # Double check if file exists
    if not os.path.exists(file_path):
        with error_count.get_lock():
            error_count.value += 1
        return "error", file_path, ""

    try:
        # Choose analysis method
        if use_tshark:
            try:
                tcp_count, udp_count, icmp_count, action, reason = fast_packet_analysis(file_path)
            except:
                # Fall back to Scapy if tshark fails
                tcp_count, udp_count, icmp_count, action, reason = analyze_pcap_with_scapy(file_path)
        else:
            tcp_count, udp_count, icmp_count, action, reason = analyze_pcap_with_scapy(file_path)

        # Create JSON string of protocol counts
        protocol_counts = json.dumps({
            'tcp': tcp_count,
            'udp': udp_count,
            'icmp': icmp_count,
            'reason': reason
        })

        # Take action based on analysis
        if action == "keep":
            with kept_count.get_lock():
                kept_count.value += 1
            return "kept", file_path, protocol_counts

        elif action == "delete":
            # Delete the file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    with deleted_count.get_lock():
                        deleted_count.value += 1
                    return "deleted", file_path, protocol_counts
                except Exception as e:
                    with error_count.get_lock():
                        error_count.value += 1
                    return "error", file_path, protocol_counts
            else:
                with error_count.get_lock():
                    error_count.value += 1
                return "error", file_path, protocol_counts

        else:  # error
            with error_count.get_lock():
                error_count.value += 1
            return "error", file_path, protocol_counts

    except Exception as e:
        with error_count.get_lock():
            error_count.value += 1
        return "error", file_path, json.dumps({'error': str(e)})


def check_tshark_available():
    """Check if tshark is available on the system"""
    try:
        subprocess.check_call(['which', 'tshark'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False


def process_directory(args):
    """
    Process all pcap files in input_dir in parallel,
    skipping those already processed
    """
    input_dir = args.input_dir
    db_file = args.db_file
    num_workers = args.workers
    batch_size = args.batch_size

    # Make sure database directory exists
    os.makedirs(os.path.dirname(db_file) if os.path.dirname(db_file) else '.', exist_ok=True)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Setup database connection
    conn = setup_database(db_file)

    # Load the set of processed files
    processed_files = load_processed_files(conn)
    print(f"Found {len(processed_files)} previously processed files")

    # Get all files in the input directory
    try:
        file_list = os.listdir(input_dir)
        all_files = []

        for f in file_list:
            full_path = os.path.join(input_dir, f)
            if os.path.isfile(full_path) and f.endswith(('.pcap', '.pcapng')):
                all_files.append(full_path)
    except Exception as e:
        print(f"Error listing files in input directory: {e}")
        return

    # Filter out already processed files
    files_to_process = [f for f in all_files if f not in processed_files]

    total_files = len(files_to_process)
    print(f"Found {total_files} new pcap files to process")

    if total_files == 0:
        print("No new files to process")
        return

    # Check if tshark is available
    use_tshark = check_tshark_available() and not args.no_tshark
    if use_tshark:
        print("Using tshark for faster packet analysis")
    else:
        print("Using Scapy for packet analysis")

    # Initialize counters
    kept_count.value = 0
    deleted_count.value = 0
    error_count.value = 0

    # Process files in batches to reduce overhead
    batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]

    print(f"Processing {len(batches)} batches with {num_workers} workers")

    start_time = time.time()

    # Using 'fork' for better performance on Linux
    ctx = mp.get_context('fork')

    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)}, {len(batch)} files")

        batch_start = time.time()

        # Create a pool for this batch
        with ctx.Pool(num_workers) as pool:
            # Prepare arguments for each file (file_path, use_tshark)
            file_args = [(f, use_tshark) for f in batch]

            # Process files in parallel
            results = pool.map(analyze_pcap, file_args)

        # Update the database with this batch
        update_log_batch(conn, results)

        batch_time = time.time() - batch_start
        files_per_second = len(batch) / batch_time if batch_time > 0 else 0

        print(f"Batch {batch_idx + 1} completed in {batch_time:.2f} seconds ({files_per_second:.2f} files/sec)")
        print(f"Current stats - Kept: {kept_count.value}, Deleted: {deleted_count.value}, Errors: {error_count.value}")

    # Close the database connection
    conn.close()

    # Calculate overall stats
    total_time = time.time() - start_time
    files_per_second = total_files / total_time if total_time > 0 else 0

    # Print final statistics
    print("\n--- Final Statistics ---")
    print(f"Total files processed this run: {total_files}")
    print(f"Files kept: {kept_count.value}")
    print(f"Files deleted: {deleted_count.value}")
    print(f"Files with errors: {error_count.value}")
    if total_files > 0:
        print(f"Success rate: {(kept_count.value + deleted_count.value) / total_files * 100:.2f}%")
        print(f"Kept rate: {kept_count.value / total_files * 100:.2f}%")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Processing speed: {files_per_second:.2f} files per second")
    print(f"Using {num_workers} worker processes")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process PCAP files in parallel')
    parser.add_argument('--input-dir', '-i', type=str, default=INPUT_DIR,
                        help=f'Input directory containing PCAP files (default: {INPUT_DIR})')
    parser.add_argument('--db-file', '-d', type=str, default=DB_FILE,
                        help=f'Database file to track processed files (default: {DB_FILE})')
    parser.add_argument('--workers', '-w', type=int, default=min(mp.cpu_count(), 64),
                        help=f'Number of worker processes (default: min(cpu_count, 64))')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                        help='Number of files to process in each batch (default: 1000)')
    parser.add_argument('--no-tshark', action='store_true',
                        help='Disable tshark and use Scapy only')
    return parser.parse_args()


if __name__ == "__main__":
    print("PCAP Processor - Started")
    args = parse_args()
    process_directory(args)
    print("Processing complete!")