from scapy.all import rdpcap
from pathlib import Path


def convert_pcap_directory(input_dir: str, output_dir: str) -> None:
    """
    Converts all .pcap files in a directory to byte sequences and saves them as .bin files.

    Args:
        input_dir: Path to the directory containing .pcap files
        output_dir: Path to the directory where .bin files will be saved

    Raises:
        FileNotFoundError: If input_dir doesn't exist
        PermissionError: If there are insufficient permissions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    output_path.mkdir(parents=True, exist_ok=True)

    for pcap_file in input_path.glob('*.pcap*'):
        try:
            output_file = output_path / f"{pcap_file.stem}.bin"

            # Read and convert the pcap file to bytes
            packets = rdpcap(str(pcap_file))
            byte_sequence = b''.join(bytes(packet) for packet in packets)

            # Save the byte sequence to a binary file
            output_file.write_bytes(byte_sequence)

            print(f"Successfully converted {pcap_file.name} to {output_file.name}")

        except Exception as e:
            print(f"Error processing {pcap_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    input_directory = 'data/pcaps'
    output_directory = 'data/bytes'
    convert_pcap_directory(input_directory, output_directory)