def main():
    print("Hello from packetmako!")
    hex_to_token = {i: i for i in range(256)}
    print(hex_to_token)

    allocated_tokens = set(range(256))
    print(allocated_tokens)

if __name__ == "__main__":
    main()
