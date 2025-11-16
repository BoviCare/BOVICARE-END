import fitz

def test_offset(text_path: str, start: int, end: int):
    """Test specific offset range - simple approach"""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if start < 0 or end > len(text) or start >= end:
        print(f"Invalid offsets: start={start}, end={end}, text_length={len(text)}")
        return
    
    extracted_text = text[start:end]
    print(f"Text from offset {start}-{end} ({end-start} characters):")
    print("=" * 60)
    print(extracted_text)
    print("=" * 60)

def main():
    text_path = "extracted_text.txt"
    
    print("Simple Offset Tester")
    print("Enter 'q' to quit")
    
    while True:
        try:
            start_input = input("Enter start offset: ").strip()
            if start_input.lower() == 'q':
                break
                
            end_input = input("Enter end offset: ").strip()
            if end_input.lower() == 'q':
                break
            
            start = int(start_input)
            end = int(end_input)
            
            test_offset(text_path, start, end)
            
        except ValueError:
            print("Please enter valid numbers")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
