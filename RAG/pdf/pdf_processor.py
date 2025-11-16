import fitz
import json

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF - simple approach like FastAPI"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text()
    
    doc.close()
    return text

def clean_text(text: str) -> str:
    """Remove unwanted phrases from text"""
    # Remove the header phrase from all pages
    text = text.replace("BIOSSEGURIDADE NA BOVINOCULTURA LEITEIRA\n", "")
    
    # Remove author names
    authors = [
        "Geferson Fischer",
        "Rogerio Rodrigues", 
        "Felipe Geraldo Pappen",
        "Maira Balbinotti Zanela",
        "Maria Edi Rocha Ribeiro",
        "Laura Lopes de Almeida",
        "Guilherme Nunes de Souza",
        "Christiano Fanck Weissheimer",
        "Ligia Margareth Cantarelli Pegoraro",
        "Jorgea Pradieé"
    ]
    
    for author in authors:
        text = text.replace(author, "")
    
    # Remove blank lines (multiple consecutive newlines)
    import re
    text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single newline
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with 2 newlines
    
    # You can add more phrases to remove here
    # text = text.replace("OTHER PHRASE", "")
    
    return text

def create_disease_chunks(text: str) -> list:
    """Create disease chunks manually - you can modify this as needed"""
    chunks = []
    
    # Example: Manual chunking based on disease names
    # You can modify these ranges manually based on your PDF content
    
    # IBR sections
    chunks.append({
        "document_id": "PrincDoencas",
        "disease_type": "DoencasViricas", 
        "disease_id": "IBR",
        "start_offset": 0,
        "end_offset": 2500,
        "disease_metadata": {
            "Diagnostico": "IBR diagnosis info",
            "Controle": "IBR control measures",
            "Controle_e_Profilaxia": "IBR prophylaxis"
        }
    })
    
    # Febre Aftosa sections
    chunks.append({
        "document_id": "PrincDoencas",
        "disease_type": "DoencasViricas",
        "disease_id": "Febre_Aftosa", 
        "start_offset": 14700,
        "end_offset": 18600,
        "disease_metadata": {
            "Diagnostico": "Febre Aftosa diagnosis info",
            "Controle": "Febre Aftosa control measures", 
            "Controle_e_Profilaxia": "Febre Aftosa prophylaxis"
        }
    })
    
    # Add more chunks manually as needed...
    
    return chunks

def main():
    """Main processing function - simple and manual"""
    pdf_path = "Cap1PrincDoencasLvCpact.pdf"
    
    print("Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    print(f"Original text length: {len(text)} characters")
    
    # Clean the text
    print("Cleaning text...")
    text = clean_text(text)
    
    print(f"Cleaned text length: {len(text)} characters")
    
    # Save raw text
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Raw text saved to: extracted_text.txt")
    
    # Create disease chunks manually
    print("Creating disease chunks...")
    chunks = create_disease_chunks(text)
    
    # Save JSON
    with open("PrincDoencas.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Disease chunks saved to: PrincDoencas.json")
    
    print(f"Created {len(chunks)} disease chunks")

if __name__ == "__main__":
    main()
