import json
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from vector_service import VectorService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def load_json_metadata(json_path: str) -> List[Dict[str, Any]]:
    """Load the JSON metadata file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON metadata: {e}")
        return []

def load_extracted_text(text_path: str) -> str:
    """Load the extracted text file."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading extracted text: {e}")
        return ""

def extract_text_chunks(metadata: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    """Extract text chunks based on metadata offsets."""
    documents = []
    
    for item in metadata:
        try:
            start_offset = item.get('start_offset', 0)
            end_offset = item.get('end_offset', 0)
            
            if start_offset < len(full_text) and end_offset <= len(full_text):
                section_text = full_text[start_offset:end_offset].strip()
                
                if section_text:  # Only add if we have text content
                    # Ensure chunk_id is a string and not empty
                    chunk_id = str(item.get('chunk_id', '')).strip()
                    if not chunk_id:
                        chunk_id = f"chunk_{len(documents) + 1}"
                    
                    document = {
                        "document_id": item.get('document_id', ''),
                        "disease_type": item.get('disease_type', ''),
                        "disease_name": item.get('disease_name', ''),
                        "disease_id": item.get('disease_id', ''),
                        "chunk_id": chunk_id,
                        "chunk_index": item.get('chunk_index', ''),
                        "section_type": item.get('section_type', ''),
                        "page_number": item.get('page_number', ''),
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                        "section_text": section_text
                    }
                    documents.append(document)
                else:
                    logger.warning(f"Empty text for chunk {item.get('chunk_id', '')}")
            else:
                logger.warning(f"Invalid offsets for chunk {item.get('chunk_id', '')}: {start_offset}-{end_offset}")
                
        except Exception as e:
            logger.error(f"Error processing chunk {item.get('chunk_id', '')}: {e}")
            continue
    
    return documents

async def main():
    """Main function to ingest data into the vector database."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "pdf")
    if not os.path.isdir(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    # File paths
    json_path = os.path.join(pdf_dir, "PrincDoencas.json")
    text_path = os.path.join(pdf_dir, "extracted_text.txt")

    milvus_dir = os.getenv("MILVUS_DATA_DIR")
    if not milvus_dir:
        milvus_dir = os.path.join(base_dir, "milvus_data")
    os.makedirs(milvus_dir, exist_ok=True)

    # Clean up old database files if they exist
    import glob
    old_db_files = glob.glob(os.path.join(milvus_dir, "milvus_data_*.db"))
    for db_file in old_db_files:
        try:
            os.remove(db_file)
            print(f"Removed old database file: {db_file}")
        except Exception as e:
            print(f"Could not remove {db_file}: {e}")
    
    # Check if files exist
    if not os.path.exists(json_path):
        logger.error(f"JSON metadata file not found: {json_path}")
        return
    
    if not os.path.exists(text_path):
        logger.error(f"Extracted text file not found: {text_path}")
        return
    
    # Load data
    logger.info("Loading metadata and text...")
    metadata = load_json_metadata(json_path)
    full_text = load_extracted_text(text_path)
    
    if not metadata:
        logger.error("No metadata loaded")
        return
    
    if not full_text:
        logger.error("No text loaded")
        return
    
    logger.info(f"Loaded {len(metadata)} metadata entries")
    logger.info(f"Loaded {len(full_text)} characters of text")
    
    # Extract text chunks
    logger.info("Extracting text chunks...")
    documents = extract_text_chunks(metadata, full_text)
    
    if not documents:
        logger.error("No documents extracted")
        return
    
    logger.info(f"Extracted {len(documents)} text chunks")
    
    # Filter out documents with empty or very short text
    filtered_documents = [
        doc for doc in documents 
        if doc.get('section_text', '').strip() and len(doc.get('section_text', '').strip()) > 50
    ]
    
    logger.info(f"Filtered to {len(filtered_documents)} valid documents")
    
    # Log a sample document for debugging
    if filtered_documents:
        sample_doc = filtered_documents[0]
        logger.info(f"Sample document structure:")
        logger.info(f"  - chunk_id: {sample_doc.get('chunk_id')} (type: {type(sample_doc.get('chunk_id'))})")
        logger.info(f"  - chunk_index: {sample_doc.get('chunk_index')} (type: {type(sample_doc.get('chunk_index'))})")
        logger.info(f"  - section_text length: {len(sample_doc.get('section_text', ''))}")
    
    # Ingest into vector database
    logger.info("Ingesting documents into vector database...")
    async with VectorService() as vector_service:
        success = await vector_service.insert_documents(filtered_documents)
        
        if success:
            logger.info(f"Successfully ingested {len(filtered_documents)} documents")
        else:
            logger.error("Failed to ingest documents")
    
    logger.info("Data ingestion completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
