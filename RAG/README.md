# 🐄 BoviCare RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for bovine disease information, featuring hybrid search capabilities and AI-powered reranking.

## ✨ Features

- **Hybrid Search**: Combines dense vector embeddings with BM25 sparse search for optimal results
- **AI Reranking**: Uses OpenAI's GPT models to intelligently rerank search results
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Beautiful Frontend**: Intuitive chatbot interface with source citations
- **Veterinary Expertise**: Specialized in bovine diseases and veterinary medicine
- **Local Vector Database**: Uses Milvus Lite for local development (no external dependencies)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Vector        │
│   (Chatbot)     │◄──►│   Backend       │◄──►│   Database      │
│                 │    │                 │    │   (Milvus)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (Embeddings   │
                       │   & Reranking)  │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
cd RAG-bovicare

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your OpenAI API key
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Data Ingestion

```bash
# Run the data ingestion script to populate the vector database
python ingest_data.py
```

### 4. Start the Application

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application

- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
RAG-bovicare/
├── main.py                 # FastAPI application
├── vector_service.py       # Milvus vector database service
├── rag.py                  # RAG operations (reranking, response generation)
├── ingest_data.py          # Data ingestion script
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── templates/
│   └── index.html         # Chatbot frontend
└── pdf/
    ├── PrincDoencas.json  # Disease metadata
    └── extracted_text.txt  # Extracted text content
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and reranking | Yes | - |
| `MILVUS_URI` | Milvus server URI (optional) | No | Local database |
| `MILVUS_API_TOKEN` | Milvus authentication token | No | - |
| `LOG_LEVEL` | Logging level | No | INFO |

### API Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `top_k` | Number of results to return | 5 | 1-20 |
| `use_reranking` | Enable AI-powered reranking | true | true/false |

## 🧠 How It Works

### 1. Document Processing
- JSON metadata contains disease information and text offsets
- Text chunks are extracted based on metadata offsets
- Each chunk is embedded using OpenAI's text-embedding-3-small model

### 2. Hybrid Search
- **Dense Search**: Semantic similarity using vector embeddings
- **Sparse Search**: Keyword-based search using BM25 algorithm
- **Combined Ranking**: Weighted combination of both search methods

### 3. AI Reranking
- Retrieved documents are scored by GPT-4o-mini for relevance
- Structured output ensures consistent scoring
- Final ranking based on AI relevance scores

### 4. Response Generation
- Context from top-ranked documents is provided to GPT-4o-mini
- Specialized veterinary medicine prompts ensure accurate responses
- Source citations are included for transparency

## 📊 Data Schema

### Document Structure
```json
{
  "document_id": "PrincDoencas",
  "disease_type": "DoencasViricas",
  "disease_name": "RinotraqueíteInfecciosaBovina(IBR)",
  "disease_id": "IBR",
  "chunk_id": "2",
  "chunk_index": "1",
  "section_type": "overview",
  "page_number": [13, 14],
  "section_text": "Disease description text...",
  "start_offset": 1059,
  "end_offset": 2554
}
```

### Search Response
```json
{
  "query": "What are the symptoms of IBR?",
  "response": "AI-generated response based on context...",
        "sources": [
        {
          "disease_name": "RinotraqueíteInfecciosaBovina(IBR)",
          "chunk_index": "1",
          "section_type": "overview",
          "relevance_score": 0.95,
          "content_preview": "Symptom description..."
        }
      ]
}
```

## 🧪 Testing

### Example Queries

Try these sample questions to test the system:

1. **"What are the symptoms of IBR?"**
2. **"How is BVD diagnosed?"**
3. **"What are the control measures for foot and mouth disease?"**
4. **"Tell me about viral diseases in cattle"**
5. **"What vaccines are available for bovine diseases?"**

### API Testing

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test a query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of IBR?", "top_k": 3}'
```

## 🚀 Deployment

### Production Considerations

1. **Vector Database**: Use Milvus cloud or self-hosted instance
2. **API Keys**: Secure environment variable management
3. **Rate Limiting**: Implement API rate limiting
4. **Monitoring**: Add logging and monitoring
5. **Scaling**: Use multiple workers with uvicorn

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For questions or issues:
- Check the API documentation at `/docs`
- Review the logs for error details
- Ensure all environment variables are set correctly
- Verify the data ingestion completed successfully

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Image analysis for disease symptoms
- [ ] Integration with veterinary databases
- [ ] Real-time disease outbreak information
- [ ] Mobile application
- [ ] Advanced analytics dashboard
