# Document Q&A API

This API provides endpoints for document upload, indexing, and question answering using FAISS vector search and LangChain.

## Setup

1. Create a virtual environment:
```bash
python -m venv faissenv
source faissenv/bin/activate  # On Windows: faissenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running locally (default: http://localhost:11434)

## API Endpoints

### Upload Documents
- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: PDF files (can upload multiple)
- **Response**:
```json
{
    "message": "Files uploaded and indexed successfully",
    "files": ["document1.pdf", "document2.pdf"],
    "document_count": 10
}
```

### Query Documents
- **URL**: `/query`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
    "query": "What are the top OWASP vulnerabilities?"
}
```
- **Response**:
```json
{
    "results": [
        {
            "content": "Document content...",
            "metadata": {
                "source": "file.pdf",
                "page": 1
            },
            "relevance_score": 0.95
        }
    ]
}
```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
    "status": "healthy"
}
```

## Running the API

```bash
python app.py
```

The API will start on http://localhost:5000

## Example Usage with cURL

1. Upload documents:
```bash
curl -X POST -F "file=@document1.pdf" -F "file=@document2.pdf" http://localhost:5000/upload
```

2. Query documents:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query":"What are the OWASP top 10?"}' http://localhost:5000/query
```

3. Health check:
```bash
curl http://localhost:5000/health
```


python3.10 -m venv faisenv1
source faisenv1/bin/activate