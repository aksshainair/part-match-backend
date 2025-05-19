# Invoice Parts Matcher API

A FastAPI-based service for matching invoice line items to a parts master catalog using semantic search with vector embeddings.

## Features

- Upload PDF documents (invoices, POs) with automatic text extraction
- Generate embeddings for document line items
- Match line items against a parts master catalog using semantic search
- Batch processing of multiple line items
- Single item matching API
- MongoDB for document storage
- Qdrant for vector similarity search

## Prerequisites

- Python 3.8+
- MongoDB instance
- Qdrant server (can be run in a container)
- OpenAI API key (for embeddings)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the `fastapi_app` directory with the following variables:
   ```
   MONGODB_URI=mongodb://localhost:27017/
   MONGODB_DB=invoice_matcher
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=parts_master
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   ```

## Running the Application

1. Start the FastAPI server:
   ```bash
   cd fastapi_app
   uvicorn main:app --reload
   ```
2. The API will be available at `http://localhost:8000`
3. API documentation (Swagger UI) will be available at `http://localhost:8000/docs`

## API Endpoints

### Upload Document
- **POST** `/upload/` - Upload a PDF document for processing

### Get Document
- **GET** `/documents/{document_id}` - Get document details by ID

### Batch Match
- **POST** `/batch-match/{document_id}` - Match all line items in a document against the parts master

### Single Match
- **POST** `/single-match/` - Match a single line item description against the parts master

## Development

### Running Tests
```bash
pytest
```

### Code Style
This project uses Black for code formatting and Flake8 for linting.

## License
MIT
