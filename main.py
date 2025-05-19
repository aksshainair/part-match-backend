from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient, IndexModel, ASCENDING
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import hashlib
import os
from dotenv import load_dotenv
import uuid
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
import pymupdf as fitz
import openai
from tqdm import tqdm
import logging

# Import our modules
from services.embedding_service import get_embeddings, get_single_embedding
from services.qdrant_service import QdrantService
from models.document import (
    DocumentInDB, DocumentCreate, DocumentResponse,
    MatchResult, BatchMatchResult, DocumentStatus, LineItem
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Invoice Parts Matcher API",
    description="API for matching invoice line items to parts master catalog",
    version="1.0.0"
)

# Path to the temporary parsed invoice JSON file
TEMP_PARSED_INVOICE_PATH = "./parsed_invoice.json"

# Constants
VECTOR_SIZE = 3072  # For text-embedding-3-large
INVOICE_PO_COLLECTION = "inv_po_data"  # Qdrant collection name

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Helper Functions ---

def load_parsed_invoice(file_path: str) -> Dict[str, Any]:
    """Load the parsed invoice JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading parsed invoice: {e}")
        return {"line_items": []}

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting text from PDF: {str(e)}"
        )

def get_openai_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector from OpenAI API."""
    if not text:
        return None
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        if len(embedding) != VECTOR_SIZE:
            raise ValueError(f"Embedding length mismatch: got {len(embedding)}, expected {VECTOR_SIZE}")
        return embedding
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embedding: {str(e)}"
        )

def parse_document_with_llm(text: str, doc_type_hint: str = None) -> Dict[str, Any]:
    """
    Use OpenAI LLM to parse invoice/PO text into structured line items and metadata.
    Returns a dict: {doc_type, doc_id, line_items: [...]}
    """
    system_prompt = (
        "You are an expert in financial document parsing. "
        "Extract all line items and key metadata from the following invoice or purchase order. "
        "For each line item, extract: part_number (or null), description, unit, quantity, price, and total. "
        "Also extract doc_type (invoice or purchase_order) and doc_id (invoice/PO number). "
        "Return a JSON object with keys: doc_type, doc_id, line_items (list of dicts)."
    )
    
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document text:\n---\n{text}\n---\nParse and output JSON as described."}
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Ensure all required fields exist in each line item
        if 'line_items' in parsed:
            for item in parsed['line_items']:
                item.setdefault('part_number', None)
                item.setdefault('description', None)
                item.setdefault('unit', None)
                item.setdefault('quantity', None)
                item.setdefault('price', None)
                item.setdefault('total', None)
                
        return parsed
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing document with LLM: {str(e)}"
        )



def extract_and_parse(file_content: bytes, upsert: bool = False, db_doc_id: str = "") -> Dict[str, Any]:
    """
    Extract text from PDF, parse it with LLM, and return structured data.
    """
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file_content)
        
        # Parse with LLM
        parsed_data = parse_document_with_llm(text)
        
        # Ensure required fields
        if not parsed_data or 'line_items' not in parsed_data or not parsed_data['line_items']:
            raise ValueError("No line items found in the parsed document")
        
        doc_type = parsed_data.get('doc_type', "")
        doc_id = parsed_data.get('doc_id', "")
        line_items = parsed_data['line_items']

        if upsert:
            failed_count = 0
            for idx, item in enumerate(tqdm(line_items, desc="Upserting line items")):
                desc = item.get("description") or item.get("line_item") or ""
                embedding = get_openai_embedding(desc)
                if embedding is None:
                    failed_count += 1
                    continue
                payload = {
                    "doc_type": doc_type,
                    "doc_id": doc_id,
                    "original_line_item_data": item,
                    "description_text": desc,
                    "db_doc_id": db_doc_id
                }
                qdrant_service.upsert_line_item_to_qdrant(doc_id, idx, embedding, payload)

        return parsed_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "invoice_matcher")

# Initialize MongoDB client
db_client = MongoClient(os.getenv("MONGODB_URI"))
db = db_client[os.getenv("MONGODB_DB", "invoice_matcher")]

# Create index for content hash
if 'documents' in db.list_collection_names():
    db.documents.create_index([("content_hash", ASCENDING)], unique=True)

# Initialize Qdrant service
qdrant_service = QdrantService()

# Models
class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str

class SingleMatchRequest(BaseModel):
    description: str

class SingleMatchResponse(BaseModel):
    Invoice_Description: str
    Document_Type: str
    Document_ID: str
    Part_description: Optional[str] = None
    Part_ID: Optional[str] = None
    Unit_of_measure: Optional[str] = None
    Similarity_Score: Optional[float] = None
    Matched: str

# API Endpoints
class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    success: bool
    document_id: str
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    is_duplicate: Optional[bool] = False

@app.post("/upload/", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF) and process it.
    """
    logger = logging.getLogger(__name__)
    upload_start_time = datetime.utcnow()
    
    try:
        logger.info(f"Starting document upload: {file.filename}")
        
        # Read file content
        contents = await file.read()
        logger.debug(f"Read {len(contents)} bytes from file")
        
        # Generate a unique ID and content hash for the document
        doc_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(contents).hexdigest()
        logger.debug(f"Generated document ID: {doc_id}, Content hash: {content_hash}")
        
        # Check for existing file with same content hash
        logger.debug("Checking for existing document with same content hash")
        existing_doc = db.documents.find_one({"content_hash": content_hash})
        
        if existing_doc:
            logger.info(f"Document already exists with hash {content_hash}, skipping processing")
            parsed_data = extract_and_parse(contents, upsert=False, db_doc_id=str(existing_doc["_id"]))
            logger.debug(f"Extracted data from existing document: {parsed_data}")
            
            return DocumentUploadResponse(
                success=True,
                document_id=str(existing_doc["_id"]),
                status="success",
                message="Document already exists with the same content",
                is_duplicate=True,
                data=parsed_data
            )
        
        logger.info("Creating new document record in database")
        # Create document in MongoDB
        document = {
            "_id": doc_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "content_hash": content_hash,
            "upload_date": datetime.utcnow(),
            "content": contents,
            "status": DocumentStatus.UPLOADED,
            "metadata": {},
            "line_items": [],
            "processing_errors": []
        }
        
        # Insert into MongoDB
        result = db.documents.insert_one(document)
        logger.info(f"Document inserted with ID: {doc_id}, MongoDB result: {result.inserted_id}")
        
        # Process the document
        logger.info("Starting document processing")
        process_start = datetime.utcnow()
        parsed_data = extract_and_parse(contents, upsert=True, db_doc_id=doc_id)
        process_time = (datetime.utcnow() - process_start).total_seconds()
        logger.info(f"Document processing completed in {process_time:.2f} seconds")
        
        # Store the parsed data in the document
        logger.debug("Updating document with parsed data")
        update_result = db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"parsed_data": parsed_data, "status": DocumentStatus.PROCESSED}}
        )
        logger.debug(f"Document update result: {update_result.modified_count} documents modified")
        
        total_time = (datetime.utcnow() - upload_start_time).total_seconds()
        logger.info(f"Document {doc_id} processed successfully in {total_time:.2f} seconds")
        
        return DocumentUploadResponse(
            success=True,
            document_id=str(doc_id),
            status="success",
            message=f"Document uploaded and processed in {total_time:.2f} seconds",
            is_duplicate=False,
            data=parsed_data
        )
        
    except HTTPException as he:
        logger.error(f"HTTP error during document processing: {str(he)}", exc_info=True)
        raise he
        
    except Exception as e:
        error_id = str(uuid.uuid4())
        logger.error(f"Error processing document (ID: {error_id}): {str(e)}", exc_info=True)
        
        # Update document status if it was created but processing failed
        if 'doc_id' in locals():
            try:
                db.documents.update_one(
                    {"_id": doc_id},
                    {"$set": {
                        "status": DocumentStatus.ERROR, 
                        "error": str(e),
                        "error_id": error_id,
                        "error_timestamp": datetime.utcnow()
                    }}
                )
                logger.error(f"Updated document {doc_id} with error state. Error ID: {error_id}")
            except Exception as update_err:
                logger.critical(f"Failed to update error state for document {doc_id}: {str(update_err)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Error processing document",
                "error_id": error_id,
                "message": str(e)
            }
        )

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get document details by ID.
    """
    document = db.documents.find_one({"_id": document_id})
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return DocumentResponse(**document)

@app.post("/batch-match/{document_id}", response_model=BatchMatchResult)
async def batch_match_document(document_id: str):
    """
    Perform batch matching of all line items in a document against the parts master.
    """
    try:
        # Load the parsed invoice data
        document = db.documents.find_one({"_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        parsed_data = extract_and_parse(document["content"], upsert=False)
        
        # Convert line items to LineItem models with all original fields
        line_items = []
        for idx, item in enumerate(parsed_data.get("line_items", [])):
            # Preserve all original fields in metadata
            metadata = {
                "part_number": item.get("part_number"),
                "unit": item.get("unit"),
                "price": item.get("price"),
                "total": item.get("total")
            }
            
            line_items.append(LineItem(
                line_number=str(idx + 1),
                description=str(item.get("description", "")),
                quantity=str(item.get("quantity")),
                unit_price=str(item.get("price")),
                total_price=str(item.get("total")),
                unit_of_measure=str(item.get("unit")),
                metadata=metadata
            ))
        
        if not line_items:
            return BatchMatchResult(
                document_id=str(document_id),
                total_items=str(0),
                matched_items=str(0),
                match_rate=str(0.0),
                matches=[]
            )
            
        # Process each line item
        matches = []
        for item in line_items:
            try:
                # Get embedding for the line item description
                embedding = get_single_embedding(item.description)
                
                if not embedding:
                    matches.append(MatchResult(
                        line_item=item,
                        matched=False,
                        score=0.0,
                        match_metadata={"error": "Failed to generate embedding"}
                    ))
                    continue
                
                # Find best match in Qdrant
                best_match = qdrant_service.find_best_match(embedding)
                
                if best_match and best_match.get('score', 0) >= 0.6:  # Threshold
                    matches.append(MatchResult(
                        line_item=item,
                        best_match=best_match.get('payload'),
                        score=best_match.get('score', 0),
                        matched=True
                    ))
                else:
                    matches.append(MatchResult(
                        line_item=item,
                        matched=False,
                        score=best_match.get('score', 0) if best_match else 0.0
                    ))
                    
            except Exception as e:
                matches.append(MatchResult(
                    line_item=item,
                    matched=False,
                    score=0.0,
                    match_metadata={"error": str(e)}
                ))
        
        # Return results using the from_matches class method which handles the conversion
        return BatchMatchResult.from_matches(document_id, matches)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/single-match/", response_model=SingleMatchResponse)
# async def single_match(request: SingleMatchRequest):
#     """
#     Find the best matching part for a single line item description.
#     """
#     try:
#         if not request.description.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Description cannot be empty"
#             )
            
#         # Get embedding for the description
#         embedding = get_single_embedding(request.description)
#         if not embedding:
#             return SingleMatchResponse(
#                 Invoice_Description=request.description,
#                 Document_Type="",
#                 Document_ID="",
#                 Part_description=None,
#                 Part_ID=None,
#                 Unit_of_measure=None,
#                 Similarity_Score=None,
#                 Matched="No"
#             )
        
#         # Search in Qdrant
#         best_match = qdrant_service.find_best_match(embedding)
#         best_match_in_inv_po = qdrant_service.find_best_match_in_inv_po(embedding)
        
#         if not best_match or best_match.get('score', 0) < 0.6:  # Threshold
#             return SingleMatchResponse(
#                 Invoice_Description=request.description,
#                 Document_Type="",
#                 Document_ID="",
#                 Part_description=None,
#                 Part_ID=None,
#                 Unit_of_measure=None,
#                 Similarity_Score=None,
#                 Matched="No"
#             )

#         if not best_match_in_inv_po or best_match_in_inv_po.get('score', 0) < 0.6:  # Threshold
#             return SingleMatchResponse(
#                 Invoice_Description=request.description,
#                 Document_Type="",
#                 Document_ID="",
#                 Part_description=None,
#                 Part_ID=None,
#                 Unit_of_measure=None,
#                 Similarity_Score=None,
#                 Matched="No"
#             )
            
#         payload = best_match.get('payload', {})
#         payload2 = best_match_in_inv_po.get('payload', {})
#         return SingleMatchResponse(
#             Invoice_Description=request.description,
#             Document_Type=str(payload2.get('doc_type')),
#             Document_ID=str(payload2.get('doc_id')),
#             Part_description=payload.get('description'),
#             Part_ID=payload.get('part_number'),
#             Unit_of_measure=payload.get('unit_of_measure'),
#             Similarity_Score=round(best_match.get('score', 0), 4),
#             Matched="Yes" if best_match.get('score', 0) >= 0.6 else "No"
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing match request: {str(e)}"
#         )
@app.post("/single-match/", response_model=SingleMatchResponse)
async def single_match(request: SingleMatchRequest):
    """
    Find the best matching part for a single line item description.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting single match request for description: {request.description[:100]}...")
    
    try:
        if not request.description.strip():
            logger.warning("Received empty description in single match request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Description cannot be empty"
            )
            
        logger.debug("Generating embedding for description")
        embedding = get_single_embedding(request.description)
        if not embedding:
            logger.warning("Failed to generate embedding for description")
            return SingleMatchResponse(
                Invoice_Description=request.description,
                Document_Type="",
                Document_ID="",
                Part_description=None,
                Part_ID=None,
                Unit_of_measure=None,
                Similarity_Score=None,
                Matched="No"
            )
        
        logger.debug("Searching for best match in Qdrant")
        best_match = qdrant_service.find_best_match(embedding)
        best_match_in_inv_po = qdrant_service.find_best_match_in_inv_po(embedding)
        
        if not best_match or 'payload' not in best_match:
            logger.info("No matching part found for the description")
            return SingleMatchResponse(
                Invoice_Description=request.description,
                Document_Type="",
                Document_ID="",
                Part_description=None,
                Part_ID=None,
                Unit_of_measure=None,
                Similarity_Score=None,
                Matched="No"
            )
            
        payload = best_match.get('payload', {})
        payload2 = best_match_in_inv_po.get('payload', {})
        original_line_item_data = payload2.get('original_line_item_data', {})
        
        logger.info(f"Found match with score: {best_match.get('score', 0):.4f}")
        logger.debug(f"Match payload: {payload}")

        print(f"payload : {payload}")
        print(f"payload2 : {payload2}")
                
        return SingleMatchResponse(
            Invoice_Description=request.description,
            Document_Type=str(payload2.get('doc_type', '')),
            Document_ID=str(payload2.get('doc_id', '')),
            Part_description=payload.get('description'),
            Part_ID=payload.get('part_number'),
            Unit_of_measure=original_line_item_data.get('unit'),
            Similarity_Score=round(best_match.get('score', 0), 4),
            Matched="Yes" if best_match.get('score', 0) >= 0.6 else "No"
        )
        
    except HTTPException as he:
        logger.error(f"HTTP error in single_match: {str(he)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in single_match: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing match request: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all documents in the database."""
    try:
        # Find all documents and sort by upload date (newest first)
        cursor = db.documents.find().sort("upload_date", -1)
        
        # Convert MongoDB documents to Pydantic models
        documents = []
        for doc in cursor:
            # Convert ObjectId to string for the response
            doc["id"] = str(doc["_id"])
            documents.append(DocumentResponse(**doc))
            
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
