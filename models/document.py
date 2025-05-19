from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"

class LineItem(BaseModel):
    """Represents a single line item in a document."""
    line_number: str
    description: str
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    total_price: Optional[str] = None
    unit_of_measure: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentBase(BaseModel):
    """Base model for document operations."""
    filename: str
    content_type: str
    size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass

class DocumentInDB(DocumentBase):
    """Model representing a document in the database."""
    id: str = Field(..., alias="_id")
    upload_date: datetime
    status: DocumentStatus
    line_items: List[LineItem] = Field(default_factory=list)
    processing_errors: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            ObjectId: str
        }
        allow_population_by_field_name = True

class DocumentResponse(DocumentInDB):
    """Response model for document endpoints."""
    pass

class MatchResult(BaseModel):
    """Represents a matching result between a line item and a part."""
    line_item: LineItem
    best_match: Optional[Dict[str, Any]] = None
    score: float = 0.0
    matched: bool = False
    match_metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchMatchResult(BaseModel):
    """Response model for batch matching results."""
    document_id: str
    total_items: int
    matched_items: int
    match_rate: float
    matches: List[MatchResult]
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 4)  # Ensure consistent decimal places
        }
    
    @classmethod
    def from_matches(
        cls, 
        document_id: str, 
        matches: List[MatchResult]
    ) -> 'BatchMatchResult':
        """Create a BatchMatchResult from a list of MatchResult objects."""
        total = len(matches)
        matched = sum(1 for m in matches if m.matched)
        match_rate = (matched / total) if total > 0 else 0.0
        
        return cls(
            document_id=str(document_id),
            total_items=total,
            matched_items=matched,
            match_rate=round(match_rate, 4),
            matches=matches
        )
