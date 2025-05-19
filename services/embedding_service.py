import os
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
openai.api_key = OPENAI_API_KEY

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using OpenAI's API.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (list of floats)
    """
    if not texts:
        return []
        
    try:
        # Clean and filter empty strings
        texts = [str(text).strip() for text in texts if str(text).strip()]
        if not texts:
            return []
            
        response = openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        
        # Return embeddings in the same order as input texts
        return [item.embedding for item in response.data]
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def get_single_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Input text string
        
    Returns:
        Embedding vector (list of floats)
    """
    if not text or not str(text).strip():
        return []
    embeddings = get_embeddings([str(text).strip()])
    return embeddings[0] if embeddings else []
