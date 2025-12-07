"""
Embeddings Module - Sentence-BERT Vectorization Service

This module provides semantic embedding generation using Sentence-BERT
for resume-job matching via cosine similarity.

Model: all-MiniLM-L6-v2
- 384-dimensional vectors
- ~10-50ms per embedding on CPU
- Good balance of speed and quality for semantic similarity
"""

from typing import List, Optional
import hashlib

import numpy as np

# Lazy load the model to avoid startup delay
_MODEL = None


def _get_model():
    """
    Lazy load the Sentence-BERT model.

    Returns the model instance, loading it on first call.
    """
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    return _MODEL


def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding vector for a single text.

    Args:
        text: Text string to embed.

    Returns:
        Normalized 384-dim numpy array (dot product = cosine similarity).
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        return np.zeros(384, dtype=np.float32)

    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.astype(np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embedding vectors for multiple texts (batched for efficiency).

    Args:
        texts: List of text strings to embed.

    Returns:
        2D numpy array of shape (len(texts), 384) with normalized vectors.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    # Filter empty texts but preserve indices
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        return np.zeros((len(texts), 384), dtype=np.float32)

    model = _get_model()
    valid_embeddings = model.encode(valid_texts, normalize_embeddings=True)

    # Create result array with zeros for empty texts
    result = np.zeros((len(texts), 384), dtype=np.float32)
    for i, embedding in zip(valid_indices, valid_embeddings):
        result[i] = embedding

    return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Since vectors are normalized, dot product equals cosine similarity.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    # Handle zero vectors
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0

    return float(np.dot(vec1, vec2))


def text_hash(text: str) -> str:
    """
    Generate MD5 hash for caching embeddings and LLM responses.

    Args:
        text: Text string to hash.

    Returns:
        32-character MD5 hex digest.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def embedding_to_list(embedding: np.ndarray) -> List[float]:
    """
    Convert numpy embedding to list for database storage.

    Args:
        embedding: Numpy array embedding.

    Returns:
        List of floats suitable for pgvector.
    """
    return embedding.tolist()


def list_to_embedding(embedding_list: List[float]) -> np.ndarray:
    """
    Convert database list back to numpy embedding.

    Args:
        embedding_list: List of floats from database.

    Returns:
        Numpy array embedding.
    """
    return np.array(embedding_list, dtype=np.float32)


def check_model_available() -> bool:
    """
    Check if the embedding model is available.

    Returns:
        True if model can be loaded, False otherwise.
    """
    try:
        _get_model()
        return True
    except ImportError:
        return False
    except Exception:
        return False


# Embedding dimension constant
EMBEDDING_DIM = 384
