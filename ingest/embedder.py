import os
import pdfplumber
import requests
import faiss
import numpy as np
import pickle
import json
import re
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """
    Semantic text chunker that splits text based on meaning and structure using cosine similarity
    """
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 200, 
                 similarity_threshold: float = 0.5):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize a simple TF-IDF vectorizer for sentence similarity
        self.sentence_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 1),
            lowercase=True,
            min_df=1
        )
        self._is_fitted = False
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Enhanced sentence splitting patterns
        patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Period, exclamation, question mark followed by capital
            r'(?<=\n)\s*(?=[A-Z])',     # Newline followed by capital
            r'(?<=:)\s*(?=[A-Z])',      # Colon followed by capital (for lists)
            r'(?<=;)\s*(?=[A-Z])',      # Semicolon followed by capital
        ]
        
        sentences = [text]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get TF-IDF embedding for sentence"""
        if not sentence.strip():
            return np.zeros(200)  # Match max_features
        
        try:
            # Transform single sentence
            vector = self.sentence_vectorizer.transform([sentence])
            return vector.toarray()[0]
        except:
            return np.zeros(200)
    
    def calculate_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate cosine similarity between two sentences using TF-IDF vectors"""
        if not sent1.strip() or not sent2.strip():
            return 0.0
        
        try:
            # Get TF-IDF vectors for both sentences
            emb1 = self.get_sentence_embedding(sent1)
            emb2 = self.get_sentence_embedding(sent2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
        except:
            return 0.0
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """
        Create chunks based on semantic similarity between sentences using cosine similarity
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        
        # Fit TF-IDF vectorizer on all sentences first
        try:
            self.sentence_vectorizer.fit(sentences)
            self._is_fitted = True
        except:
            print("⚠️ Warning: Could not fit TF-IDF vectorizer, using fallback similarity")
            self._is_fitted = False
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max_chunk_size
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Finalize current chunk if it meets minimum size
                if current_length >= self.min_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Add to current chunk even if it exceeds max size
                    current_chunk.append(sentence)
                    current_length += sentence_length
            else:
                # Check semantic similarity with the last sentence in current chunk
                if current_chunk:
                    similarity = self.calculate_similarity(current_chunk[-1], sentence)
                    
                    # If similarity is low, start a new chunk (if current chunk is large enough)
                    if (similarity < self.similarity_threshold and 
                        current_length >= self.min_chunk_size):
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
class HybridPDFEmbedder:
    def __init__(self, folder_path: str, model: str = "nomic-embed-text", 
                 chunk_size: int = 800, min_chunk_size: int = 200, 
                 alpha: float = 0.7, similarity_threshold: float = 0.5):
        """
        Initialize the Hybrid PDF Embedder with Semantic Chunking
        
        Args:
            folder_path: Path to folder containing PDF documents
            model: Embedding model name for dense embeddings
            chunk_size: Maximum size of text chunks
            min_chunk_size: Minimum size of text chunks
            alpha: Weight for dense embeddings in hybrid approach (0.7 means 70% dense, 30% sparse)
            similarity_threshold: Threshold for semantic similarity in chunking
        """
        self.folder_path = folder_path
        self.model = model
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.alpha = alpha  # Weight for dense embeddings
        self.similarity_threshold = similarity_threshold
        
        # Initialize semantic chunker
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold
        )
        
        # Storage for processed data
        self.texts: List[str] = []
        self.dense_embeddings: List[List[float]] = []
        self.sparse_embeddings = None
        self.hybrid_embeddings: List[np.ndarray] = []
        self.document_metadata: List[Dict[str, Any]] = []
        
        # TF-IDF vectorizer for sparse embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"  Extracted page {page_num}")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
        return text

    def get_dense_embedding(self, text: str) -> List[float]:
        """Get dense embedding from the API"""
        try:
            response = requests.post(
                "http://172.19.20.217:11435/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ Error getting dense embedding: {e}")
            raise

    def create_sparse_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create sparse (TF-IDF) embeddings for all texts"""
        print("🔍 Creating sparse embeddings using TF-IDF...")
        sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return sparse_matrix.toarray().astype('float32')

    def create_hybrid_embeddings(self, dense_embs: np.ndarray, sparse_embs: np.ndarray) -> np.ndarray:
        """
        Combine dense and sparse embeddings with alpha weighting
        
        Args:
            dense_embs: Dense embeddings matrix
            sparse_embs: Sparse embeddings matrix
            
        Returns:
            Hybrid embeddings matrix
        """
        print(f"🔗 Creating hybrid embeddings with alpha={self.alpha}")
        
        # Normalize both embedding types
        dense_normalized = normalize(dense_embs, norm='l2')
        sparse_normalized = normalize(sparse_embs, norm='l2')
        
        # Combine with alpha weighting
        hybrid = (self.alpha * dense_normalized + 
                 (1 - self.alpha) * sparse_normalized)
        
        return hybrid.astype('float32')

    def process_documents(self) -> Tuple[List[str], np.ndarray]:
        """
        Main pipeline: Extract, chunk, and create hybrid embeddings for all PDFs
        
        Returns:
            Tuple of (text_chunks, hybrid_embeddings)
        """
        print(f"📁 Processing PDFs in {self.folder_path}")
        
        # Step 1: Extract and chunk all documents
        all_chunks = []
        
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.folder_path, filename)
                print(f"📄 Processing: {filename}")
                
                # Extract text
                raw_text = self.extract_text_from_pdf(full_path)
                if not raw_text.strip():
                    print(f"⚠️ No text found in {filename}, skipping.")
                    continue
                
                # Chunk text using semantic chunking
                chunks = self.semantic_chunker.create_semantic_chunks(raw_text)
                print(f"  Created {len(chunks)} semantic chunks")
                
                # Store chunks with metadata
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        all_chunks.append(chunk)
                        self.document_metadata.append({
                            'filename': filename,
                            'chunk_index': i,
                            'chunk_size': len(chunk)
                        })
        
        if not all_chunks:
            print("❌ No text chunks found in any documents!")
            return [], np.array([])
        
        print(f"📝 Total chunks created: {len(all_chunks)}")
        self.texts = all_chunks
        
        # Step 2: Create dense embeddings
        print("🧠 Creating dense embeddings...")
        dense_embeddings = []
        
        for i, chunk in enumerate(all_chunks):
            try:
                embedding = self.get_dense_embedding(chunk)
                dense_embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(all_chunks)} dense embeddings")
            except Exception as e:
                print(f"❌ Error creating dense embedding for chunk {i}: {e}")
                continue
        
        if not dense_embeddings:
            print("❌ No dense embeddings created!")
            return self.texts, np.array([])
        
        self.dense_embeddings = dense_embeddings
        dense_matrix = np.array(dense_embeddings).astype('float32')
        
        # Step 3: Create sparse embeddings
        sparse_matrix = self.create_sparse_embeddings(all_chunks)
        
        # Step 4: Create hybrid embeddings
        hybrid_matrix = self.create_hybrid_embeddings(dense_matrix, sparse_matrix)
        self.hybrid_embeddings = hybrid_matrix
        
        print(f"✅ Pipeline complete! Created {len(hybrid_matrix)} hybrid embeddings")
        return self.texts, hybrid_matrix

    def save_to_faiss(self, faiss_index_path: str = "hybrid_index.faiss", 
                     metadata_path: str = "hybrid_metadata.json",
                     tfidf_path: str = "tfidf_vectorizer.pkl") -> bool:
        """
        Save hybrid embeddings and metadata to disk
        
        Args:
            faiss_index_path: Path to save FAISS index
            metadata_path: Path to save text chunks and metadata
            tfidf_path: Path to save TF-IDF vectorizer
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.hybrid_embeddings) == 0:
            print("⚠️ No hybrid embeddings to save.")
            return False

        try:
            # Save FAISS index
            vectors = np.array(self.hybrid_embeddings).astype("float32")
            dim = len(vectors[0])
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            
            faiss.write_index(index, faiss_index_path)
            print(f"✅ Saved hybrid FAISS index to {faiss_index_path}")

            # Save metadata and texts
            metadata = {
                'texts': self.texts,
                'document_metadata': self.document_metadata,
                'config': {
                    'model': self.model,
                    'chunk_size': self.chunk_size,
                    'min_chunk_size': self.min_chunk_size,
                    'alpha': self.alpha,
                    'similarity_threshold': self.similarity_threshold,
                    'total_chunks': len(self.texts),
                    'embedding_dim': dim,
                    'chunking_method': 'semantic'
                }
            }
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved metadata to {metadata_path}")

            # Save TF-IDF vectorizer for future sparse embeddings
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"✅ Saved TF-IDF vectorizer to {tfidf_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving to disk: {e}")
            return False

    def load_vectorizer(self, tfidf_path: str = "tfidf_vectorizer.pkl") -> bool:
        """Load pre-trained TF-IDF vectorizer"""
        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"✅ Loaded TF-IDF vectorizer from {tfidf_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading TF-IDF vectorizer: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed documents"""
        if not self.texts:
            return {}
        
        stats = {
            'total_documents': len(set(meta['filename'] for meta in self.document_metadata)),
            'total_chunks': len(self.texts),
            'avg_chunk_size': np.mean([len(text) for text in self.texts]),
            'dense_embedding_dim': len(self.dense_embeddings[0]) if self.dense_embeddings else 0,
            'hybrid_embedding_dim': len(self.hybrid_embeddings[0]) if len(self.hybrid_embeddings) > 0 else 0,
            'alpha_value': self.alpha,
            'model_used': self.model,
            'chunking_method': 'semantic',
            'similarity_threshold': self.similarity_threshold,
            'min_chunk_size': self.min_chunk_size
        }
        
        return stats

if __name__ == "__main__":
    print("🚀 Starting Hybrid PDF Embedding Pipeline")
    print("=" * 50)
    
    # Initialize the hybrid embedder with semantic chunking
    embedder = HybridPDFEmbedder(
        folder_path="../documents",  # Relative to ingest folder
        model="nomic-embed-text",
        chunk_size=800,          # Increased for semantic chunks
        min_chunk_size=200,      # Minimum chunk size
        alpha=0.7,               # 70% dense, 30% sparse
        similarity_threshold=0.5  # Threshold for semantic similarity
    )
    
    # Process all documents
    texts, hybrid_embeddings = embedder.process_documents()
    
    if len(hybrid_embeddings) > 0:
        # Save results
        success = embedder.save_to_faiss(
            faiss_index_path="../rag_backend/hybrid_docs_index.faiss",
            metadata_path="../rag_backend/hybrid_docs_metadata.json",
            tfidf_path="../rag_backend/tfidf_vectorizer.pkl"
        )
        
        if success:
            # Print statistics
            stats = embedder.get_stats()
            print("\n📊 Processing Statistics:")
            print("=" * 30)
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            print(f"\n✅ Hybrid embedding pipeline completed successfully!")
            print(f"📁 Files saved in rag_backend/ directory")
        else:
            print("❌ Failed to save results")
    else:
        print("❌ No embeddings created")

