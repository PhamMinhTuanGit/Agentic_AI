#!/usr/bin/env python3
"""
ZebOS Commands and Chapters Embedder
====================================

This script chunks and embeds ZebOS commands and chapters JSON files
using hybrid embeddings (dense + sparse) and stores them in the database for RAG retrieval.
"""

import json
import os
import sys
import pickle
import faiss
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZebOSCommandsEmbedder:
    """Embedder for ZebOS commands and chapters documentation using hybrid embeddings"""
    
    def __init__(
        self,
        commands_file: str = "zebos_commands.json",
        chapters_file: str = "zebos_chapters.json",
        output_dir: str = "database/commands",
        embedding_model: str = "nomic-embed-text",
        alpha: float = 0.7  # Weight for dense embeddings (70% dense, 30% sparse)
    ):
        """
        Initialize ZebOS Commands Embedder
        
        Args:
            commands_file: Path to zebos_commands.json
            chapters_file: Path to zebos_chapters.json
            output_dir: Directory to store embeddings
            embedding_model: Model name for dense embeddings
            alpha: Weight for dense vs sparse embeddings
        """
        self.commands_file = commands_file
        self.chapters_file = chapters_file
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        self.alpha = alpha
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.texts: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.dense_embeddings: List[np.ndarray] = []
        self.dense_embedding_dim: int = 0
        
        # TF-IDF vectorizer for sparse embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # SVD transformer for dimension alignment
        self.svd_transformer: TruncatedSVD = None
        
        logger.info(f"✅ ZebOS Commands Embedder initialized")
        logger.info(f"   Commands file: {commands_file}")
        logger.info(f"   Chapters file: {chapters_file}")
        logger.info(f"   Output dir: {output_dir}")
        logger.info(f"   Alpha (dense weight): {alpha}")
    
    def load_json_file(self, filepath: str) -> Any:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {filepath}: {len(data) if isinstance(data, list) else 1} items")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading {filepath}: {e}")
            return None
    
    def chunk_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a searchable chunk from a command
        
        Args:
            command: Command dictionary
            
        Returns:
            Chunked document with metadata
        """
        # Build comprehensive text content
        parts = []
        
        # Command name (most important)
        if command.get('name'):
            parts.append(f"Command: {command['name']}")
        
        # Description
        if command.get('description'):
            parts.append(f"Description: {command['description']}")
        
        # Syntax
        if command.get('syntax'):
            syntax_str = '\n'.join(command['syntax'])
            parts.append(f"Syntax:\n{syntax_str}")
        
        # Parameters
        if command.get('parameters'):
            params_parts = []
            for param in command['parameters']:
                param_name = param.get('name', '')
                param_desc = param.get('description', '')
                if param_name and param_desc:
                    params_parts.append(f"  {param_name}: {param_desc}")
            if params_parts:
                parts.append(f"Parameters:\n" + '\n'.join(params_parts))
        
        # Mode
        if command.get('mode'):
            parts.append(f"Mode: {command['mode']}")
        
        # Examples
        if command.get('examples'):
            examples_str = '\n'.join(command['examples'])
            parts.append(f"Examples:\n{examples_str}")
        
        # Chapter context
        if command.get('chapter'):
            parts.append(f"Chapter: {command['chapter']}")
        
        # Join all parts
        content = '\n\n'.join(parts)
        
        # Create metadata
        metadata = {
            'type': 'command',
            'command_name': command.get('name', 'unknown'),
            'mode': command.get('mode', 'unknown'),
            'file_path': command.get('file_path', ''),
            'chapter': command.get('chapter', 'General'),
            'has_examples': bool(command.get('examples')),
            'syntax_count': len(command.get('syntax', [])),
            'param_count': len(command.get('parameters', []))
        }
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    def chunk_chapter(self, chapter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a searchable chunk from a chapter
        
        Args:
            chapter: Chapter dictionary
            
        Returns:
            Chunked document with metadata
        """
        parts = []
        
        # Chapter title and number
        if chapter.get('chapter_number'):
            parts.append(f"{chapter['chapter_number']}: {chapter.get('title', 'Untitled')}")
        elif chapter.get('title'):
            parts.append(f"Chapter: {chapter['title']}")
        
        # Introduction
        if chapter.get('introduction'):
            parts.append(f"Introduction: {chapter['introduction']}")
        
        # Commands list
        if chapter.get('commands'):
            commands_str = ', '.join(chapter['commands'][:50])  # Limit to first 50
            if len(chapter['commands']) > 50:
                commands_str += f"... and {len(chapter['commands']) - 50} more"
            parts.append(f"Commands ({len(chapter['commands'])}): {commands_str}")
        
        content = '\n\n'.join(parts)
        
        metadata = {
            'type': 'chapter',
            'title': chapter.get('title', 'unknown'),
            'chapter_number': chapter.get('chapter_number', ''),
            'file_path': chapter.get('file_path', ''),
            'command_count': len(chapter.get('commands', []))
        }
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    def embed_commands(self) -> int:
        """
        Embed all commands from zebos_commands.json
        
        Returns:
            Number of commands embedded
        """
        logger.info("=" * 70)
        logger.info("EMBEDDING ZEBOS COMMANDS")
        logger.info("=" * 70)
        
        # Load commands
        commands = self.load_json_file(self.commands_file)
        if not commands:
            logger.error("❌ Failed to load commands")
            return 0
        
        # Chunk all commands
        logger.info(f"📝 Chunking {len(commands)} commands...")
        for i, command in enumerate(commands):
            try:
                chunk = self.chunk_command(command)
                self.texts.append(chunk['content'])
                self.metadata_list.append(chunk['metadata'])
                
                if (i + 1) % 500 == 0:
                    logger.info(f"   Chunked {i + 1}/{len(commands)} commands...")
            except Exception as e:
                logger.warning(f"⚠️  Error chunking command {i}: {e}")
        
        logger.info(f"✅ Created {len(self.texts)} command chunks")
        return len(self.texts)
    
    def embed_chapters(self) -> int:
        """
        Embed all chapters from zebos_chapters.json
        
        Returns:
            Number of chapters embedded
        """
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING ZEBOS CHAPTERS")
        logger.info("=" * 70)
        
        # Load chapters
        chapters = self.load_json_file(self.chapters_file)
        if not chapters:
            logger.error("❌ Failed to load chapters")
            return 0
        
        # Chunk all chapters
        logger.info(f"📝 Chunking {len(chapters)} chapters...")
        chapter_count = 0
        for i, chapter in enumerate(chapters):
            try:
                chunk = self.chunk_chapter(chapter)
                self.texts.append(chunk['content'])
                self.metadata_list.append(chunk['metadata'])
                chapter_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Error chunking chapter {i}: {e}")
        
        logger.info(f"✅ Created {chapter_count} chapter chunks")
        return chapter_count
    
    def get_dense_embedding(self, text: str) -> np.ndarray:
        """Get dense embedding from Ollama API"""
        try:
            response = requests.post(
                EMBEDDING_API_URL,
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            
            if not embedding:
                logger.warning(f"⚠️  Empty embedding returned")
                return np.zeros(768)  # nomic-embed-text default
            
            # Track embedding dimension
            if self.dense_embedding_dim == 0:
                self.dense_embedding_dim = len(embedding)
                logger.info(f"📊 Dense embedding dimension: {self.dense_embedding_dim}")
            
            return np.array(embedding, dtype='float32')
        except Exception as e:
            logger.error(f"❌ Error getting dense embedding: {e}")
            return np.zeros(768)
    
    def create_sparse_embeddings(self) -> np.ndarray:
        """Create sparse (TF-IDF) embeddings for all texts"""
        logger.info("🔍 Creating sparse embeddings using TF-IDF...")
        sparse_matrix = self.tfidf_vectorizer.fit_transform(self.texts)
        return sparse_matrix.toarray().astype('float32')
    
    def create_hybrid_embeddings(self, dense_embs: np.ndarray, sparse_embs: np.ndarray) -> np.ndarray:
        """
        Combine dense and sparse embeddings with alpha weighting using SVD
        
        Args:
            dense_embs: Dense embeddings (n_samples x dense_dim)
            sparse_embs: Sparse embeddings (n_samples x sparse_dim)
            
        Returns:
            Hybrid embeddings (n_samples x dense_dim)
        """
        logger.info("🔗 Creating hybrid embeddings with SVD alignment...")
        
        # Normalize embeddings
        dense_embs_norm = normalize(dense_embs, axis=1, norm='l2')
        sparse_embs_norm = normalize(sparse_embs, axis=1, norm='l2')
        
        # Use SVD to reduce sparse embeddings to same dimension as dense
        n_components = min(self.dense_embedding_dim, sparse_embs.shape[1], sparse_embs.shape[0] - 1)
        self.svd_transformer = TruncatedSVD(n_components=n_components, random_state=42)
        
        logger.info(f"   Reducing sparse dim {sparse_embs.shape[1]} → {n_components} using SVD")
        sparse_embs_reduced = self.svd_transformer.fit_transform(sparse_embs_norm)
        
        # Pad sparse embeddings if needed
        if sparse_embs_reduced.shape[1] < dense_embs_norm.shape[1]:
            padding = np.zeros((sparse_embs_reduced.shape[0], 
                               dense_embs_norm.shape[1] - sparse_embs_reduced.shape[1]))
            sparse_embs_reduced = np.hstack([sparse_embs_reduced, padding])
        
        # Combine with alpha weighting
        hybrid_embs = self.alpha * dense_embs_norm + (1 - self.alpha) * sparse_embs_reduced
        
        # Final normalization
        hybrid_embs = normalize(hybrid_embs, axis=1, norm='l2').astype('float32')
        
        logger.info(f"✅ Created hybrid embeddings: {hybrid_embs.shape}")
        return hybrid_embs
    
    def save_to_disk(self, hybrid_embeddings: np.ndarray):
        """Save FAISS index, metadata, TF-IDF vectorizer, and SVD transformer"""
        logger.info("💾 Saving to disk...")
        
        # Create FAISS index
        dimension = hybrid_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(hybrid_embeddings)
        
        # Save FAISS index
        index_path = os.path.join(self.output_dir, "zebos_commands_index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"✅ Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata = {
            'config': {
                'embedding_model': self.embedding_model,
                'alpha': self.alpha,
                'dense_dim': self.dense_embedding_dim,
                'sparse_dim': 5000,
                'hybrid_dim': dimension,
                'total_docs': len(self.texts)
            },
            'texts': self.texts,
            'metadata': self.metadata_list
        }
        
        metadata_path = os.path.join(self.output_dir, "zebos_commands_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved metadata to {metadata_path}")
        
        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(self.output_dir, "tfidf_vectorizer.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        logger.info(f"✅ Saved TF-IDF vectorizer to {tfidf_path}")
        
        # Save SVD transformer
        svd_path = os.path.join(self.output_dir, "svd_transformer.pkl")
        with open(svd_path, 'wb') as f:
            pickle.dump(self.svd_transformer, f)
        logger.info(f"✅ Saved SVD transformer to {svd_path}")
    
    def embed_all(self) -> Dict[str, int]:
        """
        Embed both commands and chapters using hybrid embeddings
        
        Returns:
            Dictionary with counts
        """
        results = {
            'commands': 0,
            'chapters': 0,
            'total': 0
        }
        
        # Chunk commands
        results['commands'] = self.embed_commands()
        
        # Chunk chapters
        results['chapters'] = self.embed_chapters()
        
        if not self.texts:
            logger.error("❌ No texts to embed!")
            return results
        
        # Create dense embeddings
        logger.info(f"\n🔄 Creating dense embeddings for {len(self.texts)} documents...")
        for i, text in enumerate(self.texts):
            if (i + 1) % 500 == 0:
                logger.info(f"   Embedded {i + 1}/{len(self.texts)} documents...")
            
            dense_emb = self.get_dense_embedding(text)
            self.dense_embeddings.append(dense_emb)
        
        dense_embs = np.array(self.dense_embeddings, dtype='float32')
        logger.info(f"✅ Created dense embeddings: {dense_embs.shape}")
        
        # Create sparse embeddings
        sparse_embs = self.create_sparse_embeddings()
        logger.info(f"✅ Created sparse embeddings: {sparse_embs.shape}")
        
        # Create hybrid embeddings
        hybrid_embs = self.create_hybrid_embeddings(dense_embs, sparse_embs)
        
        # Save to disk
        self.save_to_disk(hybrid_embs)
        
        # Calculate total
        results['total'] = len(self.texts)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Commands embedded: {results['commands']}")
        logger.info(f"Chapters embedded: {results['chapters']}")
        logger.info(f"Total documents: {results['total']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)
        
        return results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Embed ZebOS commands and chapters using hybrid embeddings"
    )
    parser.add_argument(
        '--commands',
        default='zebos_commands.json',
        help='Path to zebos_commands.json'
    )
    parser.add_argument(
        '--chapters',
        default='zebos_chapters.json',
        help='Path to zebos_chapters.json'
    )
    parser.add_argument(
        '--output',
        default='database/commands',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.7,
        help='Weight for dense embeddings (0-1)'
    )
    
    args = parser.parse_args()
    
    # Create embedder
    embedder = ZebOSCommandsEmbedder(
        commands_file=args.commands,
        chapters_file=args.chapters,
        output_dir=args.output,
        alpha=args.alpha
    )
    
    # Embed all documents
    results = embedder.embed_all()
    
    return 0 if results['total'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
