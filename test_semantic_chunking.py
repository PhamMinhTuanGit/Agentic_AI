#!/usr/bin/env python3
"""
Demo script để test Semantic Chunking với Cosine Similarity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest.embedder import SemanticChunker

def test_cosine_similarity():
    """Test cosine similarity calculation between sentences"""
    print("🧮 Testing Cosine Similarity Calculation")
    print("=" * 50)
    
    chunker = SemanticChunker()
    
    # Test sentence pairs
    test_pairs = [
        ("Machine learning is powerful.", "Deep learning uses neural networks."),
        ("Machine learning is powerful.", "Machine learning algorithms are effective."),
        ("Computer vision analyzes images.", "Natural language processing handles text."),
        ("Computer vision analyzes images.", "Image analysis is a computer vision task."),
        ("The weather is sunny today.", "Machine learning requires data."),
    ]
    
    # Fit the vectorizer with all sentences first
    all_sentences = []
    for sent1, sent2 in test_pairs:
        all_sentences.extend([sent1, sent2])
    
    try:
        chunker.sentence_vectorizer.fit(all_sentences)
        chunker._is_fitted = True
        print("✅ TF-IDF vectorizer fitted successfully")
    except Exception as e:
        print(f"❌ Error fitting vectorizer: {e}")
        return
    
    print("\nSimilarity scores:")
    print("-" * 30)
    
    for i, (sent1, sent2) in enumerate(test_pairs, 1):
        similarity = chunker.calculate_similarity(sent1, sent2)
        print(f"{i}. Similarity: {similarity:.4f}")
        print(f"   Sentence 1: '{sent1}'")
        print(f"   Sentence 2: '{sent2}'")
        print()

def test_semantic_chunking():
    print("🧪 Testing Semantic Chunking with Cosine Similarity")
    print("=" * 50)
    
    # Sample text for testing
    sample_text = """
    Machine learning is a subset of artificial intelligence. It involves training algorithms on data.
    The goal is to make predictions or decisions without being explicitly programmed.
    
    Deep learning is a specific type of machine learning. It uses neural networks with multiple layers.
    These networks can learn complex patterns in data. They are particularly effective for image and text processing.
    
    Natural language processing is another AI field. It focuses on understanding human language.
    NLP techniques are used in chatbots and translation systems. They help computers comprehend text and speech.
    
    Computer vision deals with image analysis. It enables machines to interpret visual information.
    Applications include facial recognition and autonomous vehicles. The field combines AI with image processing techniques.
    """
    
    # Test different similarity thresholds
    thresholds = [0.2, 0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n🔍 Testing with cosine similarity threshold = {threshold}")
        print("-" * 40)
        
        chunker = SemanticChunker(
            max_chunk_size=300,
            min_chunk_size=80,
            similarity_threshold=threshold
        )
        
        chunks = chunker.create_semantic_chunks(sample_text)
        
        print(f"Number of chunks: {len(chunks)}")
        
        # Show similarity scores between consecutive chunks
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i} ({len(chunk)} chars):")
            print(f"'{chunk[:150]}{'...' if len(chunk) > 150 else ''}'")
            
            # Calculate similarity with next chunk
            if i < len(chunks):
                next_chunk = chunks[i]
                similarity = chunker.calculate_similarity(chunk, next_chunk)
                print(f"  → Similarity with next chunk: {similarity:.3f}")
    
    print("\n✅ Cosine similarity chunking test completed!")

if __name__ == "__main__":
    test_cosine_similarity()
    print("\n" + "="*60 + "\n")
    test_semantic_chunking()