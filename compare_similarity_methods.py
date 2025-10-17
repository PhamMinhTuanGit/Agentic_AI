#!/usr/bin/env python3
"""
Demo so sánh Cosine Similarity vs Word Intersection
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def word_intersection_similarity(sent1: str, sent2: str) -> float:
    """Calculate similarity using word intersection (old method)"""
    if not sent1.strip() or not sent2.strip():
        return 0.0
    
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def cosine_similarity_tfidf(sent1: str, sent2: str) -> float:
    """Calculate similarity using TF-IDF cosine similarity (new method)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if not sent1.strip() or not sent2.strip():
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        vectors = vectorizer.fit_transform([sent1, sent2])
        
        vec1 = vectors[0].toarray()[0]
        vec2 = vectors[1].toarray()[0]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except:
        return 0.0

def compare_methods():
    print("🔍 So sánh Cosine Similarity vs Word Intersection")
    print("=" * 60)
    
    # Test cases với độ khó tăng dần
    test_cases = [
        # Case 1: Identical sentences
        ("Machine learning is powerful", "Machine learning is powerful"),
        
        # Case 2: Similar meaning, different words
        ("Machine learning algorithms are effective", "ML models work well"),
        
        # Case 3: Same topic, different aspects
        ("Deep learning uses neural networks", "Neural networks enable deep learning"),
        
        # Case 4: Related topics
        ("Computer vision analyzes images", "Image processing techniques are important"),
        
        # Case 5: Different topics
        ("Weather is sunny today", "Machine learning requires data"),
        
        # Case 6: Partial overlap
        ("Natural language processing handles text", "Text processing is challenging"),
        
        # Case 7: Synonyms and paraphrases
        ("AI systems are intelligent", "Artificial intelligence is smart"),
    ]
    
    print(f"{'Case':<5} {'Intersection':<12} {'Cosine':<12} {'Description'}")
    print("-" * 60)
    
    for i, (sent1, sent2) in enumerate(test_cases, 1):
        intersection_sim = word_intersection_similarity(sent1, sent2)
        cosine_sim = cosine_similarity_tfidf(sent1, sent2)
        
        # Determine which method gives better intuitive result
        if abs(cosine_sim - intersection_sim) > 0.1:
            if cosine_sim > intersection_sim:
                better = "Cosine ✓"
            else:
                better = "Intersection ✓"
        else:
            better = "Similar"
        
        print(f"{i:<5} {intersection_sim:<12.3f} {cosine_sim:<12.3f} {better}")
        print(f"      '{sent1}'")
        print(f"      '{sent2}'")
        print()
    
    print("📊 Kết luận:")
    print("• Cosine Similarity: Tốt hơn với synonyms và paraphrases")
    print("• Word Intersection: Đơn giản nhưng kém chính xác với ngữ nghĩa")
    print("• TF-IDF Cosine: Xem xét trọng số từ, ít bị ảnh hưởng bởi stop words")

if __name__ == "__main__":
    compare_methods()