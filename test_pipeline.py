#!/usr/bin/env python3
"""
Demo script to test the Hybrid PDF Embedding Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest.embedder import HybridPDFEmbedder

def main():
    print("🧪 Testing Hybrid PDF Embedding Pipeline")
    print("=" * 50)
    
    # Check if documents exist
    docs_path = "./documents"
    if not os.path.exists(docs_path):
        print(f"❌ Documents folder not found: {docs_path}")
        return
    
    pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in {docs_path}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    # Initialize embedder with different alpha values for testing
    alpha_values = [0.7]  # You can test with [0.3, 0.5, 0.7, 0.9] for comparison
    
    for alpha in alpha_values:
        print(f"\n🔬 Testing with alpha = {alpha}")
        print("-" * 30)
        
        embedder = HybridPDFEmbedder(
            folder_path=docs_path,
            model="nomic-embed-text",
            chunk_size=800,
            min_chunk_size=200,
            alpha=alpha,
            similarity_threshold=0.5
        )
        
        try:
            # Process documents
            texts, hybrid_embeddings = embedder.process_documents()
            
            if len(hybrid_embeddings) > 0:
                # Save with alpha-specific names
                success = embedder.save_to_faiss(
                    faiss_index_path=f"./rag_backend/hybrid_docs_index_alpha_{alpha}.faiss",
                    metadata_path=f"./rag_backend/hybrid_docs_metadata_alpha_{alpha}.json",
                    tfidf_path=f"./rag_backend/tfidf_vectorizer_alpha_{alpha}.pkl"
                )
                
                if success:
                    stats = embedder.get_stats()
                    print(f"📊 Results for alpha = {alpha}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"❌ Failed to save results for alpha = {alpha}")
            else:
                print(f"❌ No embeddings created for alpha = {alpha}")
                
        except Exception as e:
            print(f"❌ Error processing with alpha = {alpha}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()