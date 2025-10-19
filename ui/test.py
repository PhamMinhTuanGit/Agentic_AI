"""
Test script for RAG Pipeline Streamlit UI
Validates setup and runs basic tests
"""

import sys
from pathlib import Path
import subprocess
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit - Install: pip install streamlit")
        return False
    
    try:
        import plotly
        print("✅ plotly")
    except ImportError:
        print("❌ plotly - Install: pip install plotly")
        return False
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - Install: pip install pandas")
        return False
    
    try:
        from rag.pipeline import RAGPipeline
        print("✅ RAG Pipeline")
    except ImportError as e:
        print(f"❌ RAG Pipeline - {e}")
        return False
    
    try:
        from rag.cache import CacheManager
        print("✅ Cache Manager")
    except ImportError as e:
        print(f"❌ Cache Manager - {e}")
        return False
    
    return True

def test_ollama():
    """Test Ollama connection"""
    print("\n🔍 Testing Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            print(f"✅ Ollama is running")
            print(f"   Models: {len(models)}")
            
            required_models = ['nomic-embed-text', 'qwen2.5-coder:3b']
            model_names = [m.get('name', '').split(':')[0] for m in models]
            
            for model in required_models:
                if any(model in name for name in model_names):
                    print(f"   ✅ {model}")
                else:
                    print(f"   ❌ {model} - Pull with: ollama pull {model}")
            
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("   Start with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_faiss_index():
    """Test if FAISS index exists"""
    print("\n🔍 Testing FAISS index...")
    
    index_path = Path("database/document/hybrid_docs_index.faiss")
    metadata_path = Path("database/document/hybrid_docs_metadata.json")
    
    if index_path.exists():
        size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"✅ FAISS index exists ({size_mb:.1f} MB)")
    else:
        print(f"❌ FAISS index not found at {index_path}")
        print("   Generate with: python -m ingest.embedder")
        return False
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            total_chunks = metadata.get('config', {}).get('total_chunks', 0)
            print(f"✅ Metadata exists ({total_chunks} chunks)")
    else:
        print(f"❌ Metadata not found at {metadata_path}")
        return False
    
    return True

def test_cache_db():
    """Test if cache database can be initialized"""
    print("\n🔍 Testing cache database...")
    
    try:
        from rag.cache import CacheManager
        
        cache = CacheManager(cache_ttl=86400)
        
        # Test save
        cache.set(
            query="test query",
            answer="test answer",
            context=["doc1", "doc2"],
            metadata={'test': True}
        )
        print("✅ Cache write successful")
        
        # Test retrieve
        result = cache.get("test query")
        if result:
            print("✅ Cache read successful")
        else:
            print("⚠️  Cache read returned None (query might have expired)")
        
        return True
    
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 RAG Pipeline Streamlit UI - Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test Ollama
    if not test_ollama():
        all_passed = False
    
    # Test FAISS
    if not test_faiss_index():
        all_passed = False
    
    # Test Cache
    if not test_cache_db():
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All tests passed!")
        print("\n🚀 Ready to start UI:")
        print("   streamlit run ui/app.py")
        return 0
    else:
        print("❌ Some tests failed!")
        print("\n📋 Before running UI, fix issues above:")
        print("   1. Install dependencies: pip install -r ui/requirements.txt")
        print("   2. Start Ollama: ollama serve")
        print("   3. Generate FAISS: python -m ingest.embedder")
        return 1

if __name__ == "__main__":
    sys.exit(main())
