#!/usr/bin/env python3
"""
Test ZebOS Commands Integration with RAG Pipeline
=================================================

This script tests:
1. Embedding ZebOS commands and chapters
2. Multi-index retrieval (main docs + commands)
3. Full RAG pipeline with commands integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_embeddings():
    """Test that embeddings exist"""
    print("=" * 70)
    print("TEST 1: Check ZebOS Commands Embeddings")
    print("=" * 70)
    
    commands_dir = Path("database/commands")
    
    if not commands_dir.exists():
        print("❌ Commands database directory not found!")
        print(f"   Expected: {commands_dir.absolute()}")
        print("\n💡 Run: ./run_embed_commands.sh")
        return False
    
    # Check for index files
    index_file = commands_dir / "zebos_commands_index.faiss"
    metadata_file = commands_dir / "zebos_commands_metadata.json"
    
    if index_file.exists():
        print(f"✅ Found FAISS index: {index_file}")
    else:
        print(f"❌ FAISS index not found: {index_file}")
        return False
    
    if metadata_file.exists():
        print(f"✅ Found metadata: {metadata_file}")
    else:
        print(f"❌ Metadata not found: {metadata_file}")
        return False
    
    return True


def test_multi_index_retriever():
    """Test multi-index retriever"""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Index Retriever")
    print("=" * 70)
    
    try:
        from agent.multi_index_retriever import MultiIndexRetriever
        
        retriever = MultiIndexRetriever()
        print("✅ Multi-index retriever initialized")
        
        # Test query
        test_query = "show BGP neighbor commands"
        print(f"\n🔍 Testing query: '{test_query}'")
        
        results = retriever.search(test_query, top_k=3)
        print(f"✅ Retrieved {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result.get('source', 'unknown')}")
            print(f"Score: {result.get('score', 0):.3f}")
            
            metadata = result.get('metadata', {})
            if metadata.get('type') == 'command':
                print(f"Command: {metadata.get('command_name', 'N/A')}")
            
            content = result.get('content', '')[:150]
            print(f"Content: {content}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline_with_commands():
    """Test full RAG pipeline with commands integration"""
    print("\n" + "=" * 70)
    print("TEST 3: RAG Pipeline with Commands Integration")
    print("=" * 70)
    
    try:
        from rag.pipeline import RAGPipeline
        
        # Initialize pipeline with multi-index enabled
        pipeline = RAGPipeline(
            enable_multi_index=True,
            commands_index_dir="database/commands",
            commands_weight=0.4,
            retriever_top_k=10,
            reranker_top_k=5,
            enable_cache=False  # Disable cache for testing
        )
        print("✅ RAG Pipeline initialized with multi-index support")
        
        # Test queries
        test_queries = [
            "How do I configure OSPF on a ZebOS router?",
            "What is the syntax for the BGP neighbor command?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            result = pipeline.query(
                question=query,
                return_context=True,
                return_sources=True
            )
            
            print(f"\n📊 Results:")
            print(f"  - From cache: {result.get('from_cache', False)}")
            print(f"  - Time: {result.get('elapsed_time', 0):.2f}s")
            print(f"  - Sources: {len(result.get('sources', []))}")
            
            answer = result.get('answer', '')
            print(f"\n💬 Answer (first 300 chars):\n{answer[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("ZebOS Commands Integration Test Suite")
    print("🧪" * 35 + "\n")
    
    tests = [
        ("Embeddings Check", test_embeddings),
        ("Multi-Index Retriever", test_multi_index_retriever),
        ("RAG Pipeline Integration", test_rag_pipeline_with_commands),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    print("=" * 70)
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✨ ZebOS Commands Integration is working!")
        print("\nYou can now:")
        print("  • Query ZebOS commands through the RAG pipeline")
        print("  • Get command syntax, examples, and documentation")
        print("  • Search both main docs and commands database")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        
        # Give helpful hints
        if not results[0][1]:  # Embeddings check failed
            print("\n💡 To fix: Run the embedding script first:")
            print("   ./run_embed_commands.sh")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
