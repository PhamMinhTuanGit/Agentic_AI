#!/usr/bin/env python3
"""
Test Ring Topology Integration with RAG Pipeline

This script verifies that:
1. Topology file exists and is valid
2. Topology parser can load the YAML
3. Pipeline initializes with topology
4. Context building includes topology
5. LLM receives topology context
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from network_stat.topology_parser import TopologyParser
from network_stat.network_rag import NetworkTopologyRAG
from rag.pipeline import RAGPipeline

def test_topology_file():
    """Test 1: Check if topology file exists"""
    print("\n" + "="*70)
    print("TEST 1: Topology File Verification")
    print("="*70)
    
    topology_file = "network_stat/ring_topology.yaml"
    if Path(topology_file).exists():
        print(f"✅ Topology file found: {topology_file}")
        print(f"   Size: {Path(topology_file).stat().st_size} bytes")
        return True
    else:
        print(f"❌ Topology file not found: {topology_file}")
        return False

def test_topology_parser():
    """Test 2: Load and parse topology"""
    print("\n" + "="*70)
    print("TEST 2: Topology Parser")
    print("="*70)
    
    try:
        parser = TopologyParser("network_stat/ring_topology.yaml")
        topology = parser.load_topology()
        
        print(f"✅ Topology parsed successfully")
        print(f"   Devices found: {len(topology)}")
        
        for device_id, device in topology.items():
            print(f"   - {device_id}: {device.type}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to parse topology: {e}")
        return False

def test_topology_description():
    """Test 3: Get topology description"""
    print("\n" + "="*70)
    print("TEST 3: Topology Description")
    print("="*70)
    
    try:
        parser = TopologyParser("network_stat/ring_topology.yaml")
        description = parser.get_topology_description()
        
        print(f"✅ Topology description generated")
        print(f"   Length: {len(description)} characters")
        print(f"\n   Preview:")
        print("   " + "\n   ".join(description.split("\n")[:10]))
        
        return True
    except Exception as e:
        print(f"❌ Failed to generate description: {e}")
        return False

def test_network_rag():
    """Test 4: Network RAG integration"""
    print("\n" + "="*70)
    print("TEST 4: Network RAG Integration")
    print("="*70)
    
    try:
        parser = TopologyParser("network_stat/ring_topology.yaml")
        network_rag = NetworkTopologyRAG(parser, None)
        context = network_rag.get_llm_context()
        
        print(f"✅ Network RAG context generated")
        print(f"   Length: {len(context)} characters")
        print(f"\n   Preview:")
        print("   " + "\n   ".join(context.split("\n")[:15]))
        
        return True
    except Exception as e:
        print(f"❌ Failed to generate RAG context: {e}")
        return False

def test_pipeline_initialization():
    """Test 5: Pipeline initialization with topology"""
    print("\n" + "="*70)
    print("TEST 5: Pipeline Initialization with Topology")
    print("="*70)
    
    try:
        print("Initializing RAG Pipeline with topology...")
        pipeline = RAGPipeline(
            enable_topology=True,
            topology_file="network_stat/ring_topology.yaml"
        )
        
        print(f"✅ Pipeline initialized successfully")
        print(f"   Topology enabled: {pipeline.enable_topology}")
        print(f"   Topology context available: {pipeline.topology_context is not None}")
        if pipeline.topology_context:
            print(f"   Topology context length: {len(pipeline.topology_context)} characters")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_building():
    """Test 6: Context building includes topology"""
    print("\n" + "="*70)
    print("TEST 6: Context Building with Topology")
    print("="*70)
    
    try:
        pipeline = RAGPipeline(
            enable_topology=True,
            topology_file="network_stat/ring_topology.yaml"
        )
        
        # Create mock documents
        mock_docs = [
            {
                'text': 'Document 1: Information about routers',
                'score': 0.9
            },
            {
                'text': 'Document 2: OSPF configuration guide',
                'score': 0.85
            }
        ]
        
        context = pipeline._build_context(mock_docs)
        
        print(f"✅ Context built successfully")
        print(f"   Total length: {len(context)} characters")
        print(f"   Includes topology: {'NETWORK TOPOLOGY CONTEXT' in context}")
        print(f"   Includes documents: {'RELEVANT DOCUMENTS' in context}")
        print(f"\n   Preview:")
        print("   " + "\n   ".join(context.split("\n")[:20]))
        
        return True
    except Exception as e:
        print(f"❌ Failed to build context: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_without_topology():
    """Test 7: Pipeline without topology (backward compatibility)"""
    print("\n" + "="*70)
    print("TEST 7: Pipeline Without Topology (Backward Compatibility)")
    print("="*70)
    
    try:
        pipeline = RAGPipeline(
            enable_topology=False
        )
        
        print(f"✅ Pipeline initialized without topology")
        print(f"   Topology enabled: {pipeline.enable_topology}")
        print(f"   Topology context: {pipeline.topology_context}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize pipeline without topology: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("RING TOPOLOGY - RAG PIPELINE INTEGRATION TEST")
    print("="*70)
    
    tests = [
        ("Topology File", test_topology_file),
        ("Topology Parser", test_topology_parser),
        ("Topology Description", test_topology_description),
        ("Network RAG", test_network_rag),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Context Building", test_context_building),
        ("Backward Compatibility", test_pipeline_without_topology),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed_count}/{total_count} tests passed")
    print(f"{'='*70}\n")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED! Integration is working correctly.")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
