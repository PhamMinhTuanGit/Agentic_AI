#!/usr/bin/env python3
"""
Network Topology LLM Integration - Testing Guide
Comprehensive examples and test cases
"""

# =============================================================================
# SETUP
# =============================================================================

"""
Before running tests, ensure:
1. Backend is running: docker-compose up -d
2. Or: cd backend && python3 main.py
3. Ollama is running with model: ollama pull llama3.1:8b
"""

import requests
import json
from typing import Dict, Any

# Configuration
BACKEND_URL = "http://localhost:8000"
MODEL = "llama3.1:8b"  # or "tinyllama", "mistral", etc.


# =============================================================================
# TEST UTILITIES
# =============================================================================

def print_response(title: str, response: Dict[str, Any], max_chars: int = 500) -> None:
    """Pretty print response"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")
    
    if isinstance(response, dict):
        for key, value in response.items():
            if isinstance(value, str) and len(str(value)) > max_chars:
                print(f"{key}:")
                print(f"  {str(value)[:max_chars]}...")
                print()
            else:
                print(f"{key}: {value}")
    else:
        print(response)


def make_request(method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make HTTP request"""
    url = f"{BACKEND_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# =============================================================================
# TEST CASES
# =============================================================================

def test_1_get_topology():
    """Test 1: Get complete topology"""
    print("\n" + "█"*70)
    print("  TEST 1: Get Network Topology")
    print("█"*70)
    
    response = make_request("GET", "/network/topology")
    print_response("Response", response, max_chars=300)
    
    assert "all_devices" in response
    assert "topology_summary" in response
    print("✅ PASSED")


def test_2_get_device_config():
    """Test 2: Get specific device configuration"""
    print("\n" + "█"*70)
    print("  TEST 2: Get Device Configuration (SW1)")
    print("█"*70)
    
    response = make_request("GET", "/network/device/SW1")
    print_response("SW1 Configuration", response)
    
    assert "device_info" in response
    assert "cli_commands" in response
    assert response["device_info"]["id"] == "SW1"
    print("✅ PASSED")


def test_3_get_router_config():
    """Test 3: Get router configuration"""
    print("\n" + "█"*70)
    print("  TEST 3: Get Router Configuration (Router1)")
    print("█"*70)
    
    response = make_request("GET", "/network/device/Router1")
    print_response("Router1 Configuration", response)
    
    assert response["device_info"]["type"] == "router"
    print("✅ PASSED")


def test_4_list_all_devices():
    """Test 4: List all devices"""
    print("\n" + "█"*70)
    print("  TEST 4: List All Devices")
    print("█"*70)
    
    response = make_request("GET", "/network/devices")
    print_response("All Devices", response, max_chars=200)
    
    assert "total_count" in response
    assert "devices" in response
    print(f"✅ PASSED - Found {response['total_count']} devices")


def test_5_filter_by_type():
    """Test 5: Filter devices by type"""
    print("\n" + "█"*70)
    print("  TEST 5: Filter Devices by Type (routers)")
    print("█"*70)
    
    response = make_request("GET", "/network/devices?device_type=router")
    print_response("All Routers", response)
    
    assert response["type"] == "router"
    assert "count" in response
    print(f"✅ PASSED - Found {response['count']} routers")


def test_6_filter_hosts():
    """Test 6: Filter hosts"""
    print("\n" + "█"*70)
    print("  TEST 6: Filter Devices by Type (hosts)")
    print("█"*70)
    
    response = make_request("GET", "/network/devices?device_type=host")
    print_response("All Hosts", response)
    
    assert response["type"] == "host"
    print(f"✅ PASSED - Found {response['count']} hosts")


def test_7_get_network_context():
    """Test 7: Get network context"""
    print("\n" + "█"*70)
    print("  TEST 7: Get Network Context (for LLM)")
    print("█"*70)
    
    response = make_request("GET", "/network/context")
    print_response("Network Context Info", response)
    
    assert "context" in response
    assert "devices_count" in response
    print(f"✅ PASSED - Context size: {len(response['context'])} chars")


def test_8_configure_device():
    """Test 8: Generate device configuration"""
    print("\n" + "█"*70)
    print("  TEST 8: Configure Device (generate CLI)")
    print("█"*70)
    
    request_data = {
        "device_id": "Router1",
        "device_type": None,
        "action": "configure",
        "query": "Generate Router1 configuration"
    }
    
    response = make_request("POST", "/network/configure", request_data)
    print_response("Router1 Generated Config", response)
    
    assert "configuration" in response
    print("✅ PASSED")


def test_9_query_llm_simple():
    """Test 9: Query LLM about network"""
    print("\n" + "█"*70)
    print("  TEST 9: Query LLM - Simple Question")
    print("█"*70)
    
    request_data = {
        "query": "What devices exist in this network?",
        "model": MODEL,
        "max_tokens": 256
    }
    
    print("Sending query to LLM...")
    response = make_request("POST", "/network/query", request_data)
    print_response("LLM Answer", response)
    
    assert "answer" in response
    assert response["topology_context_used"]
    print("✅ PASSED")


def test_10_query_llm_technical():
    """Test 10: Query LLM - Technical Question"""
    print("\n" + "█"*70)
    print("  TEST 10: Query LLM - Technical Question")
    print("█"*70)
    
    request_data = {
        "query": "What are the IP addresses of all network devices?",
        "model": MODEL,
        "max_tokens": 512
    }
    
    print("Sending query to LLM...")
    response = make_request("POST", "/network/query", request_data)
    print_response("LLM Answer", response)
    
    assert "answer" in response
    print("✅ PASSED")


def test_11_query_llm_configuration():
    """Test 11: Query LLM - Configuration Question"""
    print("\n" + "█"*70)
    print("  TEST 11: Query LLM - Configuration Steps")
    print("█"*70)
    
    request_data = {
        "query": "Give me step-by-step instructions to configure this network",
        "model": MODEL,
        "max_tokens": 1024
    }
    
    print("Sending query to LLM (may take longer)...")
    response = make_request("POST", "/network/query", request_data)
    print_response("LLM Configuration Guide", response, max_chars=800)
    
    assert "answer" in response
    print("✅ PASSED")


def test_12_invalid_device():
    """Test 12: Error handling - Invalid device"""
    print("\n" + "█"*70)
    print("  TEST 12: Error Handling - Invalid Device")
    print("█"*70)
    
    response = make_request("GET", "/network/device/INVALID_DEVICE")
    print_response("Error Response", response)
    
    # Could be error or 404, both are valid
    if "error" in response:
        print("✅ PASSED - Error handled correctly")
    else:
        print("⚠️  No error returned (may be expected)")


# =============================================================================
# CURL EXAMPLES
# =============================================================================

CURL_EXAMPLES = """
# ============================================================================
# CURL COMMAND EXAMPLES (Run these directly)
# ============================================================================

# 1. Get topology
curl http://localhost:8000/network/topology

# 2. Get SW1 configuration
curl http://localhost:8000/network/device/SW1

# 3. Get Router1 configuration
curl http://localhost:8000/network/device/Router1

# 4. List all devices
curl http://localhost:8000/network/devices

# 5. List only routers
curl "http://localhost:8000/network/devices?device_type=router"

# 6. List only hosts
curl "http://localhost:8000/network/devices?device_type=host"

# 7. Get network context
curl http://localhost:8000/network/context

# 8. Configure a device
curl -X POST http://localhost:8000/network/configure \\
  -H "Content-Type: application/json" \\
  -d '{
    "device_id": "SW1",
    "action": "configure",
    "query": "Generate configuration for SW1"
  }'

# 9. Ask LLM simple question
curl -X POST http://localhost:8000/network/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What is the network topology?",
    "model": "llama3.1:8b"
  }'

# 10. Ask LLM configuration question
curl -X POST http://localhost:8000/network/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "How do I configure OSPF on Router1?",
    "model": "llama3.1:8b",
    "max_tokens": 512
  }'

# 11. Ask about device connections
curl -X POST http://localhost:8000/network/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What is connected to SW1 and on which ports?",
    "model": "llama3.1:8b"
  }'

# 12. Ask for troubleshooting help
curl -X POST http://localhost:8000/network/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "PC1 cannot reach the internet. What could be the issue?",
    "model": "llama3.1:8b",
    "max_tokens": 512
  }'
"""


# =============================================================================
# PYTHON SDK EXAMPLES
# =============================================================================

PYTHON_EXAMPLES = """
# ============================================================================
# PYTHON SDK EXAMPLES
# ============================================================================

from network_stat.network_rag import NetworkTopologyRAG, NetworkConfigRequest

# Initialize
rag = NetworkTopologyRAG("network_stat/topo.yaml")

# Example 1: Get topology description
print(rag.parser.get_topology_description())

# Example 2: Get specific device info
device_info = rag.get_device_info("SW1")
print(device_info)

# Example 3: Get device configuration
config = rag.get_device_config("Router1")
print(config)

# Example 4: Get all routers
routers = rag.get_devices_by_type("router")
print(f"Found {len(routers)} routers")

# Example 5: Process configuration request
request = NetworkConfigRequest(
    device_id="SW1",
    action="configure",
    query="Configure SW1"
)
result = rag.process_configuration_request(request)
print(result)

# Example 6: Generate LLM prompt
prompt = rag.generate_llm_prompt(request)
print(prompt)

# Example 7: Get full context
context = rag.get_llm_context()
print(f"Context size: {len(context)} characters")
"""


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "NETWORK TOPOLOGY LLM INTEGRATION" + " "*21 + "║")
    print("║" + " "*23 + "COMPREHENSIVE TEST SUITE" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        test_1_get_topology,
        test_2_get_device_config,
        test_3_get_router_config,
        test_4_list_all_devices,
        test_5_filter_by_type,
        test_6_filter_hosts,
        test_7_get_network_context,
        test_8_configure_device,
        test_9_query_llm_simple,
        test_10_query_llm_technical,
        test_11_query_llm_configuration,
        test_12_invalid_device,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR - {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    print("="*70 + "\n")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!\n")
    else:
        print(f"⚠️  {failed} test(s) failed\n")
    
    return failed == 0


# =============================================================================
# DISPLAY EXAMPLES
# =============================================================================

def show_examples():
    """Show example commands"""
    print("\n" + "="*70)
    print("  CURL EXAMPLES")
    print("="*70)
    print(CURL_EXAMPLES)
    
    print("\n" + "="*70)
    print("  PYTHON SDK EXAMPLES")
    print("="*70)
    print(PYTHON_EXAMPLES)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            show_examples()
        elif sys.argv[1] == "--curl":
            print(CURL_EXAMPLES)
        elif sys.argv[1] == "--python":
            print(PYTHON_EXAMPLES)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python test_network_topology.py [--examples|--curl|--python]")
    else:
        success = run_all_tests()
        
        print("\n" + "="*70)
        print("  NEXT STEPS")
        print("="*70)
        print("✅ Run tests with: python3 test_network_topology.py")
        print("✅ Show curl examples with: python3 test_network_topology.py --curl")
        print("✅ Show python examples with: python3 test_network_topology.py --python")
        print("✅ Show all examples with: python3 test_network_topology.py --examples")
        print("="*70 + "\n")
        
        sys.exit(0 if success else 1)
