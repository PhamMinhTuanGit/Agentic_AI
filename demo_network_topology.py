#!/usr/bin/env python3
"""
Network Topology LLM Integration - Demo/Test Script
Shows how to use the network topology system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from network_stat.topology_parser import TopologyParser
from network_stat.cli_generator import CLIGenerator
from network_stat.network_rag import NetworkTopologyRAG, NetworkConfigRequest


def print_section(title):
    """Print formatted section"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_parser():
    """Demo: Topology Parser"""
    print_section("1. TOPOLOGY PARSER DEMO")
    
    parser = TopologyParser("network_stat/topo.yaml")
    
    # Show topology description
    print("Topology Description:")
    print(parser.get_topology_description())
    
    # Show network map
    print("\nNetwork Map:")
    print(parser.get_network_map())
    
    # Show device info
    print("\n\nDevice Information (SW1):")
    device = parser.get_device("SW1")
    if device:
        print(f"  ID: {device.id}")
        print(f"  Type: {device.type}")
        print(f"  IP: {device.ip}")
        print(f"  Interfaces: {device.interfaces}")
    
    # Show all switches
    print("\n\nAll Switches:")
    switches = parser.get_devices_by_type("switch")
    for sw in switches:
        print(f"  - {sw.id}: {sw.ip}")


def demo_cli_generator():
    """Demo: CLI Generator"""
    print_section("2. CLI GENERATOR DEMO")
    
    generator = CLIGenerator("network_stat/topo.yaml")
    
    # Get switch configuration
    print("Switch Configuration (SW1):")
    sw_config = generator.get_device_config("SW1")
    print(sw_config)
    
    # Get summary
    print("\n\nConfiguration Summary Guide:")
    print(generator.get_summary_guide())


def demo_network_rag():
    """Demo: Network RAG System"""
    print_section("3. NETWORK RAG DEMO")
    
    rag = NetworkTopologyRAG("network_stat/topo.yaml")
    
    # Show topology context
    print("Topology Context for LLM (first 500 chars):")
    context = rag.get_llm_context()
    print(context[:500])
    print("...")
    
    # Process different requests
    print("\n\n--- Request 1: Configure specific device ---")
    req1 = NetworkConfigRequest(
        device_id="Router1",
        action="configure",
        query="Configure Router1"
    )
    result1 = rag.process_configuration_request(req1)
    print(f"Status: {result1['status']}")
    print(f"Device: {result1['data']['device_info']['id']}")
    print(f"Config Preview (first 200 chars):")
    print(result1['data']['configuration'][:200])
    print("...")
    
    # Request 2: Get all routers
    print("\n\n--- Request 2: Get all routers ---")
    req2 = NetworkConfigRequest(
        device_type="router",
        action="configure",
        query="Show all router configurations"
    )
    result2 = rag.process_configuration_request(req2)
    print(f"Status: {result2['status']}")
    print(f"Router count: {result2['data']['count']}")
    print(f"Routers: {list(result2['data']['configurations'].keys())}")
    
    # Request 3: General topology
    print("\n\n--- Request 3: General topology ---")
    req3 = NetworkConfigRequest(
        action="verify",
        query="Show network topology"
    )
    result3 = rag.process_configuration_request(req3)
    print(f"Status: {result3['status']}")
    print(f"Topology summary (first 200 chars):")
    print(result3['data']['topology_summary'][:200])
    print("...")


def demo_llm_prompt():
    """Demo: LLM Prompt Generation"""
    print_section("4. LLM PROMPT GENERATION DEMO")
    
    rag = NetworkTopologyRAG("network_stat/topo.yaml")
    
    # Create a request
    request = NetworkConfigRequest(
        device_id="SW1",
        action="configure",
        query="How do I configure VLAN 10 on SW1?"
    )
    
    # Generate LLM prompt
    prompt = rag.generate_llm_prompt(request)
    
    print("Generated LLM Prompt (for 'How do I configure VLAN 10 on SW1?'):")
    print("\n" + prompt)
    print("\n[This prompt would be sent to the LLM with full topology context]")


def demo_api_simulation():
    """Demo: Simulating API requests"""
    print_section("5. API SIMULATION DEMO")
    
    print("The following would be actual HTTP requests:\n")
    
    print("1. GET /network/topology")
    print("   Returns: Complete topology and all devices\n")
    
    print("2. GET /network/device/SW1")
    print("   Returns: Device info and CLI commands for SW1\n")
    
    print("3. GET /network/devices?device_type=router")
    print("   Returns: All router devices and their configs\n")
    
    print("4. POST /network/configure")
    print('   Body: {"device_id": "Router1", "action": "configure"}')
    print("   Returns: Configuration and LLM assistance prompt\n")
    
    print("5. POST /network/query")
    print('   Body: {"query": "How to set up OSPF?", "model": "llama3.1:8b"}')
    print("   Returns: LLM answer with topology context\n")
    
    print("6. GET /network/context")
    print("   Returns: Full topology context for embedding\n")


def demo_device_connections():
    """Demo: Device connection analysis"""
    print_section("6. DEVICE CONNECTION ANALYSIS")
    
    parser = TopologyParser("network_stat/topo.yaml")
    
    # Show all devices and their connections
    print("All Device Connections:\n")
    
    for device_id in parser.get_all_devices().keys():
        connections = parser.get_device_connections(device_id)
        print(f"{device_id} ({connections['device_type']}):")
        for conn in connections['connections']:
            print(f"  {conn['port']} → {conn['connected_to']}")
        print()


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "NETWORK TOPOLOGY LLM INTEGRATION DEMO" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        demo_parser()
        demo_cli_generator()
        demo_network_rag()
        demo_llm_prompt()
        demo_device_connections()
        demo_api_simulation()
        
        print_section("DEMO COMPLETE")
        print("✅ All components working successfully!\n")
        print("Next Steps:")
        print("  1. Start the backend: docker-compose up -d")
        print("  2. Query the API:")
        print("     curl http://localhost:8000/network/topology")
        print("     curl http://localhost:8000/network/device/SW1")
        print("  3. Ask questions:")
        print('     curl -X POST http://localhost:8000/network/query \\')
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"query": "What devices are in this network?"}\'')
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
