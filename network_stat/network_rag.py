"""
Network Topology RAG Integration
Integrates network topology into RAG pipeline for LLM-based configuration assistance
"""

import json
from typing import Optional, Dict, Any
from pydantic import BaseModel
from network_stat.topology_parser import TopologyParser
from network_stat.cli_generator import CLIGenerator


class NetworkConfigRequest(BaseModel):
    """Request for network device configuration"""
    device_id: Optional[str] = None  # Specific device
    device_type: Optional[str] = None  # All devices of type
    action: str = "configure"  # configure, status, verify, troubleshoot
    query: Optional[str] = None  # Natural language query


class NetworkTopologyRAG:
    """RAG system for network topology and configuration"""
    
    def __init__(self, topology_file: str = "network_stat/topo.yaml"):
        self.parser = TopologyParser(topology_file)
        self.generator = CLIGenerator(topology_file)
        self.context_cache = None
    
    def get_llm_context(self) -> str:
        """Get topology context for LLM"""
        if self.context_cache is None:
            self.context_cache = self._build_context()
        return self.context_cache
    
    def _build_context(self) -> str:
        """Build comprehensive topology context"""
        context = """
# NETWORK TOPOLOGY KNOWLEDGE BASE

## Topology Overview
"""
        context += self.parser.get_topology_description()
        context += self.parser.get_network_map()
        
        context += """

## Device Information
"""
        for device_id, device in self.parser.get_all_devices().items():
            context += f"""
### Device: {device_id}
- Type: {device.type}
- IP Address: {device.ip}
"""
            if device.mac:
                context += f"- MAC Address: {device.mac}\n"
            if device.ip_range:
                context += f"- IP Range: {device.ip_range}\n"
            
            if device.interfaces:
                context += "- Connected Interfaces:\n"
                for intf in device.interfaces:
                    port_name = intf.get('port', intf.get('name', 'unknown'))
                    context += f"  * {port_name} connected to {intf['connected_to']}\n"
        
        context += """

## Configuration Commands

### Switches
- Configure interfaces: `interface <port>`
- Set interface speed: `speed 1000`
- Configure VLAN: `vlan <number>`
- Configure spanning tree: `spanning-tree mode pvst`

### Routers
- Configure interface: `interface <port>`
- Set IP address: `ip address <ip> <subnet_mask>`
- Enable routing: `ip routing`
- Configure routing protocol: `router ospf <process_id>`

### Hosts
- Set IP address: `ip addr add <ip>/<prefix> dev <interface>`
- Configure gateway: `route add default gw <gateway_ip>`
- Verify connectivity: `ping <ip>`

## Network Services
- DHCP for automatic IP assignment
- DNS for hostname resolution
- VLAN for network segmentation
- OSPF for dynamic routing
- Spanning Tree for loop prevention

## Common Troubleshooting
- Use `show ip route` to verify routing
- Use `ping` to test connectivity
- Use `traceroute` to find path issues
- Use `show interface` to check port status
- Use `show vlan` to verify VLAN configuration
"""
        return context
    
    def get_device_config(self, device_id: str) -> str:
        """Get configuration for specific device"""
        return self.generator.get_device_config(device_id)
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get device information"""
        device = self.parser.get_device(device_id)
        if not device:
            return {"error": f"Device {device_id} not found"}
        
        return {
            "id": device.id,
            "type": device.type,
            "ip": device.ip,
            "mac": device.mac,
            "ip_range": device.ip_range,
            "interfaces": device.interfaces,
            "connections": self.parser.get_device_connections(device_id)
        }
    
    def get_devices_by_type(self, device_type: str) -> list:
        """Get all devices of specific type"""
        devices = self.parser.get_devices_by_type(device_type)
        return [d.to_dict() for d in devices]
    
    def get_topology_summary(self) -> str:
        """Get topology summary"""
        return self.generator.get_summary_guide()
    
    def process_configuration_request(self, request: NetworkConfigRequest) -> Dict[str, Any]:
        """Process configuration request"""
        result = {
            "status": "success",
            "request": request.dict(),
            "data": {}
        }
        
        if request.device_id:
            # Specific device
            device_info = self.get_device_info(request.device_id)
            if "error" not in device_info:
                result["data"]["device_info"] = device_info
                result["data"]["configuration"] = self.get_device_config(request.device_id)
            else:
                result["status"] = "error"
                result["data"] = device_info
        
        elif request.device_type:
            # All devices of type
            devices = self.get_devices_by_type(request.device_type)
            result["data"]["devices"] = devices
            result["data"]["count"] = len(devices)
            
            # Get configs for each
            configs = {}
            for device in devices:
                configs[device['id']] = self.get_device_config(device['id'])
            result["data"]["configurations"] = configs
        
        else:
            # General topology
            result["data"]["topology_summary"] = self.get_topology_summary()
            result["data"]["all_devices"] = {
                d_id: d.to_dict() 
                for d_id, d in self.parser.get_all_devices().items()
            }
        
        return result
    
    def generate_llm_prompt(self, request: NetworkConfigRequest) -> str:
        """Generate prompt for LLM based on request"""
        base_context = self.get_llm_context()
        
        prompt = f"""
Using the network topology knowledge provided below, help with the following request:

REQUEST: {request.query or f"Configure {request.device_id or request.device_type}"}

NETWORK TOPOLOGY CONTEXT:
{base_context}

INSTRUCTIONS:
1. Based on the topology, provide relevant CLI commands
2. Explain what each command does
3. List dependencies and prerequisites
4. Provide verification steps
5. Include troubleshooting tips if applicable
"""
        return prompt


# FastAPI integration functions
def create_network_rag_system(topology_file: str = "network_stat/topo.yaml") -> NetworkTopologyRAG:
    """Factory function to create RAG system"""
    return NetworkTopologyRAG(topology_file)


def get_topology_context_for_embedding() -> str:
    """Get topology context for document embedding"""
    rag = create_network_rag_system()
    return rag.get_llm_context()


if __name__ == "__main__":
    # Example usage
    rag = NetworkTopologyRAG()
    
    print("=" * 80)
    print("NETWORK TOPOLOGY RAG SYSTEM")
    print("=" * 80)
    
    # Example requests
    requests = [
        NetworkConfigRequest(
            device_id="SW1",
            action="configure",
            query="Configure switch SW1"
        ),
        NetworkConfigRequest(
            device_type="router",
            action="configure",
            query="Show all router configurations"
        ),
        NetworkConfigRequest(
            action="verify",
            query="How to verify network connectivity?"
        )
    ]
    
    for req in requests:
        print(f"\n\nREQUEST: {req.query}\n")
        result = rag.process_configuration_request(req)
        print(json.dumps(result, indent=2, default=str))
        
        print("\n\nLLM PROMPT:")
        print(rag.generate_llm_prompt(req))
