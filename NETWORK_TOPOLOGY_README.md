# Network Topology LLM Integration

Integrates network topology information with LLM to provide intelligent device configuration assistance.

## Overview

This system allows LLM to:
- Understand network topology structure
- Generate CLI commands for device configuration (ZebOS syntax)
- Answer questions about network setup
- Provide configuration recommendations
- Support multi-vendor commands (ZebOS, Cisco, Arista, Juniper)

## Components

### 1. **Topology Parser** (`topology_parser.py`)
Parses YAML topology files and provides device information.

**Features:**
- Loads topology from YAML
- Provides device information by ID or type
- Generates topology descriptions and maps
- Exports topology as JSON for LLM

**Usage:**
```python
from network_stat.topology_parser import TopologyParser

parser = TopologyParser("network_stat/topo.yaml")

# Get specific device
device = parser.get_device("SW1")

# Get all devices of type
switches = parser.get_devices_by_type("switch")

# Get description
print(parser.get_topology_description())
print(parser.get_network_map())
```

### 2. **CLI Generator** (`cli_generator.py`)
Generates vendor-specific CLI commands for device configuration.

**Features:**
- Generate configuration for switches, routers, hosts
- Support for ZebOS (default)
- Extensible for other vendors (Cisco IOS, Arista, Juniper)
- Configuration guides and troubleshooting

**Usage:**
```python
from network_stat.cli_generator import CLIGenerator

generator = CLIGenerator("network_stat/topo.yaml")

# Get switch configuration
sw_config = generator.get_device_config("SW1")
print(sw_config)

# Get all configurations
all_configs = generator.get_all_device_configs()

# Get summary guide
print(generator.get_summary_guide())
```

### 3. **Network RAG** (`network_rag.py`)
Integrates network topology with RAG for LLM-based assistance.

**Features:**
- Provides comprehensive topology context for LLM
- Processes configuration requests
- Generates LLM prompts with topology context
- Supports natural language queries about network

**Usage:**
```python
from network_stat.network_rag import NetworkTopologyRAG, NetworkConfigRequest

rag = NetworkTopologyRAG("network_stat/topo.yaml")

# Get topology context
context = rag.get_llm_context()

# Process configuration request
request = NetworkConfigRequest(
    device_id="SW1",
    action="configure",
    query="Configure switch SW1"
)
result = rag.process_configuration_request(request)

# Generate LLM prompt
prompt = rag.generate_llm_prompt(request)
```

## API Endpoints

### 1. Get Network Topology
```
GET /network/topology
```
Returns complete topology information and all devices.

**Response:**
```json
{
  "topology_summary": "...",
  "all_devices": {
    "SW1": {...},
    "Router1": {...}
  }
}
```

### 2. Get Device Configuration
```
GET /network/device/{device_id}
```
Get CLI commands and info for specific device.

**Parameters:**
- `device_id`: Device identifier (e.g., "SW1")

**Response:**
```json
{
  "device_info": {
    "id": "SW1",
    "type": "switch",
    "ip": "192.168.1.1",
    "interfaces": [...]
  },
  "cli_commands": "configure terminal\n..."
}
```

### 3. Get All Devices
```
GET /network/devices?device_type=switch
```
Get list of all devices, optionally filtered by type.

**Parameters:**
- `device_type` (optional): Filter by type (switch, router, host)

**Response:**
```json
{
  "total_count": 6,
  "devices": {
    "SW1": {...},
    "Router1": {...}
  }
}
```

### 4. Configure Device
```
POST /network/configure
```
Generate configuration with optional LLM assistance.

**Request:**
```json
{
  "device_id": "SW1",
  "device_type": null,
  "action": "configure",
  "query": "Configure switch SW1"
}
```

**Response:**
```json
{
  "configuration": {...},
  "llm_prompt_for_assistance": "..."
}
```

### 5. Query About Network
```
POST /network/query
```
Ask natural language questions about the network.

**Request:**
```json
{
  "query": "How do I configure BGP on Router1?",
  "model": "llama3.1:8b",
  "max_tokens": 512
}
```

**Response:**
```json
{
  "question": "How do I configure BGP on Router1?",
  "answer": "To configure BGP on Router1...",
  "topology_context_used": true
}
```

### 6. Get Network Context
```
GET /network/context
```
Get full topology context for embedding or analysis.

**Response:**
```json
{
  "context": "...",
  "devices_count": 6,
  "switches": 1,
  "routers": 1,
  "hosts": 3
}
```

## Example Workflow

### 1. Set Up
```bash
# Start backend with network topology support
docker-compose up -d

# Or run locally
cd backend
python main.py
```

### 2. Query Network Information
```bash
# Get topology
curl http://localhost:8000/network/topology

# Get specific device
curl http://localhost:8000/network/device/SW1

# Get all routers
curl "http://localhost:8000/network/devices?device_type=router"
```

### 3. Get Configuration
```bash
# Configure specific device
curl -X POST http://localhost:8000/network/configure \
  -H "Content-Type: application/json" \
  -d '{"device_id": "SW1", "action": "configure"}'
```

### 4. Ask LLM About Network
```bash
curl -X POST http://localhost:8000/network/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What interfaces does SW1 have and what are they connected to?",
    "model": "llama3.1:8b"
  }'
```

## Integration with RAG

The network topology is automatically integrated into the RAG context when:

1. **User asks about network configuration** - LLM receives topology context
2. **Configuration requests are made** - System generates vendor-specific commands
3. **Network queries occur** - Full topology is provided as context

### Example: LLM-Powered Configuration

```python
import requests

# Query: "Configure BGP between Router1 and other networks"
response = requests.post("http://localhost:8000/network/query", json={
    "query": "How to configure OSPF on Router1 to advertise all networks?",
    "model": "llama3.1:8b"
})

# LLM receives:
# 1. Network topology (all devices, IPs, connections)
# 2. Device configuration examples
# 3. Your specific question

# Response includes:
# - Detailed CLI commands
# - Explanation of each step
# - Verification procedures
```

## Supported Device Types

### Switches
- Configure VLANs
- Set interface speeds
- Configure spanning tree
- Example: SW1, SW2

### Routers
- Configure interfaces
- Set routing protocols (OSPF, BGP)
- Configure static routes
- Example: Router1, Router2

### Hosts
- Set IP addresses
- Configure gateways
- DHCP configuration
- Example: PC1, PC2, PC3

## Topology YAML Format

```yaml
network_topology:
  name: "Network Name"
  description: "Description"
  devices:
    - id: SW1
      type: switch
      ip: 192.168.1.1
      interfaces:
        - port: G0/1
          connected_to: Router1
    - id: Router1
      type: router
      ip: 192.168.1.254
      interfaces:
        - port: G0/0
          connected_to: SW1
        - port: G0/1
          connected_to: Internet
    - id: PC1
      type: host
      ip: 192.168.1.10
      mac: "00:11:22:33:44:01"
      connected_to: SW1
```

## Extending for New Vendors

Add support for new vendors in `cli_generator.py`:

```python
def _configure_switch_arista(self, device: Dict[str, Any]) -> str:
    """Generate Arista switch configuration"""
    config = "..."
    return config
```

Then update `get_device_config()` to use vendor-specific methods.

## LLM Context Example

The LLM receives context like:

```
NETWORK TOPOLOGY INFORMATION:

Network Topology: Star_Topology_Sample
Description: Mạng nội bộ văn phòng nhỏ với mô hình hình sao

Devices:
- SW1 (switch): IP 192.168.1.1
  - G0/1 → PC1
  - G0/2 → PC2
  - G0/3 → PC3
  - G0/4 → Router1

- Router1 (router): IP 192.168.1.254
  - G0/0 → SW1
  - G0/1 → Internet

- PC1 (host): IP 192.168.1.10
  - MAC: 00:11:22:33:44:01

[... more device info ...]

Common Commands:
- Interface configuration: interface <port>
- IP configuration: ip address <ip> <mask>
- Routing: router ospf <id>
```

## Troubleshooting

### Network RAG Not Initialized
Check that `network_stat/topo.yaml` exists and is valid YAML.

### Device Not Found
Verify device ID in topology file matches request.

### No CLI Commands Generated
Ensure device type is supported (switch, router, host).

### LLM Gives Generic Answers
Provide more specific query with device IDs or types.

## Performance Considerations

- **Topology Loading**: Happens once on startup
- **Context Size**: Full topology fits in LLM context (usually <4K tokens)
- **Query Latency**: Added minimal overhead (topology lookup is O(1))

## Future Enhancements

- [ ] Multi-site topology support
- [ ] Device authentication credentials management
- [ ] Configuration change history tracking
- [ ] Automated configuration validation
- [ ] Integration with actual network devices (SSH/Telnet)
- [ ] Real-time device status monitoring
- [ ] Configuration template generation
- [ ] Network diagram visualization
