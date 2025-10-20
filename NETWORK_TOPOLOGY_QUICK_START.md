# Network Topology LLM Integration - Quick Start Guide

## What's New

Your system now understands network topology and can help configure devices using LLM. The topology information from `network_stat/topo.yaml` is:

1. **Parsed** and structured
2. **Provided to LLM** as context
3. **Used to generate** CLI commands
4. **Exposed via API** for easy access

## Quick Start

### 1. Start the Backend

```bash
# Using Docker Compose
docker-compose up -d

# Or run locally (requires dependencies)
cd backend
python3 main.py
```

### 2. Check Topology is Loaded

```bash
curl http://localhost:8000/network/topology
```

You should see all devices from your `topo.yaml` file.

### 3. Get Device Configuration

```bash
# Get SW1 configuration
curl http://localhost:8000/network/device/SW1

# Get all routers
curl "http://localhost:8000/network/devices?device_type=router"

# Get all hosts
curl "http://localhost:8000/network/devices?device_type=host"
```

### 4. Ask LLM About Network

```bash
curl -X POST http://localhost:8000/network/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What interfaces does SW1 have?",
    "model": "llama3.1:8b"
  }'
```

## API Reference

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/network/topology` | Get full topology |
| GET | `/network/device/{id}` | Get device config |
| GET | `/network/devices` | Get all devices |
| GET | `/network/devices?device_type=X` | Filter by type |
| POST | `/network/configure` | Generate config |
| POST | `/network/query` | Ask LLM about network |
| GET | `/network/context` | Get topology context |

### Example: Get Device Configuration

```bash
curl http://localhost:8000/network/device/Router1
```

**Response:**
```json
{
  "device_info": {
    "id": "Router1",
    "type": "router",
    "ip": "192.168.1.254",
    "interfaces": [
      {"port": "G0/0", "connected_to": "SW1"},
      {"port": "G0/1", "connected_to": "Internet"}
    ]
  },
  "cli_commands": "configure terminal\nhostname Router1\n..."
}
```

### Example: Query LLM

```bash
curl -X POST http://localhost:8000/network/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure OSPF on Router1?",
    "model": "llama3.1:8b",
    "max_tokens": 512
  }'
```

**Response:**
```json
{
  "question": "How do I configure OSPF on Router1?",
  "answer": "To configure OSPF on Router1:\n1. Enter configuration mode...",
  "topology_context_used": true
}
```

## Understanding the System

### Flow Diagram

```
network_stat/topo.yaml
        ↓
  TopologyParser (reads YAML)
        ↓
    Network Info (structured)
        ↓
    CLIGenerator (generates commands)
        ↓
    NetworkTopologyRAG (LLM context)
        ↓
    FastAPI Backend
        ↓
    API Endpoints
        ↓
    User / LLM Queries
```

### How LLM Gets Context

When you ask a question:

1. System loads your topology from YAML
2. Creates a structured representation
3. Generates CLI command examples
4. Sends all this to LLM as context
5. LLM answers using topology knowledge

Example context sent to LLM:
```
Network Topology: Star_Topology_Sample

Devices:
- SW1 (switch): 192.168.1.1
  - G0/1 → PC1
  - G0/2 → PC2
  - G0/3 → PC3
  - G0/4 → Router1

- Router1 (router): 192.168.1.254
  - G0/0 → SW1
  - G0/1 → Internet

[Configuration examples for each device]
```

## Use Cases

### 1. Get Configuration for Device

```bash
curl http://localhost:8000/network/device/SW1
```
→ Outputs: Cisco IOS CLI commands to configure SW1

### 2. Understand Network Structure

```bash
curl http://localhost:8000/network/topology
```
→ Shows: All devices, IPs, connections, types

### 3. Ask LLM Questions

```bash
curl -X POST http://localhost:8000/network/query \
  -d '{"query": "What is the network topology?"}'
```
→ LLM answers: "Your network has 1 switch, 1 router, 3 PCs..."

### 4. Get Configuration Guide

```bash
curl http://localhost:8000/network/configure \
  -X POST \
  -d '{"device_id": null, "action": "configure"}'
```
→ Shows: Step-by-step configuration guide for all devices

### 5. Configure Specific Device Type

```bash
curl "http://localhost:8000/network/devices?device_type=router"
```
→ Lists: All routers and their CLI configurations

## Python Integration

### In Your Code

```python
from network_stat.network_rag import NetworkTopologyRAG, NetworkConfigRequest

# Initialize
rag = NetworkTopologyRAG("network_stat/topo.yaml")

# Get device configuration
config = rag.get_device_config("SW1")
print(config)

# Process request
request = NetworkConfigRequest(
    device_id="Router1",
    action="configure",
    query="Configure Router1"
)
result = rag.process_configuration_request(request)

# Get context for LLM
context = rag.get_llm_context()
# Use context in your LLM prompt
```

## Customizing Topology

Edit `network_stat/topo.yaml`:

```yaml
network_topology:
  name: "Your Network Name"
  description: "Your description"
  devices:
    - id: DeviceID
      type: switch|router|host|external_network
      ip: 192.168.x.x
      interfaces:
        - port: PortName
          connected_to: OtherDeviceID
```

After editing:
- **Restart backend** to reload topology
- Existing API calls use new topology

## Adding New Vendors

To support Cisco NX-OS, Arista, Juniper, etc.:

1. Edit `network_stat/cli_generator.py`
2. Add new method like `_configure_switch_nxos()`
3. Update `__init__()` to accept vendor parameter
4. Use in backend with `CLIGenerator(..., config_type=ConfigType.CISCO_NXOS)`

## Troubleshooting

### "Network RAG not initialized"
- Check `network_stat/topo.yaml` exists
- Verify YAML syntax is valid
- Check file permissions

### Device not found
- Verify device ID matches `topo.yaml`
- Device IDs are case-sensitive
- Check topology is loaded: `curl http://localhost:8000/network/context`

### LLM gives generic answers
- Include device_id or device_type in query
- Ask more specific questions
- Verify LLM model supports topology context

### Configuration looks wrong
- Check device type in YAML (switch/router/host)
- Verify interfaces are properly defined
- Try accessing `/network/device/{id}` to see parsed info

## Examples

### Scenario: New Employee Needs Network Setup

**Employee:** "How do I set up the network?"

```bash
curl -X POST http://localhost:8000/network/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me step-by-step instructions to set up this network",
    "model": "llama3.1:8b",
    "max_tokens": 2048
  }'
```

**LLM Response:** Detailed setup guide including:
- Configuration for each device
- Verification steps
- Troubleshooting tips

### Scenario: Troubleshoot Device

**Admin:** "PC1 cannot reach Internet"

```bash
curl -X POST http://localhost:8000/network/query \
  -d '{"query": "PC1 cannot ping Internet. What could be wrong? Check the topology."}'
```

**LLM Response:** 
- Check SW1 configuration (connected to Router1)
- Check Router1 routing to Internet
- Verify PC1 IP and gateway
- Troubleshooting steps

### Scenario: Review Configuration

**Manager:** "Show me all router configs"

```bash
curl "http://localhost:8000/network/devices?device_type=router"
```

**Response:** CLI commands for all routers, ready to deploy

## Performance Notes

- **Topology loading**: ~10ms (one-time on startup)
- **Device lookup**: O(1) dictionary access
- **Context generation**: ~100ms (for LLM)
- **LLM query**: Depends on model (usually 1-5 seconds)

## Security Considerations

⚠️ **Important:**
- This generates configuration commands, doesn't execute them
- Always review generated configs before deploying
- Don't include real credentials in topology file
- Restrict API access in production

## Next Steps

1. ✅ Verify API endpoints work
2. ✅ Test LLM queries about your topology
3. ✅ Generate configurations for your devices
4. ✅ Customize topology for your network
5. ✅ Integrate into your automation workflow

## Files Structure

```
network_stat/
├── __init__.py                  # Package initialization
├── topo.yaml                    # Your network topology
├── topology_parser.py           # YAML parsing
├── cli_generator.py             # CLI command generation
└── network_rag.py              # LLM integration

backend/
├── main.py                      # FastAPI with network endpoints
├── requirements.txt
└── ...

demo_network_topology.py         # Test/demo script

NETWORK_TOPOLOGY_README.md       # Full documentation
NETWORK_TOPOLOGY_QUICK_START.md  # This file
```

## Support

For issues or questions:
1. Check topology YAML syntax
2. Review example requests above
3. Check LLM model is available
4. Verify backend is running

---

Happy networking! 🌐
