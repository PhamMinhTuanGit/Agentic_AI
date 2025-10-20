# Ring Topology Configuration Guide

## Overview

The ring topology is a network configuration where 4 routers are connected in a circular pattern, creating a closed loop. This topology provides redundancy - if one link fails, traffic can still flow through the alternative path.

**File Location:** `network_stat/ring_topology.yaml`

---

## Topology Structure

### Network Diagram
```
        R1 (10.0.12.1/30)
       /  \
      /    \
    G0/1   G0/0 (to R2)
  (to R4)   |
    /       |
   /        |
  R4        R2
   \        |
    \       |
    G0/1   G0/1 (to R3)
  (to R1)   |
      \    /
       \  /
        R3 (10.0.34.1/30)
```

### Devices Configuration

#### Router 1 (R1)
- **Role:** Core Router
- **Interface G0/0:** Connected to R2
  - IP: `10.0.12.1/30`
  - Network: `10.0.12.0/30`
- **Interface G0/1:** Connected to R4
  - IP: `10.0.14.1/30`
  - Network: `10.0.14.0/30`

#### Router 2 (R2)
- **Role:** Core Router
- **Interface G0/0:** Connected to R1
  - IP: `10.0.12.2/30`
  - Network: `10.0.12.0/30`
- **Interface G0/1:** Connected to R3
  - IP: `10.0.23.1/30`
  - Network: `10.0.23.0/30`

#### Router 3 (R3)
- **Role:** Core Router
- **Interface G0/0:** Connected to R2
  - IP: `10.0.23.2/30`
  - Network: `10.0.23.0/30`
- **Interface G0/1:** Connected to R4
  - IP: `10.0.34.1/30`
  - Network: `10.0.34.0/30`

#### Router 4 (R4)
- **Role:** Core Router
- **Interface G0/0:** Connected to R3
  - IP: `10.0.34.2/30`
  - Network: `10.0.34.0/30`
- **Interface G0/1:** Connected to R1
  - IP: `10.0.14.2/30`
  - Network: `10.0.14.0/30`

---

## Network Links Summary

| Link | Router1 | IP1 | Router2 | IP2 | Subnet |
|------|---------|-----|---------|-----|--------|
| Link 1 | R1 | 10.0.12.1/30 | R2 | 10.0.12.2/30 | 10.0.12.0/30 |
| Link 2 | R2 | 10.0.23.1/30 | R3 | 10.0.23.2/30 | 10.0.23.0/30 |
| Link 3 | R3 | 10.0.34.1/30 | R4 | 10.0.34.2/30 | 10.0.34.0/30 |
| Link 4 | R4 | 10.0.14.2/30 | R1 | 10.0.14.1/30 | 10.0.14.0/30 |

---

## Benefits of Ring Topology

### ✅ Advantages
1. **Redundancy:** If one link fails, traffic can use the alternate path
2. **Easy to Expand:** Can add additional routers by breaking the ring
3. **Balanced Load:** Traffic can be distributed across multiple paths
4. **Simple Management:** Clear, predictable structure

### ⚠️ Limitations
1. **Loop Prevention Required:** Must implement STP (Spanning Tree Protocol)
2. **Bandwidth Constraints:** Each link is shared in both directions
3. **Scaling Complexity:** Gets complicated with many devices
4. **Network Latency:** Traffic between distant routers may take longer

---

## Configuration Steps

### Step 1: Load Topology
```python
from network_stat.topology_parser import TopologyParser

parser = TopologyParser(topology_file="network_stat/ring_topology.yaml")
topology = parser.load_topology()
```

### Step 2: Get Device Information
```python
# Get specific router
r1 = parser.get_device("R1")
print(f"Router: {r1.id}, Type: {r1.type}, IP: {r1.ip}")

# Get all routers
routers = parser.get_devices_by_type("router")
print(f"Found {len(routers)} routers in the network")
```

### Step 3: Analyze Connections
```python
# View network connections
connections = parser.get_device_connections("R1")
print(f"R1 is connected to: {connections}")

# Get network map
network_map = parser.get_network_map()
print(network_map)
```

### Step 4: Generate CLI Commands
```python
from network_stat.cli_generator import CLIGenerator

cli_gen = CLIGenerator(parser)

# Get configuration for R1
config = cli_gen.get_device_config("R1")
print(config)
```

### Step 5: Use with LLM
```python
from network_stat.network_rag import NetworkTopologyRAG

network_rag = NetworkTopologyRAG(parser, cli_gen)

# Get LLM context for this topology
llm_context = network_rag.get_llm_context()

# Query about the network
result = network_rag.process_configuration_request(
    query="Configure R1 to connect to all other routers"
)
print(result)
```

---

## Routing Protocols for Ring Topology

### Recommended: OSPF (Open Shortest Path First)
```
Router R1 Configuration:
  router ospf 1
    network 10.0.12.0 0.0.0.3 area 0
    network 10.0.14.0 0.0.0.3 area 0
```

### Alternative: BGP (Border Gateway Protocol)
```
Router R1 Configuration:
  router bgp 65000
    neighbor 10.0.12.2 remote-as 65000
    neighbor 10.0.14.2 remote-as 65000
```

### Alternative: RIP (Routing Information Protocol)
```
Router R1 Configuration:
  router rip
    network 10.0.12.0
    network 10.0.14.0
```

---

## Testing the Topology

### Verify Connectivity
```bash
# SSH to R1 and test ping
ssh admin@10.0.12.1
ping 10.0.23.2  # Ping to R3 (via R2 and R4)
ping 10.0.34.1  # Ping to R3 direct link
```

### Check Routing Table
```bash
# View routing table on R1
show ip route

# Show OSPF neighbors
show ip ospf neighbor

# Show interface status
show interface status
```

### Simulate Link Failure
```bash
# Disable interface G0/0 on R1 (break link to R2)
interface GigabitEthernet0/0
  shutdown

# Check if traffic reroutes via R4-R3-R2
show ip route
```

---

## LLM Integration with Ring Topology

The Ring Topology is automatically understood by the LLM through the RAG pipeline integration. When you query about the network:

### Example Queries
1. **"What is the network topology?"**
   - LLM returns complete ring topology structure

2. **"Configure R1 to have optimal routing"**
   - LLM generates OSPF/BGP configuration for R1

3. **"Show the connection between R2 and R4"**
   - LLM explains the path: R2 → R3 → R4

4. **"What happens if link R1-R2 fails?"**
   - LLM analyzes redundancy: traffic goes via R1 → R4 → R3 → R2

---

## Advanced Configuration

### Enable Spanning Tree Protocol (STP)
```
spanning-tree mode rapid

interface range G0/0,G0/1
  spanning-tree portfast
  spanning-tree bpduguard enable
```

### Load Balancing Across Ring
```
interface G0/0
  ip address 10.0.12.1 255.255.255.252
  load-interval 30

interface G0/1
  ip address 10.0.14.1 255.255.255.252
  load-interval 30
```

### QoS Configuration
```
class-map CRITICAL
  match protocol ospf
  
policy-map RING-POLICY
  class CRITICAL
    priority 100000
```

---

## Troubleshooting

### Issue: Routing Loop
**Solution:** Implement STP or use dynamic routing protocol like OSPF with cost metrics

### Issue: Asymmetric Routing
**Solution:** Configure consistent routing metrics on all routers

### Issue: High Latency
**Solution:** Optimize routing metrics; consider adding more links

### Issue: Link Failover Not Working
**Solution:** Check if backup routes exist; verify OSPF timers

---

## Files Reference

- **Topology YAML:** `network_stat/ring_topology.yaml`
- **Parser Module:** `network_stat/topology_parser.py`
- **CLI Generator:** `network_stat/cli_generator.py`
- **RAG Integration:** `network_stat/network_rag.py`
- **RAG Pipeline:** `rag/pipeline.py` (with topology context added)
- **API Backend:** `backend/main.py`

---

## Quick Start

```bash
# 1. Load and view the topology
python3 -c "
from network_stat.topology_parser import TopologyParser
parser = TopologyParser('network_stat/ring_topology.yaml')
print(parser.get_topology_description())
"

# 2. Generate configuration
python3 -c "
from network_stat.topology_parser import TopologyParser
from network_stat.cli_generator import CLIGenerator
parser = TopologyParser('network_stat/ring_topology.yaml')
cli_gen = CLIGenerator(parser)
print(cli_gen.get_device_config('R1'))
"

# 3. Query with LLM
python3 -c "
from network_stat.topology_parser import TopologyParser
from network_stat.cli_generator import CLIGenerator
from network_stat.network_rag import NetworkTopologyRAG

parser = TopologyParser('network_stat/ring_topology.yaml')
cli_gen = CLIGenerator(parser)
rag = NetworkTopologyRAG(parser, cli_gen)
print(rag.get_llm_context())
"
```

---

## Integration with RAG Pipeline

The ring topology is now integrated into the RAG pipeline. When you ask questions about network configuration:

1. **Query Processing:** Your question is processed through the RAG pipeline
2. **Topology Context:** The ring topology is automatically included in the context
3. **LLM Understanding:** The LLM understands the full network structure
4. **Smart Recommendations:** The LLM provides topology-aware configuration suggestions
5. **Cached Results:** Similar queries return cached topology information

See `rag/pipeline.py` for details on the topology integration step.

---

## Version History

- **v1.0** (Oct 2025): Initial ring topology configuration
- **v1.1** (Oct 2025): Added LLM context integration to RAG pipeline
- **v1.2** (Oct 2025): Added STP and advanced routing protocols

---

## Support

For questions or issues with the ring topology:
1. Check the topology YAML structure
2. Verify router interfaces and IPs are correct
3. Test connectivity between adjacent routers
4. Review LLM context generation logs
5. Check cache for previous topology queries

---

**Last Updated:** October 20, 2025  
**Status:** ✅ Production Ready  
**LLM Integration:** ✅ Active in RAG Pipeline
