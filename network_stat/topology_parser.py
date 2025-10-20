"""
Network Topology Parser
Parses YAML topology files and provides device configuration information
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Interface:
    """Network interface information"""
    port: str
    connected_to: str
    ip: Optional[str] = None
    mac: Optional[str] = None
    vlan: Optional[int] = None


@dataclass
class Device:
    """Network device information"""
    id: str
    type: str  # switch, router, host, external_network
    ip: str
    interfaces: List[Dict[str, Any]]
    mac: Optional[str] = None
    ip_range: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TopologyParser:
    """Parse and manage network topology"""
    
    def __init__(self, topology_file: str):
        self.topology_file = Path(topology_file)
        self.topology = None
        self.devices = {}
        self.load_topology()
    
    def load_topology(self) -> None:
        """Load topology from YAML file"""
        with open(self.topology_file, 'r', encoding='utf-8') as f:
            self.topology = yaml.safe_load(f)
        
        # Parse devices
        for device_info in self.topology['network_topology']['devices']:
            device = Device(
                id=device_info['id'],
                type=device_info['type'],
                ip=device_info.get('ip', ''),
                interfaces=device_info.get('interfaces', []),
                mac=device_info.get('mac'),
                ip_range=device_info.get('ip_range')
            )
            self.devices[device.id] = device
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID"""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> Dict[str, Device]:
        """Get all devices"""
        return self.devices
    
    def get_devices_by_type(self, device_type: str) -> List[Device]:
        """Get devices by type (switch, router, host, etc.)"""
        return [d for d in self.devices.values() if d.type == device_type]
    
    def get_topology_description(self) -> str:
        """Get human-readable topology description"""
        topo = self.topology['network_topology']
        description = f"""
Network Topology: {topo['name']}
Description: {topo['description']}

Devices:
"""
        for device in self.devices.values():
            description += f"\n  {device.id} ({device.type}):\n"
            description += f"    IP: {device.ip}\n"
            if device.mac:
                description += f"    MAC: {device.mac}\n"
            if device.ip_range:
                description += f"    IP Range: {device.ip_range}\n"
            
            if device.interfaces:
                description += f"    Interfaces:\n"
                for intf in device.interfaces:
                    # Support both 'port' and 'name' keys
                    port_name = intf.get('port', intf.get('name', 'unknown'))
                    description += f"      {port_name} -> {intf['connected_to']}\n"
        
        return description
    
    def get_topology_json(self) -> str:
        """Get topology as JSON for LLM"""
        devices_dict = {}
        for device_id, device in self.devices.items():
            devices_dict[device_id] = device.to_dict()
        
        return json.dumps({
            'name': self.topology['network_topology']['name'],
            'description': self.topology['network_topology']['description'],
            'devices': devices_dict
        }, indent=2)
    
    def get_device_connections(self, device_id: str) -> Dict[str, List[str]]:
        """Get all connections for a device"""
        device = self.get_device(device_id)
        if not device:
            return {}
        
        connections = {
            'device_id': device.id,
            'device_type': device.type,
            'connections': []
        }
        
        for intf in device.interfaces:
            connections['connections'].append({
                'port': intf.get('port', intf.get('name', 'unknown')),
                'connected_to': intf['connected_to']
            })
        
        return connections
    
    def get_network_map(self) -> str:
        """Get ASCII representation of network"""
        map_str = "\n=== NETWORK TOPOLOGY MAP ===\n\n"
        
        switches = self.get_devices_by_type('switch')
        routers = self.get_devices_by_type('router')
        hosts = self.get_devices_by_type('host')
        
        map_str += "SWITCHES:\n"
        for sw in switches:
            map_str += f"  ┌─ {sw.id} ({sw.ip})\n"
            for intf in sw.interfaces:
                port_name = intf.get('port', intf.get('name', 'unknown'))
                map_str += f"  │   └─ {port_name} → {intf['connected_to']}\n"
        
        map_str += "\nROUTERS:\n"
        for router in routers:
            map_str += f"  ┌─ {router.id} ({router.ip})\n"
            for intf in router.interfaces:
                port_name = intf.get('port', intf.get('name', 'unknown'))
                map_str += f"  │   └─ {port_name} → {intf['connected_to']}\n"
        
        map_str += "\nHOSTS:\n"
        for host in hosts:
            map_str += f"  ├─ {host.id} ({host.ip})\n"
            if host.mac:
                map_str += f"  │  MAC: {host.mac}\n"
        
        return map_str


if __name__ == "__main__":
    # Example usage
    parser = TopologyParser("network_stat/topo.yaml")
    print(parser.get_topology_description())
    print(parser.get_network_map())
    print("\nJSON Format:")
    print(parser.get_topology_json())
