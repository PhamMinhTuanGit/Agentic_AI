"""
Network Device Configuration CLI Generator
Generates CLI commands for configuring network devices based on topology
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from network_stat.topology_parser import TopologyParser, Device


class ConfigType(Enum):
    """Configuration command types"""
    ZEBOS = "zebos"
    CISCO_IOS = "cisco_ios"
    CISCO_NXOS = "cisco_nxos"
    ARISTA = "arista"
    JUNIPER = "juniper"


class CLIGenerator:
    """Generate CLI commands for network devices"""
    
    def __init__(self, topology_file: str, config_type: ConfigType = ConfigType.ZEBOS):
        self.parser = TopologyParser(topology_file)
        self.config_type = config_type
    
    def get_device_config(self, device_id: str) -> str:
        """Get configuration for a specific device"""
        device = self.parser.get_device(device_id)
        if not device:
            return f"Device {device_id} not found"
        
        if device.type == 'switch':
            return self._configure_switch(device)
        elif device.type == 'router':
            return self._configure_router(device)
        elif device.type == 'host':
            return self._configure_host(device)
        else:
            return f"Unknown device type: {device.type}"
    
    def _configure_switch(self, device: Device) -> str:
        """Generate switch configuration (ZebOS)"""
        config = f"""
!================== SWITCH CONFIGURATION: {device.id} ==================
! Device Type: {device.type.upper()}
! Management IP: {device.ip}
! OS: ZebOS
!

configure
!
! Hostname
hostname {device.id}
!
! Management Interface
interface ethernet 1
 ipv4 address {device.ip} 255.255.255.0
 no shutdown
 exit
!
! Interface Configuration
"""
        for intf in device.interfaces:
            port_name = intf.get('port', intf.get('name', 'unknown'))
            config += f"""interface ethernet {port_name}
 description Connected to {intf['connected_to']}
 no shutdown
 exit
!
"""
        
        config += """! Enable routing and spanning tree
ip routing
spanning-tree enable
!
exit
"""
        return config
    
    def _configure_router(self, device: Device) -> str:
        """Generate router configuration (ZebOS)"""
        config = f"""
!================== ROUTER CONFIGURATION: {device.id} ==================
! Device Type: {device.type.upper()}
! OS: ZebOS
!

configure
!
! Hostname
hostname {device.id}
!
"""
        
        # Configure interfaces
        for intf in device.interfaces:
            port_name = intf.get('port', intf.get('name', 'unknown'))
            # Get IP from interface if available, otherwise calculate
            if 'ip' in intf:
                ip_info = intf['ip']
                # Handle CIDR notation (e.g., "10.0.12.1/30")
                if '/' in str(ip_info):
                    ip_addr = str(ip_info).split('/')[0]
                    cidr = str(ip_info).split('/')[1]
                    # Convert CIDR to netmask
                    netmask = self._cidr_to_netmask(int(cidr))
                else:
                    ip_addr = str(ip_info)
                    netmask = "255.255.255.0"  # Default
            else:
                ip_addr = self._calculate_ip_for_interface(device.ip if device.ip else "192.168.1.0", 0)
                netmask = "255.255.255.0"
            
            config += f"""! Interface {port_name}
interface ethernet {port_name}
 description Connected to {intf['connected_to']}
 ipv4 address {ip_addr} {netmask}
 no shutdown
 exit
!
"""
        
        config += """! Enable routing protocols (example: OSPF)
router ospf 1
 network 10.0.0.0 0.0.255.255 area 0
 exit
!
exit
"""
        return config
    
    def _configure_host(self, device: Device) -> str:
        """Generate host/PC configuration"""
        config = f"""
!================== HOST CONFIGURATION: {device.id} ==================
! Device Type: {device.type.upper()}
! IP Address: {device.ip}
"""
        
        if device.mac:
            config += f"! MAC Address: {device.mac}\n"
        
        config += f"""!
! Linux/Windows Commands:

# Set IP address and gateway
ip addr add {device.ip}/24 dev eth0  # Linux
netsh interface ipv4 set address name="Ethernet" static {device.ip} 255.255.255.0 192.168.1.254  # Windows

# Or configure with DHCP
dhclient eth0  # Linux
ipconfig /all  # Windows

# Verify connectivity
ping 192.168.1.254  # Test gateway
ping 192.168.1.1    # Test switch
"""
        return config
    
    def _calculate_ip_for_interface(self, base_ip: str, interface_index: int) -> str:
        """Calculate IP for interface (simplified)"""
        parts = base_ip.split('.')
        if len(parts) >= 4:
            parts[3] = str(int(parts[3]) + interface_index + 1)
            return '.'.join(parts)
        return base_ip
    
    def _cidr_to_netmask(self, cidr: int) -> str:
        """Convert CIDR notation to netmask"""
        # Common CIDR to netmask conversions
        cidr_map = {
            8: "255.0.0.0",
            16: "255.255.0.0",
            24: "255.255.255.0",
            25: "255.255.255.128",
            26: "255.255.255.192",
            27: "255.255.255.224",
            28: "255.255.255.240",
            29: "255.255.255.248",
            30: "255.255.255.252",
            31: "255.255.255.254",
            32: "255.255.255.255",
        }
        return cidr_map.get(cidr, "255.255.255.0")
    
    def get_all_device_configs(self) -> Dict[str, str]:
        """Get configurations for all devices"""
        configs = {}
        for device_id in self.parser.get_all_devices().keys():
            configs[device_id] = self.get_device_config(device_id)
        return configs
    
    def get_summary_guide(self) -> str:
        """Get summary configuration guide"""
        guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                 NETWORK CONFIGURATION GUIDE                               ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 TOPOLOGY SUMMARY:
"""
        guide += self.parser.get_network_map()
        
        guide += "\n\n📝 CONFIGURATION STEPS:\n\n"
        
        switches = self.parser.get_devices_by_type('switch')
        routers = self.parser.get_devices_by_type('router')
        hosts = self.parser.get_devices_by_type('host')
        
        guide += f"1. CONFIGURE SWITCHES ({len(switches)}):\n"
        for sw in switches:
            guide += f"   - {sw.id}: Use commands from 'configure_sw_config' section\n"
        
        guide += f"\n2. CONFIGURE ROUTERS ({len(routers)}):\n"
        for router in routers:
            guide += f"   - {router.id}: Use commands from 'configure_router_config' section\n"
        
        guide += f"\n3. CONFIGURE HOSTS ({len(hosts)}):\n"
        for host in hosts:
            guide += f"   - {host.id}: Set IP {host.ip}\n"
            if host.mac:
                guide += f"     MAC: {host.mac}\n"
        
        guide += """

🔗 CONNECTIVITY VERIFICATION:

Step 1: Verify device connectivity
  ping <device_ip>

Step 2: Check routing table
  (On routers) show ip route

Step 3: Verify VLAN configuration
  (On switches) show vlan

Step 4: Check port status
  (On switches) show interface status

✅ QUICK START COMMANDS:

# SSH into device
ssh -u admin 192.168.1.1

# Telnet into device (legacy)
telnet 192.168.1.1

# Configuration backup
show running-config > device_backup.txt

# Save configuration
write memory  # or: copy running-config startup-config
"""
        return guide
    
    def get_llm_context(self) -> str:
        """Generate context for LLM to understand topology and configurations"""
        context = f"""
NETWORK TOPOLOGY INFORMATION:

{self.parser.get_topology_json()}

DEVICE CONFIGURATIONS:

"""
        for device_id, config in self.get_all_device_configs().items():
            context += f"\n{device_id}:\n{config}\n"
        
        context += self.get_summary_guide()
        return context


if __name__ == "__main__":
    generator = CLIGenerator("network_stat/topo.yaml")
    
    print("=" * 80)
    print("NETWORK TOPOLOGY CONFIGURATION GUIDE")
    print("=" * 80)
    
    print(generator.get_summary_guide())
    
    print("\n\n" + "=" * 80)
    print("DETAILED DEVICE CONFIGURATIONS")
    print("=" * 80)
    
    for device_id, config in generator.get_all_device_configs().items():
        print(config)
