# ZebOS Command Reference Guide

## Overview

ZebOS is a network operating system used for configuring and managing network devices. This guide provides common ZebOS commands used in our RAG system.

## Table of Contents

1. [Basic Commands](#basic-commands)
2. [Configuration Mode](#configuration-mode)
3. [Interface Configuration](#interface-configuration)
4. [Routing Protocols](#routing-protocols)
5. [VLAN Configuration](#vlan-configuration)
6. [ACLs and Firewall](#acls-and-firewall)
7. [QoS Configuration](#qos-configuration)
8. [Monitoring and Diagnostics](#monitoring-and-diagnostics)
9. [Common Configuration Examples](#common-configuration-examples)

---

## Basic Commands

### User Mode Commands

```zsh
# Show device information
show version
show chassis
show system

# Connect to device
telnet <device_ip>
ssh -u admin <device_ip>

# Exit current session
quit
exit
logout
```

### Enable Mode

```zsh
# Enter privileged mode (if required)
enable
enable secret <password>

# Disable privileged mode
disable
```

---

## Configuration Mode

### Entering and Exiting Configuration

```zsh
# Enter configuration mode
R1#configure

# Show current configuration
R1#show running-config
R1#show startup-config

# Save configuration
R1#write memory
R1#copy running-config startup-config

# Exit configuration mode
R1(config)#exit
```

### Basic Configuration Commands

```zsh
# Set hostname
R1(config)#hostname R1

# Set system description
R1(config)#description "Router 1 - Core Network"

# Configure system clock (NTP)
R1(config)#ntp server 192.168.1.1
R1(config)#clock timezone PST -8
R1(config)#clock summer-time PDT recurring last Sunday March 2:00 last Sunday October 2:00
```

---

## Interface Configuration

### Basic Interface Configuration

```zsh
# Enter interface configuration mode
R1(config)#interface ethernet G0/0
R1(config-if)#

# Configure IPv4 address
R1(config-if)#ipv4 address 10.1.1.1 255.255.255.0

# Configure IPv6 address (if supported)
R1(config-if)#ipv6 address 2001:db8:1::1/64

# Enable/disable interface
R1(config-if)#no shutdown       # Enable
R1(config-if)#shutdown          # Disable

# Set interface description
R1(config-if)#description "Link to R2"

# Set interface bandwidth
R1(config-if)#bandwidth 100000

# Set interface MTU
R1(config-if)#mtu 1500

# Exit interface configuration
R1(config-if)#exit
```

### Port Configuration

```zsh
# Configure multiple interfaces
R1(config)#interface ethernet range G0/0-3
R1(config-if-range)#no shutdown
R1(config-if-range)#exit

# Configure as trunk port (for switches)
SW1(config)#interface ethernet G0/0
SW1(config-if)#switchport mode trunk
SW1(config-if)#switchport trunk allowed vlan 1,10,20
SW1(config-if)#exit

# Configure as access port
SW1(config-if)#switchport mode access
SW1(config-if)#switchport access vlan 10
```

### Secondary IP Addresses

```zsh
# Configure secondary IP
R1(config-if)#ipv4 address secondary 10.1.2.1 255.255.255.0
```

---

## Routing Protocols

### OSPF Configuration

```zsh
# Enable OSPF
R1(config)#router ospf 1
R1(config-router)#router-id 1.1.1.1

# Advertise networks
R1(config-router)#network 10.1.1.0 0.0.0.255 area 0
R1(config-router)#network 10.1.2.0 0.0.0.255 area 1
R1(config-router)#network 1.1.1.1 0.0.0.0 area 0

# Enable OSPF on specific interface
R1(config-router)#interface ethernet G0/0
R1(config-router-if)#ip ospf priority 100
R1(config-router-if)#ip ospf cost 10
R1(config-router-if)#exit

# Configure OSPF timers
R1(config-router)#timers spf 200 1000
R1(config-router)#timers throttle spf 200 1000 10000

# Enable BFD for fast failure detection
R1(config-router)#bfd all-interfaces

# Exit OSPF configuration
R1(config-router)#exit
```

### BGP Configuration

```zsh
# Enable BGP
R1(config)#router bgp 65001
R1(config-router)#router-id 1.1.1.1

# Configure neighbor
R1(config-router)#neighbor 10.1.1.2 remote-as 65001
R1(config-router)#neighbor 10.1.1.2 description "R2-iBGP"

# Advertise networks
R1(config-router)#network 192.168.0.0 mask 255.255.255.0
R1(config-router)#redistribute ospf 1

# Configure BGP timers
R1(config-router)#timers bgp 60 180

# Exit BGP configuration
R1(config-router)#exit
```

### RIP Configuration

```zsh
# Enable RIP
R1(config)#router rip
R1(config-router)#version 2

# Advertise networks
R1(config-router)#network 10.0.0.0

# Disable RIP on specific interfaces
R1(config-router)#passive-interface ethernet G0/0

# Exit RIP configuration
R1(config-router)#exit
```

### EIGRP Configuration

```zsh
# Enable EIGRP
R1(config)#router eigrp 100
R1(config-router)#eigrp router-id 1.1.1.1

# Advertise networks
R1(config-router)#network 10.1.1.0 0.0.0.255
R1(config-router)#network 10.1.2.0 0.0.0.255

# Configure timers
R1(config-router)#timers active-time 3

# Exit EIGRP configuration
R1(config-router)#exit
```

---

## VLAN Configuration

### VLAN Setup

```zsh
# Create VLAN
SW1(config)#vlan 10
SW1(config-vlan)#name "Department10"
SW1(config-vlan)#exit

# Assign port to VLAN
SW1(config)#interface ethernet G0/0
SW1(config-if)#switchport access vlan 10
SW1(config-if)#exit

# Configure VLAN interface
SW1(config)#interface vlan 10
SW1(config-if)#ipv4 address 10.10.10.1 255.255.255.0
SW1(config-if)#no shutdown
SW1(config-if)#exit

# Configure trunk
SW1(config)#interface ethernet G0/24
SW1(config-if)#switchport mode trunk
SW1(config-if)#switchport trunk allowed vlan 1,10,20
SW1(config-if)#exit
```

---

## ACLs and Firewall

### Standard ACLs

```zsh
# Create standard ACL
R1(config)#access-list 1 permit 10.1.1.0 0.0.0.255
R1(config)#access-list 1 deny any

# Apply ACL to interface
R1(config)#interface ethernet G0/0
R1(config-if)#ip access-group 1 in
R1(config-if)#exit
```

### Extended ACLs

```zsh
# Create extended ACL
R1(config)#access-list 100 permit tcp 10.1.1.0 0.0.0.255 192.168.0.0 0.0.0.255 eq 80
R1(config)#access-list 100 permit tcp 10.1.1.0 0.0.0.255 192.168.0.0 0.0.0.255 eq 443
R1(config)#access-list 100 deny icmp 10.1.1.0 0.0.0.255 192.168.0.0 0.0.0.255
R1(config)#access-list 100 permit ip any any

# Apply ACL to interface
R1(config)#interface ethernet G0/0
R1(config-if)#ip access-group 100 in
R1(config-if)#exit
```

---

## QoS Configuration

### Basic QoS

```zsh
# Create class-map
R1(config)#class-map match-any VOICE
R1(config-cmap)#match protocol rtp
R1(config-cmap)#exit

# Create policy-map
R1(config)#policy-map QOS_POLICY
R1(config-pmap)#class VOICE
R1(config-pmap-c)#priority 100000
R1(config-pmap-c)#exit
R1(config-pmap)#class class-default
R1(config-pmap-c)#fair-queue
R1(config-pmap-c)#exit
R1(config-pmap)#exit

# Apply policy to interface
R1(config)#interface ethernet G0/0
R1(config-if)#service-policy output QOS_POLICY
R1(config-if)#exit
```

---

## Monitoring and Diagnostics

### Show Commands

```zsh
# Device information
R1#show version
R1#show interfaces
R1#show interface brief
R1#show ip route
R1#show ip interface brief

# Routing protocol information
R1#show ip ospf neighbor
R1#show ip ospf database
R1#show ip bgp summary
R1#show ip eigrp neighbors
R1#show ip rip database

# Connectivity tests
R1#ping 10.1.1.2
R1#ping 10.1.1.2 count 10
R1#traceroute 192.168.1.1

# Debug commands
R1#debug ip ospf events
R1#debug ip bgp keepalives
R1#debug ip routing

# Stop debugging
R1#undebug all
R1#no debug all
```

### Configuration Verification

```zsh
# Show running configuration
R1#show running-config

# Show specific configuration
R1#show running-config | include ospf
R1#show running-config interface ethernet G0/0

# Compare configurations
R1#show running-config > backup.txt
R1#copy running-config tftp://192.168.1.1/backup.txt
```

---

## Common Configuration Examples

### Basic Router Setup

```zsh
! Configure hostname
R1#configure
R1(config)#hostname R1

! Configure interfaces
R1(config)#interface ethernet G0/0
R1(config-if)#ipv4 address 10.1.1.1 255.255.255.0
R1(config-if)#no shutdown
R1(config-if)#exit

R1(config)#interface ethernet G0/1
R1(config-if)#ipv4 address 10.2.1.1 255.255.255.0
R1(config-if)#no shutdown
R1(config-if)#exit

! Configure OSPF
R1(config)#router ospf 1
R1(config-router)#router-id 1.1.1.1
R1(config-router)#network 10.1.1.0 0.0.0.255 area 0
R1(config-router)#network 10.2.1.0 0.0.0.255 area 0
R1(config-router)#exit

! Save configuration
R1(config)#exit
R1#write memory
```

### Ring Topology Configuration

```zsh
! Configure R1 in ring topology
R1#configure
R1(config)#hostname R1

! Interface to R2
R1(config)#interface ethernet G0/0
R1(config-if)#ipv4 address 10.0.12.1 255.255.255.252
R1(config-if)#no shutdown
R1(config-if)#exit

! Interface to R4
R1(config)#interface ethernet G0/1
R1(config-if)#ipv4 address 10.0.14.2 255.255.255.252
R1(config-if)#no shutdown
R1(config-if)#exit

! Configure OSPF for ring
R1(config)#router ospf 1
R1(config-router)#router-id 1.1.1.1
R1(config-router)#network 10.0.12.0 0.0.0.3 area 0
R1(config-router)#network 10.0.14.0 0.0.0.3 area 0
R1(config-router)#exit

R1(config)#exit
R1#write memory
```

### Switch VLAN Configuration

```zsh
! Create VLANs
SW1#configure
SW1(config)#vlan 10
SW1(config-vlan)#name "Management"
SW1(config-vlan)#exit

SW1(config)#vlan 20
SW1(config-vlan)#name "Production"
SW1(config-vlan)#exit

! Configure VLAN interfaces
SW1(config)#interface vlan 10
SW1(config-if)#ipv4 address 10.10.10.254 255.255.255.0
SW1(config-if)#no shutdown
SW1(config-if)#exit

! Assign ports to VLANs
SW1(config)#interface ethernet G0/0
SW1(config-if)#switchport access vlan 10
SW1(config-if)#exit

SW1(config)#interface ethernet G0/1
SW1(config-if)#switchport access vlan 20
SW1(config-if)#exit

! Configure trunk
SW1(config)#interface ethernet G0/24
SW1(config-if)#switchport mode trunk
SW1(config-if)#switchport trunk allowed vlan 10,20
SW1(config-if)#no shutdown
SW1(config-if)#exit

SW1(config)#exit
SW1#write memory
```

---

## Key Differences: ZebOS vs Cisco IOS

| Feature | Cisco IOS | ZebOS |
|---------|-----------|-------|
| Config mode entry | `configure terminal` | `configure` |
| IP configuration | `ip address x.x.x.x y.y.y.y` | `ipv4 address x.x.x.x y.y.y.y` |
| Interface type | `interface G0/0` | `interface ethernet G0/0` |
| Exit mode | `end` or `exit` | `exit` |
| Save config | `copy run start` | `write memory` |
| Show commands | `show` | `show` (similar) |
| Routing mode | `router ospf 1` | `router ospf 1` (similar) |

---

## Best Practices

1. **Always backup configuration** before making changes:
   ```zsh
   R1#copy running-config tftp://192.168.1.1/backup.txt
   ```

2. **Verify changes** after configuration:
   ```zsh
   R1#show running-config
   R1#show ip interface brief
   ```

3. **Use descriptions** for better documentation:
   ```zsh
   R1(config-if)#description "Link to R2 - Core"
   ```

4. **Enable logging** for troubleshooting:
   ```zsh
   R1(config)#logging host 192.168.1.100
   R1(config)#logging level informational
   ```

5. **Use ACLs** to control traffic:
   ```zsh
   R1(config)#access-list 100 permit ip 10.0.0.0 0.0.255.255 any
   ```

---

## Additional Resources

- ZebOS Official Documentation
- Network Configuration Examples
- Routing Protocol Configuration Guides
- Security Best Practices
- Performance Tuning Guidelines

---

**Last Updated**: October 2025
**Version**: 1.0
