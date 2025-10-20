# LLM Output Format Configuration - Device-Specific CLI

**Date:** October 20, 2025  
**Status:** ✅ **CONFIGURED**

---

## 📋 Overview

The LLM has been configured to generate device-specific CLI configurations with a clear, structured format that includes:
- Device name header
- Complete CLI commands in a single code block
- Explanations for each device configuration

---

## 🎯 Output Format

The LLM now generates output in this exact format:

```
Configure for R1:
```cisco
! Configuration for Router R1
R1#configure terminal
R1(config)#router ospf 1
R1(config-router)#network 10.1.1.0 0.0.0.255 area 0
R1(config-router)#exit
R1(config)#interface G0/0
R1(config-if)#ip address 10.0.12.1 255.255.255.0
R1(config-if)#no shutdown
R1(config-if)#exit
R1(config)#exit
R1#end
```

Explain: This configuration sets up OSPF routing on R1, advertising the 10.1.1.0/24 network in area 0. It also configures interface G0/0 with IP address 10.0.12.1/24 which connects to R2.

Configure for R2:
```cisco
! Configuration for Router R2
R2#configure terminal
R2(config)#router ospf 1
R2(config-router)#network 10.2.2.0 0.0.0.255 area 0
R2(config-router)#exit
R2(config)#interface G0/0
R2(config-if)#ip address 10.0.12.2 255.255.255.0
R2(config-if)#no shutdown
R2(config-if)#exit
R2(config)#exit
R2#end
```

Explain: This configuration sets up OSPF on R2 and configures the interface that connects back to R1.
```

---

## 🔧 Format Specifications

### Device Section Header
- **Format:** `Configure for [DEVICE_ID]:`
- **Example:** `Configure for R1:`, `Configure for SW1:`, `Configure for PC1:`
- **Purpose:** Clearly identifies which device the commands are for

### Code Block
- **Start marker:** ` ```cisco `
- **Contents:** All CLI commands for that device
- **End marker:** ` ``` `
- **Language:** Always `cisco` for network device configurations
- **Content:** Complete, ready-to-execute commands

### Explanation Section
- **Format:** `Explain: [explanation text]`
- **Purpose:** Brief explanation of what the configuration does and why
- **Length:** 1-3 sentences, concise but informative

### Multiple Devices
- Separate each device section with a blank line
- Follow same format for each device
- Commands specific to each device only

---

## 💡 Example Output

### Query
```
"Configure all routers in the ring topology for OSPF routing"
```

### LLM Output

```
Configure for R1:
```cisco
! Enable OSPF on R1
R1#configure terminal
R1(config)#router ospf 1
R1(config-router)#network 10.0.12.0 0.0.0.3 area 0
R1(config-router)#network 10.0.14.0 0.0.0.3 area 0
R1(config-router)#bfd all-interfaces
R1(config-router)#exit
R1(config)#exit
R1#
```

Explain: This enables OSPF on R1 with two networks (connections to R2 and R4) in area 0. BFD is enabled for faster failure detection between neighbors.

Configure for R2:
```cisco
! Enable OSPF on R2
R2#configure terminal
R2(config)#router ospf 1
R2(config-router)#network 10.0.12.0 0.0.0.3 area 0
R2(config-router)#network 10.0.23.0 0.0.0.3 area 0
R2(config-router)#bfd all-interfaces
R2(config-router)#exit
R2(config)#exit
R2#
```

Explain: R2 is configured similarly, advertising its connections to R1 and R3 in OSPF area 0.

Configure for R3:
```cisco
! Enable OSPF on R3
R3#configure terminal
R3(config)#router ospf 1
R3(config-router)#network 10.0.23.0 0.0.0.3 area 0
R3(config-router)#network 10.0.34.0 0.0.0.3 area 0
R3(config-router)#bfd all-interfaces
R3(config-router)#exit
R3(config)#exit
R3#
```

Explain: R3 connects to R2 and R4. This configuration enables OSPF with BFD for redundancy in the ring topology.

Configure for R4:
```cisco
! Enable OSPF on R4
R4#configure terminal
R4(config)#router ospf 1
R4(config-router)#network 10.0.34.0 0.0.0.3 area 0
R4(config-router)#network 10.0.14.0 0.0.0.3 area 0
R4(config-router)#bfd all-interfaces
R4(config-router)#exit
R4(config)#exit
R4#
```

Explain: R4 completes the ring by connecting back to R1 and R3. All four routers now form a redundant OSPF network.
```

---

## 🚀 Usage

### Initialize Pipeline with Device Format
```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    enable_topology=True,
    enable_cli_format=True,
    cli_output_format="single_code_block"
)
```

### Query for Device Configurations
```python
result = pipeline.query(
    "Configure all routers for OSPF routing",
    output_format="single_code_block"
)

print(result['answer'])
```

### Output Will Include
- ✅ Configure for R1: ... Explain: ...
- ✅ Configure for R2: ... Explain: ...
- ✅ Configure for R3: ... Explain: ...
- ✅ Configure for R4: ... Explain: ...

---

## 📊 System Prompt

The LLM uses this system prompt for device-specific output:

```
You are an expert ZebOS network device configuration assistant.

## Output Format (CRITICAL - MUST FOLLOW):

For each device, format output EXACTLY like this:

Configure for R1:
```cisco
! Configuration commands for R1
R1#configure terminal
... all commands ...
R1#end
```

Explain: Brief explanation of what these commands do and why.

## REQUIREMENTS:
1. Start each device section with: "Configure for Rn:" where n is the device number
2. Follow with ONE code block containing ALL commands for that device
3. Use ```cisco for the language marker
4. Each code block must contain complete, executable commands
5. After code block, add "Explain: " section with brief explanation
6. Separate each device section with a blank line
7. Do NOT use multiple code blocks per device
8. Do NOT mix different devices in one code block
```

---

## ✅ Implementation Details

### Files Modified
- **rag/cli_output_config.py** - Updated prompts and format specifications
- **rag/llm_client.py** - Integrated CLI output config (previous step)
- **rag/pipeline.py** - Integrated CLI configuration (previous step)

### Key Features
1. **Device-Specific Headers** - Clear identification of which device
2. **Single Code Block** - All commands for one device in one block
3. **Language Specification** - Cisco syntax highlighting
4. **Explanation Section** - Context for each configuration
5. **Clean Separation** - Multiple devices clearly separated

---

## 🎯 Benefits

| Aspect | Benefit |
|--------|---------|
| **Clarity** | Each device's config is clearly separated |
| **Usability** | Commands are copy-paste ready |
| **Understanding** | Explanations help users understand why |
| **Scalability** | Supports multiple devices in one response |
| **Maintenance** | Easy to identify which device each command is for |

---

## 📝 Common Patterns

### Single Device Configuration
```
Query: "Configure R1 for OSPF"

Output:
Configure for R1:
```cisco
R1#configure terminal
...
```

Explain: This configures R1...
```

### Multiple Device Configuration
```
Query: "Configure all routers in the ring topology"

Output:
Configure for R1:
```cisco
...
```

Explain: ...

Configure for R2:
```cisco
...
```

Explain: ...

[continues for each device]
```

### Device-Specific Query
```
Query: "Show interface configuration for R2"

Output:
Configure for R2:
```cisco
R2#show interface brief
R2#show interface G0/0
R2#show ip interface brief
```

Explain: These commands display...
```

---

## 🔄 Integration with Ring Topology

The pipeline automatically detects topology and uses device-specific format:

```python
# Automatically uses TOPOLOGY_DEVICE_PROMPT
result = pipeline.query("Configure all routers")

# Output will be:
# Configure for R1: ... Explain: ...
# Configure for R2: ... Explain: ...
# Configure for R3: ... Explain: ...
# Configure for R4: ... Explain: ...
```

---

## 🛠️ Configuration Options

### In RAGPipeline
```python
RAGPipeline(
    enable_cli_format=True,           # Enable format
    cli_output_format="single_code_block"  # Format type
)
```

### In query() method
```python
pipeline.query(
    "Your question",
    output_format="single_code_block"
)
```

---

## 📋 Prompt Structure

The system now uses:
- **General queries:** `CLI_SYSTEM_PROMPT` - Flexible format
- **Topology queries:** `TOPOLOGY_DEVICE_PROMPT` - Device-specific format

Selection is automatic based on context type.

---

## ✨ Key Improvements

✅ **Clear Device Identification** - No ambiguity about which device
✅ **Single Code Block** - All commands together, no fragmentation
✅ **Explanations** - Users understand the configuration
✅ **Scalable** - Works with 1 to N devices
✅ **Professional** - Clean, organized output
✅ **Copy-Paste Ready** - Commands ready to deploy
✅ **Context-Aware** - Different format based on query type

---

## 🎉 Summary

The LLM is now configured to generate device-specific CLI configurations with:
- Clear device headers (Configure for Rn:)
- Single code blocks with all commands
- Explanations for each configuration
- Professional, organized output
- Ready for immediate deployment

**Status:** ✅ **READY FOR USE**

---

**Last Updated:** October 20, 2025  
**Configuration:** ✅ Complete  
**Testing:** Ready
