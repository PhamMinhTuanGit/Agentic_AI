"""
CLI Output Configuration for RAG Pipeline
==========================================

Configures LLM to generate CLI commands in single code blocks per session
with proper formatting, syntax highlighting, and session management.
"""

from typing import Optional, Dict, Any


class CLIOutputConfig:
    """Configuration for CLI command generation and formatting"""
    
    # System prompt for CLI generation with single code block output
    CLI_SYSTEM_PROMPT = """You are an expert ZebOS network device configuration assistant specialized in:

## Expertise Areas:
1. **ZebOS CLI Configuration**: Generating CLI commands for network device configuration
2. **Routing Protocols**: BGP, OSPF, EIGRP, IS-IS, RIP configuration
3. **Network Interfaces**: Port configuration, VLAN setup, LAG/Port-channel
4. **ACLs & Security**: Access control lists, firewall rules, AAA authentication
5. **QoS**: Quality of Service policies and traffic shaping
6. **High Availability**: Redundancy, failover, and clustering
7. **Monitoring**: SNMP, syslog, NetFlow configuration

## Response Format for Network Configuration:
- **For CLI requests**: Provide ALL commands in a SINGLE code block
- **For configuration**: Include step-by-step instructions in ONE code block
- **For troubleshooting**: All diagnostic commands in ONE code block
- **For examples**: Complete configuration block in ONE code block

## CRITICAL REQUIREMENTS FOR CLI CODE BLOCKS:
1. Use ONLY ONE code block per session/request
2. Start code block with: ```
3. Specify language: ```zsh or ```bash or ```cisco
4. Include device prompt (R1#, R2(config)#, SW1#, etc.)
5. Each command on a new line
6. Add inline comments after commands (e.g., # Enable routing)
7. Include output validation commands at the end
8. End code block with: ```
9. NO multiple separate code blocks - combine all commands in ONE block
10. NO mixing of different output formats - keep it consistent

## Instructions:
1. Answer based ONLY on the provided context
2. When generating CLI commands, ensure they are valid ZebOS syntax
3. Include comments (!) to explain complex configurations
4. If the answer is not in the context, say "I don't have enough information to answer this question"
5. Be concise, accurate, and provide working configurations
6. For multi-step configurations, number the steps clearly as comments
7. Highlight important warnings or prerequisites with ⚠️
8. Cite relevant parts of the context when appropriate

## Single Code Block Example (REQUIRED FORMAT):

```zsh
! Step 1: Configure Router R1 for OSPF
R1#configure
R1(config)#router ospf 100
R1(config-router)#network 10.1.1.0 0.0.0.255 area 0
R1(config-router)#network 1.1.1.1 0.0.0.0 area 0
R1(config-router)#bfd all-interfaces
R1(config-router)#exit

! Step 2: Configure Interface ethernet G0/0
R1(config)#interface ethernet G0/0
R1(config-if)#ipv4 address 10.1.1.1 255.255.255.0
R1(config-if)#no shutdown
R1(config-if)#exit

! Step 3: Verify Configuration
R1#show ip ospf neighbor
R1#show ip route
R1#show interface brief
```

## Session Management:
- Each user request is a NEW SESSION
- All commands for that session go into ONE code block
- Previous sessions are NOT referenced
- Each session starts fresh with full context

## DO NOT:
- Create multiple code blocks in one response
- Mix code blocks with explanatory text blocks
- Use different formatting styles within one session
- Split commands across multiple blocks
- Add extra line breaks between sections within the code block

## Output Order:
1. Brief explanation (optional, before code block)
2. ONE COMPLETE CODE BLOCK with all commands
3. Brief validation notes (optional, after code block)
4. Link to documentation if needed (optional)

Remember: ONE code block per session, ALL commands combined.
"""

    # System prompt for CLI generation with device-specific sections
    TOPOLOGY_DEVICE_PROMPT = """You are an expert ZebOS network device configuration assistant.

## Output Format (CRITICAL - MUST FOLLOW):

For each device, format output EXACTLY like this:

Configure for R1:
```zsh
! Configuration commands for R1
R1#configure
R1(config)#router ospf 1
... all commands ...
R1(config)#exit
```

Explain: Brief explanation of what these commands do and why.

Configure for R2:
```zsh
! Configuration commands for R2
R2#configure
... all commands ...
R2(config)#exit
```

Explain: Brief explanation of R2 configuration.

## REQUIREMENTS:
1. Start each device section with: "Configure for Rn:" where n is the device number
2. Follow with ONE code block containing ALL commands for that device
3. Use ```zsh for the language marker (or ```bash)
4. Each code block must contain complete, executable ZebOS commands
5. After code block, add "Explain: " section with brief explanation
6. Separate each device section with a blank line
7. Do NOT use multiple code blocks per device
8. Do NOT mix different devices in one code block

## ZebOS Device Configuration Rules:
- Use "configure" (not "configure terminal")
- Use "ipv4 address" (not "ip address")
- Use "interface ethernet" (not just "interface")
- Include "exit" after each configuration section
- Include validation commands at the end (show commands)
- Make commands ready to copy-paste into ZebOS CLI

## Answer Based on:
- Provided network topology
- Device connections and roles
- Best practices from ZebOS documentation
"""

    @staticmethod
    def get_prompt_for_context(context_type: str = "general") -> str:
        """
        Get appropriate system prompt based on context type
        
        Args:
            context_type: Type of context (general, router, switch, topology)
        
        Returns:
            System prompt string
        """
        prompts = {
            "general": CLIOutputConfig.CLI_SYSTEM_PROMPT,
            "topology": CLIOutputConfig.TOPOLOGY_DEVICE_PROMPT,
        }
        return prompts.get(context_type, CLIOutputConfig.CLI_SYSTEM_PROMPT)

    @staticmethod
    def format_cli_session(commands: str, device_type: str = "router", session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Format CLI commands for a session
        
        Args:
            commands: Raw CLI commands
            device_type: Type of device (router, switch, host)
            session_id: Optional session identifier
        
        Returns:
            Formatted session object
        """
        return {
            "session_id": session_id,
            "device_type": device_type,
            "commands": commands,
            "format": "single_code_block",
            "language": "zsh",
            "validated": False
        }

    @staticmethod
    def create_session_prompt(session_context: Dict[str, Any]) -> str:
        """
        Create a session-specific prompt
        
        Args:
            session_context: Session context information
        
        Returns:
            Session prompt
        """
        context_type = session_context.get("type", "general")
        device = session_context.get("device", "").upper()
        network_info = session_context.get("network_info", "")
        
        prompt = f"""Current Session Configuration:
- Device: {device}
- Type: {context_type}
- Network: {network_info}

Generate all CLI commands in a SINGLE code block.
Do NOT create multiple code blocks.
Start with 'configure terminal' and end with 'exit'.
Include all necessary commands in sequence.
"""
        return prompt


class SessionCLIManager:
    """Manages CLI generation sessions with single code block output"""
    
    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_count = 0
    
    def create_session(self, device_id: str, device_type: str) -> str:
        """
        Create a new CLI generation session
        
        Args:
            device_id: Device identifier
            device_type: Type of device
        
        Returns:
            Session ID
        """
        self.session_count += 1
        session_id = f"session_{device_id}_{self.session_count}"
        
        self.sessions[session_id] = {
            "device_id": device_id,
            "device_type": device_type,
            "commands": [],
            "created_at": None,
            "status": "active"
        }
        
        return session_id
    
    def add_command_to_session(self, session_id: str, command: str) -> None:
        """
        Add command to session (accumulates in list, combines later)
        
        Args:
            session_id: Session identifier
            command: Command to add
        """
        if session_id in self.sessions:
            self.sessions[session_id]["commands"].append(command)
    
    def finalize_session(self, session_id: str) -> str:
        """
        Finalize session and return single code block
        
        Args:
            session_id: Session identifier
        
        Returns:
            Formatted code block with all commands
        """
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        device_type = session["device_type"]
        commands = session["commands"]
        
        # Combine all commands into single code block
        language = "cisco" if device_type in ["router", "switch"] else "bash"
        
        code_block = f"```{language}\n"
        code_block += "\n".join(commands)
        code_block += "\n```"
        
        session["status"] = "finalized"
        
        return code_block
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        return self.sessions.get(session_id, {})
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global session manager instance
_session_manager = SessionCLIManager()


def get_session_manager() -> SessionCLIManager:
    """Get global session manager instance"""
    return _session_manager


def create_cli_prompt(query: str, context: str, session_type: str = "general", output_type: str = "default") -> str:
    """
    Create a complete prompt for CLI generation in single code block format
    
    Args:
        query: User question
        context: Retrieved context
        session_type: Type of session (general, router, switch, topology)
    
    Returns:
        Complete prompt
    """
    system_prompt = CLIOutputConfig.get_prompt_for_context(session_type)
    if output_type == "single_code_block":
        system_prompt += "\nRemember: Generate ALL commands in a SINGLE code block. Do not create multiple code blocks."
    if output_type == "multi_code_block":
        system_prompt += "\nRemember: You can create MULTIPLE code blocks if needed for different devices or sections."
    else:
        pass
    prompt = f"""{system_prompt}

Context:
{context}

Question: {query}


Answer:"""
    
    return prompt


# Example usage documentation
"""
USAGE EXAMPLES:

1. Using with RAG Pipeline:
   
   from rag.cli_output_config import create_cli_prompt
   
   prompt = create_cli_prompt(
       query="Configure R1 for OSPF routing",
       context=topology_context,
       session_type="router"
   )
   
   result = llm_client.generate(query, context, system_prompt=prompt)

2. Using Session Manager:
   
   from rag.cli_output_config import get_session_manager
   
   manager = get_session_manager()
   session_id = manager.create_session("R1", "router")
   
   # LLM generates all commands in one block
   commands = generate_for_session(session_id)
   
   output = manager.finalize_session(session_id)
   # Returns: ```cisco\n...all commands...\n```

3. In Pipeline:
   
   pipeline = RAGPipeline(
       cli_output_config=CLIOutputConfig(),
       enable_topology=True
   )
   
   result = pipeline.query(
       "Configure R1 and R2 for OSPF",
       output_format="single_code_block"
   )
"""
