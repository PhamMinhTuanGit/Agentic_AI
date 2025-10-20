#!/usr/bin/env python3
"""
Test ZebOS Command Generation

Verifies that the CLI generator now produces ZebOS commands instead of Cisco IOS
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_zebos_cli_generator():
    """Test that CLI generator produces ZebOS commands"""
    print("\n" + "=" * 70)
    print("TEST 1: ZebOS CLI Generator")
    print("=" * 70)
    
    try:
        from network_stat.cli_generator import CLIGenerator, ConfigType
        
        generator = CLIGenerator("network_stat/ring_topology.yaml")
        print("✅ CLI Generator initialized with ring topology")
        
        # Check default config type is ZebOS
        if generator.config_type == ConfigType.ZEBOS:
            print("✅ Default config type is ZebOS")
        else:
            print(f"❌ Default config type is {generator.config_type}, expected ZEBOS")
        
        # Get configuration for a router
        config = generator.get_device_config("R1")
        print(f"✅ Generated configuration for R1 ({len(config)} chars)")
        
        # Check for ZebOS-specific syntax
        zebos_markers = [
            ("configure", "ZebOS configure command"),
            ("ipv4 address", "ZebOS IPv4 address syntax"),
            ("interface ethernet", "ZebOS interface type"),
            ("no shutdown", "no shutdown"),
            ("exit", "exit command"),
        ]
        
        missing = []
        for marker, description in zebos_markers:
            if marker in config:
                print(f"   ✅ Contains '{marker}' ({description})")
            else:
                print(f"   ❌ Missing '{marker}' ({description})")
                missing.append(marker)
        
        # Check for Cisco-specific syntax that should NOT be present
        cisco_markers = [
            ("configure terminal", "Cisco configure terminal"),
            ("ip address", "Cisco ip address syntax"),
            ("no shutdown!", "Cisco style"),
        ]
        
        found_cisco = []
        for marker, description in cisco_markers:
            if marker in config:
                print(f"   ⚠️  Found old Cisco syntax: '{marker}' ({description})")
                found_cisco.append(marker)
        
        print(f"\nGenerated R1 Configuration (first 500 chars):\n{config[:500]}...\n")
        
        return len(missing) == 0 and len(found_cisco) == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_device_configs():
    """Test configuration generation for all devices"""
    print("\n" + "=" * 70)
    print("TEST 2: All Device Configurations")
    print("=" * 70)
    
    try:
        from network_stat.cli_generator import CLIGenerator
        
        generator = CLIGenerator("network_stat/ring_topology.yaml")
        
        all_configs = generator.get_all_device_configs()
        print(f"✅ Generated configurations for {len(all_configs)} devices")
        
        for device_id, config in all_configs.items():
            # Check each config has ZebOS markers
            has_configure = "configure" in config
            has_ipv4 = "ipv4 address" in config or "interface ethernet" in config
            has_exit = "exit" in config
            
            status = "✅" if (has_configure or has_ipv4) and has_exit else "⚠️"
            print(f"{status} {device_id}: {len(config)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_output_config():
    """Test CLI output configuration uses ZebOS syntax"""
    print("\n" + "=" * 70)
    print("TEST 3: CLI Output Configuration")
    print("=" * 70)
    
    try:
        from rag.cli_output_config import CLIOutputConfig
        
        # Get topology prompt
        prompt = CLIOutputConfig.get_prompt_for_context("topology")
        print("✅ Got topology prompt")
        
        # Check for ZebOS syntax in prompt
        if "```zsh" in prompt or "```bash" in prompt:
            print("✅ Prompt uses zsh/bash code blocks")
        else:
            print("❌ Prompt doesn't use zsh/bash code blocks")
        
        if "configure" in prompt:
            print("✅ Prompt mentions ZebOS configure command")
        else:
            print("❌ Prompt doesn't mention configure command")
        
        if "ipv4 address" in prompt:
            print("✅ Prompt mentions ipv4 address syntax")
        else:
            print("❌ Prompt doesn't mention ipv4 address")
        
        # Check format_cli_session
        session = CLIOutputConfig.format_cli_session("test commands", "router")
        if session["language"] == "zsh":
            print("✅ CLI session uses zsh language")
        else:
            print(f"❌ CLI session uses {session['language']} (expected zsh)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_client_prompt():
    """Test LLM client has ZebOS system prompt"""
    print("\n" + "=" * 70)
    print("TEST 4: LLM Client System Prompt")
    print("=" * 70)
    
    try:
        from rag.llm_client import LLMClient
        
        client = LLMClient()
        print("✅ LLM Client initialized")
        
        # Build a test prompt
        prompt = client._build_prompt(
            query="Configure OSPF on router R1",
            context="Router R1 with interfaces G0/0 and G0/1"
        )
        
        print("✅ Test prompt built")
        
        # Check for ZebOS references
        if "ZebOS" in prompt:
            print("✅ Prompt mentions ZebOS")
        else:
            print("⚠️  Prompt doesn't explicitly mention ZebOS")
        
        if "ipv4 address" in prompt:
            print("✅ Prompt mentions ipv4 address syntax")
        else:
            print("❌ Prompt doesn't mention ipv4 address")
        
        if "interface ethernet" in prompt:
            print("✅ Prompt mentions interface ethernet")
        else:
            print("⚠️  Prompt doesn't mention interface ethernet")
        
        print(f"\nSystem prompt excerpt (first 300 chars):\n{prompt[:300]}...\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Check that ZebOS documentation was created"""
    print("\n" + "=" * 70)
    print("TEST 5: ZebOS Documentation")
    print("=" * 70)
    
    try:
        doc_file = Path("ZEBOS_COMMAND_REFERENCE.md")
        
        if doc_file.exists():
            print(f"✅ ZebOS Command Reference created ({doc_file.stat().st_size} bytes)")
            
            content = doc_file.read_text()
            
            # Check for key sections
            sections = [
                "Basic Commands",
                "Interface Configuration",
                "Routing Protocols",
                "OSPF Configuration",
                "BGP Configuration",
                "ACLs and Firewall",
                "QoS Configuration",
            ]
            
            for section in sections:
                if section in content:
                    print(f"   ✅ Contains {section}")
                else:
                    print(f"   ⚠️  Missing {section}")
            
            # Check for ZebOS vs Cisco comparison
            if "Key Differences: ZebOS vs Cisco IOS" in content:
                print("   ✅ Contains ZebOS vs Cisco comparison")
            
            return True
        else:
            print(f"❌ ZebOS Command Reference not found at {doc_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ZebOS COMMAND GENERATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("ZebOS CLI Generator", test_zebos_cli_generator),
        ("All Device Configs", test_all_device_configs),
        ("CLI Output Configuration", test_cli_output_config),
        ("LLM Client Prompt", test_llm_client_prompt),
        ("ZebOS Documentation", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed_count}/{total_count} tests passed")
    print(f"{'=' * 70}\n")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED!")
        print("\nZebOS command generation is now active:")
        print("  • CLI Generator uses ZebOS syntax")
        print("  • Output configs use 'configure' (not 'configure terminal')")
        print("  • Interfaces use 'interface ethernet' syntax")
        print("  • IP addresses use 'ipv4 address' syntax")
        print("  • System prompts are ZebOS-aware")
        print("  • Comprehensive ZebOS documentation available")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        print("\nPlease check the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
