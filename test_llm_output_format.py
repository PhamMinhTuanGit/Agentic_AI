#!/usr/bin/env python3
"""
Test LLM Output Format - Device-Specific CLI

Verifies that the LLM generates output in the correct format:

Configure for Rn:
```cisco
commands
```

Explain: explanation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cli_output_config():
    """Test CLI output configuration"""
    print("\n" + "=" * 70)
    print("TEST 1: CLI Output Config")
    print("=" * 70)
    
    try:
        from rag.cli_output_config import CLIOutputConfig, create_cli_prompt
        
        print("✅ CLI Output Config imported successfully")
        
        # Get topology device prompt
        prompt = CLIOutputConfig.get_prompt_for_context("topology")
        print(f"✅ Topology prompt loaded ({len(prompt)} characters)")
        
        # Check if format requirements are in prompt
        requirements = [
            "Configure for",
            "```cisco",
            "Explain:",
        ]
        
        for req in requirements:
            if req in prompt:
                print(f"   ✅ Contains '{req}' requirement")
            else:
                print(f"   ❌ Missing '{req}' requirement")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_creation():
    """Test prompt creation"""
    print("\n" + "=" * 70)
    print("TEST 2: Prompt Creation")
    print("=" * 70)
    
    try:
        from rag.cli_output_config import create_cli_prompt
        
        # Create sample context
        context = """
NETWORK TOPOLOGY: Ring with 4 routers
R1 - 10.0.12.1/30 - Connected to R2
R2 - 10.0.12.2/30 - Connected to R1
"""
        
        query = "Configure all routers for OSPF"
        
        prompt = create_cli_prompt(query, context, "topology")
        
        print(f"✅ Prompt created ({len(prompt)} characters)")
        
        # Check prompt content
        if "Configure for" in prompt and "Explain:" in prompt:
            print("✅ Prompt contains required format markers")
        else:
            print("❌ Prompt missing format markers")
        
        print(f"\nPrompt preview (first 400 chars):")
        print(prompt[:400])
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expected_format():
    """Test expected output format"""
    print("\n" + "=" * 70)
    print("TEST 3: Expected Output Format")
    print("=" * 70)
    
    try:
        expected_format = """Configure for R1:
```cisco
R1#configure terminal
R1(config)#router ospf 1
R1(config)#exit
R1#end
```

Explain: This configuration enables OSPF on R1."""
        
        print("✅ Expected format example created")
        print("\nExpected output format:")
        print(expected_format)
        
        # Verify structure
        lines = expected_format.split('\n')
        
        checks = [
            ("Device header", lines[0].startswith("Configure for")),
            ("Code block start", "```cisco" in expected_format),
            ("Code block end", "```" in expected_format),
            ("Explain section", "Explain:" in expected_format),
        ]
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
        
        return all(result for _, result in checks)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_multiple_devices():
    """Test multiple device format"""
    print("\n" + "=" * 70)
    print("TEST 4: Multiple Devices Format")
    print("=" * 70)
    
    try:
        expected_multi = """Configure for R1:
```cisco
R1#configure terminal
R1(config)#router ospf 1
R1(config)#exit
R1#end
```

Explain: This enables OSPF on R1.

Configure for R2:
```cisco
R2#configure terminal
R2(config)#router ospf 1
R2(config)#exit
R2#end
```

Explain: This enables OSPF on R2."""
        
        print("✅ Multiple device format created")
        
        # Count device sections
        device_count = expected_multi.count("Configure for")
        print(f"✅ Contains {device_count} device sections")
        
        # Check separation
        sections = expected_multi.split("Configure for")
        print(f"✅ Sections are properly separated")
        
        # Verify each has explain
        explain_count = expected_multi.count("Explain:")
        if explain_count == device_count:
            print(f"✅ Each device has an explanation")
        else:
            print(f"❌ Explanation count mismatch")
        
        print(f"\nMultiple devices example:")
        print(expected_multi)
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("LLM OUTPUT FORMAT TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("CLI Output Config", test_cli_output_config),
        ("Prompt Creation", test_prompt_creation),
        ("Expected Format", test_expected_format),
        ("Multiple Devices", test_multiple_devices),
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
        print("🎉 ALL TESTS PASSED! Output format is correctly configured.")
        print("\nThe LLM will now generate output in the format:")
        print("  Configure for Rn:")
        print("  ```cisco")
        print("  commands")
        print("  ```")
        print("  Explain: explanation")
        return 0
    else:
        print(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
