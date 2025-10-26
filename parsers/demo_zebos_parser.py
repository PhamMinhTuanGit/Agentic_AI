"""
Simple example of using the ZebOS HTML parser
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.zebos_html_parser import ZebOSHTMLParser, CommandInfo


def demo_parse_single_file():
    """Demo: Parse a single command file"""
    print("="*70)
    print("DEMO 1: Parsing a single command file")
    print("="*70)
    
    docs_dir = "/Users/phamminhtuan/Desktop/VHT_work/Agentic/ZebOS-XP_1.4_HTML"
    parser = ZebOSHTMLParser(docs_dir)
    
    # Parse AAA accounting command
    file_path = Path(docs_dir) / "ZebOS-XP 1.4" / "AAA Commands.603.02.html"
    
    if file_path.exists():
        cmd_info = parser.parse_command_file(file_path)
        
        if cmd_info:
            print(f"\nCommand Name: {cmd_info.name}")
            print(f"Chapter: {cmd_info.chapter}")
            print(f"\nDescription:\n{cmd_info.description}")
            print(f"\nSyntax:")
            for syntax in cmd_info.syntax:
                print(f"  {syntax}")
            
            print(f"\nParameters:")
            for param in cmd_info.parameters:
                print(f"  - {param['name']}: {param['description']}")
            
            print(f"\nMode: {cmd_info.mode}")
            
            if cmd_info.examples:
                print(f"\nExamples:")
                for example in cmd_info.examples:
                    print(f"  {example}")
        else:
            print("Failed to parse the file")
    else:
        print(f"File not found: {file_path}")


def demo_parse_all_and_search():
    """Demo: Parse all commands and search"""
    print("\n" + "="*70)
    print("DEMO 2: Parse all commands and search")
    print("="*70)
    
    docs_dir = "/Users/phamminhtuan/Desktop/VHT_work/Agentic/ZebOS-XP_1.4_HTML"
    parser = ZebOSHTMLParser(docs_dir)
    
    # Parse first 10 command files for demo
    print("\nParsing command files (limited to 10 for demo)...")
    commands = []
    count = 0
    
    html_dir = Path(docs_dir) / "ZebOS-XP 1.4"
    for html_file in sorted(html_dir.glob("AAA Commands.*.html")):
        if count >= 10:
            break
        cmd_info = parser.parse_command_file(html_file)
        if cmd_info:
            commands.append(cmd_info)
            count += 1
            print(f"  {count}. {cmd_info.name}")
    
    # Search for specific keywords
    print(f"\n\nSearching for 'authentication' in {len(commands)} commands...")
    results = parser.search_commands("authentication", commands)
    print(f"Found {len(results)} matching commands:")
    for cmd in results:
        print(f"  - {cmd.name}")


def demo_extract_all_chapters():
    """Demo: Extract all chapter information"""
    print("\n" + "="*70)
    print("DEMO 3: Extract chapter information")
    print("="*70)
    
    docs_dir = "/Users/phamminhtuan/Desktop/VHT_work/Agentic/ZebOS-XP_1.4_HTML"
    parser = ZebOSHTMLParser(docs_dir)
    
    # Parse chapter files (limited for demo)
    print("\nParsing chapter files (first 5 chapters)...")
    chapters = []
    count = 0
    
    html_dir = Path(docs_dir) / "ZebOS-XP 1.4"
    for html_file in sorted(html_dir.glob("*.01.html")):
        if count >= 5:
            break
        chapter_info = parser.parse_chapter_file(html_file)
        if chapter_info and chapter_info.chapter_number:
            chapters.append(chapter_info)
            count += 1
            print(f"\n{chapter_info.chapter_number}{chapter_info.title}")
            print(f"  Commands: {len(chapter_info.commands)}")
            if chapter_info.introduction:
                intro_preview = chapter_info.introduction[:100] + "..."
                print(f"  Intro: {intro_preview}")


def demo_json_export():
    """Demo: Export parsed data to JSON"""
    print("\n" + "="*70)
    print("DEMO 4: Export to JSON")
    print("="*70)
    
    docs_dir = "/Users/phamminhtuan/Desktop/VHT_work/Agentic/ZebOS-XP_1.4_HTML"
    parser = ZebOSHTMLParser(docs_dir)
    
    # Parse limited set of commands
    print("\nParsing AAA commands...")
    commands = []
    
    html_dir = Path(docs_dir) / "ZebOS-XP 1.4"
    for html_file in sorted(html_dir.glob("AAA Commands.*.html")):
        cmd_info = parser.parse_command_file(html_file)
        if cmd_info:
            commands.append(cmd_info)
    
    print(f"Parsed {len(commands)} AAA commands")
    
    # Save to JSON
    output_file = "demo_aaa_commands.json"
    parser.save_to_json(commands, output_file)
    print(f"\nData exported to: {output_file}")


def main():
    """Run all demos"""
    print("\n" + "🔍 ZebOS HTML Parser - Demo Examples\n")
    
    try:
        # Run demos
        demo_parse_single_file()
        demo_parse_all_and_search()
        demo_extract_all_chapters()
        demo_json_export()
        
        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
