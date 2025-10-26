"""
ZebOS-XP HTML Documentation Parser

This module parses ZebOS-XP HTML documentation files to extract:
- Commands and their syntax
- Parameters and descriptions
- Configuration examples
- Chapter and section information
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
import json


@dataclass
class CommandInfo:
    """Represents a ZebOS command with its details"""
    name: str
    description: str
    syntax: List[str]
    parameters: List[Dict[str, str]]
    mode: str
    examples: List[str]
    file_path: str
    chapter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterInfo:
    """Represents a chapter/section in the documentation"""
    title: str
    chapter_number: Optional[str]
    introduction: str
    commands: List[str]
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ZebOSHTMLParser:
    """Parser for ZebOS-XP HTML documentation"""
    
    def __init__(self, docs_directory: str):
        """
        Initialize the parser
        
        Args:
            docs_directory: Path to ZebOS-XP_1.4_HTML directory
        """
        self.docs_directory = Path(docs_directory)
        self.html_dir = self.docs_directory / "ZebOS-XP 1.4"
        
    def parse_command_file(self, file_path: Path) -> Optional[CommandInfo]:
        """
        Parse a single command HTML file
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            CommandInfo object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract command name from title or heading
            title = soup.find('title')
            command_name = title.text.strip() if title else ""
            
            # Try to get from Heading1NewPage or Heading1
            heading = soup.find('div', class_=['Heading1NewPage', 'Heading1'])
            if heading:
                command_name = heading.get_text(strip=True)
            
            # Extract description (usually in Body class divs)
            description_parts = []
            for body_div in soup.find_all('div', class_='Body'):
                text = body_div.get_text(strip=True)
                if text and not text.startswith('Use the'):
                    description_parts.append(text)
            description = ' '.join(description_parts[:2]) if description_parts else ""
            
            # Extract command syntax
            syntax = []
            for cmd_div in soup.find_all('div', class_=['CmdMaster', 'Command']):
                syntax_text = cmd_div.get_text(strip=True)
                if syntax_text:
                    syntax.append(syntax_text)
            
            # Extract parameters
            parameters = []
            param_name = None
            param_desc = None
            
            for div in soup.find_all('div'):
                class_list = div.get('class', [])
                if not class_list:
                    continue
                    
                class_name = class_list[0] if isinstance(class_list, list) else class_list
                
                if class_name in ['CmdHead1', 'CmdHead2', 'CmdHead3']:
                    # Save previous parameter if exists
                    if param_name and param_desc:
                        parameters.append({
                            'name': param_name,
                            'description': param_desc
                        })
                    param_name = div.get_text(strip=True)
                    param_desc = None
                elif class_name in ['Cmd1', 'Cmd2'] and param_name:
                    param_desc = div.get_text(strip=True)
            
            # Add last parameter
            if param_name and param_desc:
                parameters.append({
                    'name': param_name,
                    'description': param_desc
                })
            
            # Extract command mode
            mode = ""
            for div in soup.find_all('div', class_='Heading3'):
                if 'Command Mode' in div.get_text():
                    next_div = div.find_next_sibling('div')
                    if next_div:
                        mode = next_div.get_text(strip=True)
                    break
            
            # Extract examples
            examples = []
            example_section = False
            for div in soup.find_all('div'):
                class_list = div.get('class', [])
                if not class_list:
                    continue
                    
                class_name = class_list[0] if isinstance(class_list, list) else class_list
                
                if class_name == 'Heading3' and 'Example' in div.get_text():
                    example_section = True
                    continue
                    
                if example_section and class_name == 'CmdExample':
                    examples.append(div.get_text(strip=True))
            
            # Extract chapter info from breadcrumbs
            chapter = None
            breadcrumb = soup.find('div', class_='ww_skin_breadcrumbs_parent')
            if breadcrumb:
                chapter_link = breadcrumb.find('a')
                if chapter_link:
                    chapter = chapter_link.get_text(strip=True)
            
            if not command_name:
                return None
                
            return CommandInfo(
                name=command_name,
                description=description,
                syntax=syntax,
                parameters=parameters,
                mode=mode,
                examples=examples,
                file_path=str(file_path.relative_to(self.docs_directory)),
                chapter=chapter
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def parse_chapter_file(self, file_path: Path) -> Optional[ChapterInfo]:
        """
        Parse a chapter introduction file
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            ChapterInfo object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract chapter title
            title_div = soup.find('div', class_='ChapterTitle')
            if not title_div:
                return None
                
            title_text = title_div.get_text(strip=True)
            
            # Extract chapter number if present
            chapter_num = None
            chapter_span = title_div.find('span', class_='chapter')
            if chapter_span:
                chapter_num = chapter_span.get_text(strip=True)
                # Remove chapter number from title
                title_text = title_text.replace(chapter_num, '').strip()
            
            # Extract introduction text
            intro_parts = []
            for body_div in soup.find_all('div', class_='Body'):
                intro_parts.append(body_div.get_text(strip=True))
            introduction = ' '.join(intro_parts)
            
            # Extract command list
            commands = []
            for bulleted_div in soup.find_all('div', class_=['Bulleted1', 'Bulleted']):
                link = bulleted_div.find('a')
                if link:
                    cmd_name = link.get_text(strip=True)
                    commands.append(cmd_name)
            
            return ChapterInfo(
                title=title_text,
                chapter_number=chapter_num,
                introduction=introduction,
                commands=commands,
                file_path=str(file_path.relative_to(self.docs_directory))
            )
            
        except Exception as e:
            print(f"Error parsing chapter {file_path}: {e}")
            return None
    
    def parse_all_commands(self, output_file: Optional[str] = None) -> List[CommandInfo]:
        """
        Parse all command files in the documentation
        
        Args:
            output_file: Optional path to save results as JSON
            
        Returns:
            List of CommandInfo objects
        """
        commands = []
        
        # Pattern to identify command files (e.g., "AAA Commands.603.02.html")
        command_pattern = re.compile(r'.+Commands?\.\d+\.\d+\.html$')
        
        if not self.html_dir.exists():
            print(f"Directory not found: {self.html_dir}")
            return commands
        
        for html_file in self.html_dir.glob("*.html"):
            # Check if it's a command file
            if command_pattern.match(html_file.name):
                cmd_info = self.parse_command_file(html_file)
                if cmd_info:
                    commands.append(cmd_info)
                    print(f"Parsed: {cmd_info.name}")
        
        print(f"\nTotal commands parsed: {len(commands)}")
        
        # Save to JSON if output file specified
        if output_file:
            self.save_to_json(commands, output_file)
        
        return commands
    
    def parse_all_chapters(self, output_file: Optional[str] = None) -> List[ChapterInfo]:
        """
        Parse all chapter introduction files
        
        Args:
            output_file: Optional path to save results as JSON
            
        Returns:
            List of ChapterInfo objects
        """
        chapters = []
        
        if not self.html_dir.exists():
            print(f"Directory not found: {self.html_dir}")
            return chapters
        
        # Pattern to identify chapter files (e.g., "AAA Commands.603.01.html")
        chapter_pattern = re.compile(r'.+\.\d+\.01\.html$')
        
        for html_file in self.html_dir.glob("*.html"):
            if chapter_pattern.match(html_file.name):
                chapter_info = self.parse_chapter_file(html_file)
                if chapter_info and chapter_info.chapter_number:
                    chapters.append(chapter_info)
                    print(f"Parsed chapter: {chapter_info.title}")
        
        print(f"\nTotal chapters parsed: {len(chapters)}")
        
        # Save to JSON if output file specified
        if output_file:
            self.save_to_json(chapters, output_file)
        
        return chapters
    
    def save_to_json(self, data: List, output_file: str):
        """
        Save parsed data to JSON file
        
        Args:
            data: List of CommandInfo or ChapterInfo objects
            output_file: Path to output JSON file
        """
        try:
            json_data = [item.to_dict() for item in data]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nSaved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving to JSON: {e}")
    
    def search_commands(self, keyword: str, commands: List[CommandInfo]) -> List[CommandInfo]:
        """
        Search for commands matching a keyword
        
        Args:
            keyword: Search keyword
            commands: List of CommandInfo objects to search
            
        Returns:
            List of matching CommandInfo objects
        """
        keyword_lower = keyword.lower()
        results = []
        
        for cmd in commands:
            if (keyword_lower in cmd.name.lower() or 
                keyword_lower in cmd.description.lower() or
                any(keyword_lower in syntax.lower() for syntax in cmd.syntax)):
                results.append(cmd)
        
        return results
    
    def get_commands_by_chapter(self, chapter_name: str, 
                                commands: List[CommandInfo]) -> List[CommandInfo]:
        """
        Get all commands from a specific chapter
        
        Args:
            chapter_name: Name of the chapter
            commands: List of CommandInfo objects
            
        Returns:
            List of CommandInfo objects from that chapter
        """
        chapter_lower = chapter_name.lower()
        results = []
        
        for cmd in commands:
            if cmd.chapter and chapter_lower in cmd.chapter.lower():
                results.append(cmd)
        
        return results


def main():
    """Example usage of the parser"""
    
    # Initialize parser
    docs_dir = "ZebOS-XP_1.4_HTML"
    parser = ZebOSHTMLParser(docs_dir)
    
    # Parse all commands
    print("Parsing all commands...")
    commands = parser.parse_all_commands(output_file="zebos_commands.json")
    
    # Parse all chapters
    print("\n" + "="*60)
    print("Parsing all chapters...")
    chapters = parser.parse_all_chapters(output_file="zebos_chapters.json")
    
    # Example: Search for AAA commands
    print("\n" + "="*60)
    print("Searching for 'aaa' commands...")
    aaa_commands = parser.search_commands("aaa", commands)
    print(f"Found {len(aaa_commands)} AAA commands")
    for cmd in aaa_commands[:5]:  # Show first 5
        print(f"  - {cmd.name}")
    
    # Example: Get commands from a specific chapter
    print("\n" + "="*60)
    print("Getting commands from 'Authentication' chapter...")
    auth_commands = parser.get_commands_by_chapter("Authentication", commands)
    print(f"Found {len(auth_commands)} authentication commands")
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS:")
    print(f"Total commands: {len(commands)}")
    print(f"Total chapters: {len(chapters)}")
    print(f"Commands with examples: {sum(1 for c in commands if c.examples)}")
    print(f"Commands with parameters: {sum(1 for c in commands if c.parameters)}")


if __name__ == "__main__":
    main()
