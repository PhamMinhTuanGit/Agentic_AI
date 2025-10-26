"""
ZebOS HTML Documentation Parsers

This package contains parsers for extracting structured information
from ZebOS-XP HTML documentation files.
"""

from .zebos_html_parser import (
    ZebOSHTMLParser,
    CommandInfo,
    ChapterInfo
)

__all__ = [
    'ZebOSHTMLParser',
    'CommandInfo',
    'ChapterInfo'
]

__version__ = '1.0.0'
