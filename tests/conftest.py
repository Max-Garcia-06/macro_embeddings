"""Pytest configuration for text_cleaning tests."""

import sys
from pathlib import Path

# Add parent directory to path so text_cleaning can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
