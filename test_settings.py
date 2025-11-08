#!/usr/bin/env python
"""Test settings loading."""
import os
import sys

# Set environment variables
os.environ['DEVELOPER_MODE'] = 'true'
os.environ['DEBUG_MODE'] = 'true'

print("Environment variables set:")
print(f"  DEVELOPER_MODE = {os.getenv('DEVELOPER_MODE')}")
print(f"  DEBUG_MODE = {os.getenv('DEBUG_MODE')}")
print()

# Import and test
from src.config import get_settings

settings = get_settings()
print("Settings loaded:")
print(f"  developer_mode = {settings.developer_mode} (type: {type(settings.developer_mode)})")
print(f"  debug_mode = {settings.debug_mode} (type: {type(settings.debug_mode)})")
print()

# Test app/main.py loading
print("Testing app/main.py import...")
sys.path.insert(0, 'app')
import main
print(f"  DEVELOPER_MODE = {main.DEVELOPER_MODE}")
print(f"  debug_mode = {main.debug_mode}")

