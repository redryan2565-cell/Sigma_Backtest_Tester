#!/usr/bin/env python
"""Check if DEVELOPER_MODE and DEBUG_MODE are loaded correctly."""
import os
import sys

# Set environment variables
os.environ['DEVELOPER_MODE'] = 'true'
os.environ['DEBUG_MODE'] = 'true'
os.environ['DEBUG_SETTINGS'] = 'true'

print("=" * 60)
print("Environment Variables Check")
print("=" * 60)
print(f"DEVELOPER_MODE = {os.getenv('DEVELOPER_MODE')}")
print(f"DEBUG_MODE = {os.getenv('DEBUG_MODE')}")
print(f"DEBUG_SETTINGS = {os.getenv('DEBUG_SETTINGS')}")
print()

# Import settings
print("Loading Settings...")
from src.config import get_settings
settings = get_settings()
print(f"Settings.developer_mode = {settings.developer_mode} (type: {type(settings.developer_mode)})")
print(f"Settings.debug_mode = {settings.debug_mode} (type: {type(settings.debug_mode)})")
print()

# Import main module (simulate what Streamlit does)
print("Loading app/main.py module...")
sys.path.insert(0, 'app')
import importlib
import main
importlib.reload(main)

print("=" * 60)
print("Final Values in main.py")
print("=" * 60)
print(f"DEVELOPER_MODE = {main.DEVELOPER_MODE} (type: {type(main.DEVELOPER_MODE)})")
print(f"debug_mode = {main.debug_mode} (type: {type(main.debug_mode)})")
print()

# Check if they are True
if main.DEVELOPER_MODE:
    print("✅ DEVELOPER_MODE is ENABLED")
else:
    print("❌ DEVELOPER_MODE is DISABLED")

if main.debug_mode:
    print("✅ DEBUG_MODE is ENABLED")
else:
    print("❌ DEBUG_MODE is DISABLED")
print("=" * 60)


