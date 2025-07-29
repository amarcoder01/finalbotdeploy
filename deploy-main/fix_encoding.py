#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix encoding issues in telegram_handler.py
"""

import codecs
import os

def fix_file_encoding(file_path):
    """Fix encoding issues in a file by reading with different encodings and writing as UTF-8"""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    content = None
    used_encoding = None
    
    # Try to read with different encodings
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError as e:
            print(f"Failed to read with {encoding}: {e}")
            continue
    
    if content is None:
        print("Could not read file with any encoding")
        return False
    
    # Write back as UTF-8
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully converted file from {used_encoding} to UTF-8")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

if __name__ == "__main__":
    file_path = "telegram_handler.py"
    if os.path.exists(file_path):
        print(f"Fixing encoding for {file_path}...")
        success = fix_file_encoding(file_path)
        if success:
            print("Encoding fix completed successfully!")
        else:
            print("Encoding fix failed!")
    else:
        print(f"File {file_path} not found!")