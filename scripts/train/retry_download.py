#!/usr/bin/env python3
"""
Quick retry script to continue downloading failed images.
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python retry_download.py <dataset_directory>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    failed_file = dataset_dir / "failed_downloads.json"
    
    if not failed_file.exists():
        print("No failed downloads found. Either download was successful or hasn't been attempted.")
        return
    
    print(f"Found failed downloads file: {failed_file}")
    
    # Run the download script again with the same directory
    import subprocess
    cmd = [
        "python", "scripts/train/download_llava_pretrain.py",
        "--output_dir", str(dataset_dir),
        "--download_images",
        "--sample_size", "100"  # Adjust as needed
    ]
    
    print("Re-running download script...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 