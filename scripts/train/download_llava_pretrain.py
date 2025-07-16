#!/usr/bin/env python3
"""
Fixed script to download the LLaVA-Pretrain dataset (BLIP 558K) from Hugging Face.
Handles the schema inconsistency issues in the original dataset.

Dataset: liuhaotian/LLaVA-Pretrain
Target structure:
- /blip_558k/blip_558k_plain.json
- /blip_558k/images/
"""

import os
import json
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse
from urllib.parse import urlparse
import zipfile


def download_image(url, save_path, max_retries=3, timeout=60):
    """Download an image from URL with retry logic."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True, headers=headers)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                # Only print error for final attempt to reduce spam
                print(f"\n❌ Failed to download {url} after {max_retries} attempts: {e}")
                return False
            # Brief pause before retry
            import time
            time.sleep(1)
    return False


def download_dataset_files(output_dir):
    """Download the dataset files directly from HuggingFace."""
    print("📥 Downloading dataset files from HuggingFace...")
    
    files_to_download = [
        "blip_laion_cc_sbu_558k.json",
        "blip_laion_cc_sbu_558k_meta.json"
    ]
    
    downloaded_files = {}
    for filename in files_to_download:
        try:
            filepath = hf_hub_download(
                repo_id="liuhaotian/LLaVA-Pretrain",
                filename=filename,
                repo_type="dataset",
                cache_dir=output_dir / "cache"
            )
            downloaded_files[filename] = filepath
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    return downloaded_files


def process_dataset_files(downloaded_files, output_dir, sample_size=None):
    """Process the downloaded JSON files and create unified dataset."""
    
    # Load the conversation file (main dataset)
    conv_file = downloaded_files.get("blip_laion_cc_sbu_558k.json")
    meta_file = downloaded_files.get("blip_laion_cc_sbu_558k_meta.json")
    
    if not conv_file:
        print("❌ Main conversation file not found!")
        return None
    
    print(f"📖 Loading conversation data from: {conv_file}")
    with open(conv_file, 'r') as f:
        conv_data = json.load(f)
    
    # Load metadata if available
    meta_data = {}
    if meta_file:
        print(f"📖 Loading metadata from: {meta_file}")
        with open(meta_file, 'r') as f:
            meta_list = json.load(f)
            # Create a mapping from id to metadata
            for item in meta_list:
                meta_data[item['id']] = item
    
    # Process and combine data
    processed_data = []
    total_items = len(conv_data)
    
    if sample_size:
        conv_data = conv_data[:sample_size]
        total_items = min(sample_size, total_items)
    
    print(f"📊 Processing {total_items} items...")
    
    for i, item in enumerate(tqdm(conv_data, desc="Processing items")):
        # Get metadata for this item
        item_id = item.get('id', str(i))
        meta_info = meta_data.get(item_id, {})
        
        # Create processed entry
        processed_entry = {
            "id": item_id,
            "image": item.get('image', f"{item_id}.jpg"),
            "conversations": item.get('conversations', []),
            "url": meta_info.get('url', ''),
            "blip_caption": meta_info.get('blip_caption', '')
        }
        
        processed_data.append(processed_entry)
    
    # Save processed data
    output_file = output_dir / "blip_558k_plain.json"
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"✅ Saved processed data to: {output_file}")
    return processed_data


def download_images_from_urls(processed_data, images_dir):
    """Download images from URLs in the processed data."""
    print("🖼️  Downloading images from URLs...")
    
    failed_downloads = []
    success_count = 0
    
    for entry in tqdm(processed_data, desc="Downloading images"):
        image_url = entry.get('url', '')
        image_filename = entry['image']
        image_path = images_dir / image_filename
        
        # Create subdirectories if they don't exist
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if image already exists
        if image_path.exists():
            success_count += 1
            continue
        
        if image_url:
            success = download_image(image_url, image_path)
            if success:
                success_count += 1
            else:
                failed_downloads.append({
                    'id': entry['id'],
                    'url': image_url,
                    'filename': image_filename
                })
    
    print(f"✅ Successfully downloaded: {success_count}/{len(processed_data)} images")
    if failed_downloads:
        print(f"⚠️  Failed to download: {len(failed_downloads)} images")
    
    return failed_downloads


def download_images_zip(output_dir):
    """Download and extract the images.zip file."""
    print("📦 Downloading images.zip file...")
    
    try:
        zip_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Pretrain",
            filename="images.zip",
            repo_type="dataset",
            cache_dir=output_dir / "cache"
        )
        
        print(f"📁 Extracting images from: {zip_path}")
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        
        print("✅ Images extracted successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download/extract images.zip: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download LLaVA-Pretrain dataset (Fixed)')
    parser.add_argument('--output_dir', type=str, default='./blip_558k',
                        help='Output directory for dataset')
    parser.add_argument('--download_images', action='store_true',
                        help='Download images')
    parser.add_argument('--use_zip', action='store_true',
                        help='Use images.zip instead of downloading from URLs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Download only a sample of the dataset')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"📦 Downloading LLaVA-Pretrain dataset (Fixed Version)...")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Download dataset files
        downloaded_files = download_dataset_files(output_dir)
        
        if not downloaded_files:
            print("❌ No dataset files downloaded!")
            return 1
        
        # Process the files
        processed_data = process_dataset_files(downloaded_files, output_dir, args.sample_size)
        
        if not processed_data:
            print("❌ Failed to process dataset!")
            return 1
        
        # Download images if requested
        if args.download_images:
            if args.use_zip:
                success = download_images_zip(output_dir)
                if not success:
                    print("⚠️  Falling back to URL-based download...")
                    failed_downloads = download_images_from_urls(processed_data, images_dir)
            else:
                failed_downloads = download_images_from_urls(processed_data, images_dir)
                
                if failed_downloads:
                    # Save failed downloads
                    with open(output_dir / "failed_downloads.json", 'w') as f:
                        json.dump(failed_downloads, f, indent=2)
        
        # Create summary
        summary = {
            "dataset_name": "LLaVA-Pretrain (BLIP 558K)",
            "total_samples": len(processed_data),
            "json_file": str(output_dir / "blip_558k_plain.json"),
            "images_dir": str(images_dir),
            "format": "LLaVA training format",
            "usage": {
                "data_path": str(output_dir / "blip_558k_plain.json"),
                "image_folder": str(images_dir)
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Dataset download complete!")
        print(f"📄 JSON file: {output_dir / 'blip_558k_plain.json'}")
        print(f"📁 Images directory: {images_dir}")
        print(f"📋 Dataset info: {output_dir / 'dataset_info.json'}")
        
        # Print usage instructions
        print("\n" + "="*50)
        print("📚 USAGE INSTRUCTIONS")
        print("="*50)
        print(f"Update your training script with:")
        print(f"  --data_path {output_dir / 'blip_558k_plain.json'}")
        print(f"  --image_folder {images_dir}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Try running with --sample_size 100 for testing")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 