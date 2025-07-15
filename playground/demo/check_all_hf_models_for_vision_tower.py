#!/usr/bin/env python3
"""
Fixed script to check all HF models for vision tower
"""

import requests
import json
import time

def main():
    # Search for models
    search_url = 'https://huggingface.co/api/models?search=LLAVA&limit=400&sort=downloads&direction=-1'
    models = requests.get(search_url).json()

    print(f'📄 Checking {len(models)} models...\n')

    results = []
    for i, model in enumerate(models, 1):  # Check top X by downloads
        try:
            model_id = model['id']  # Extract the ID first
            config_url = f'https://huggingface.co/{model_id}/raw/main/config.json'
            config_resp = requests.get(config_url, timeout=5)
            
            if config_resp.status_code == 200 and 'imagebind_huge' in config_resp.text.lower():
                config = config_resp.json()
                
                # Extract vision tower
                vision_tower = None
                for key in ['mm_vision_tower', 'vision_tower']:
                    if key in config and 'imagebind_huge' in str(config[key]).lower():
                        vision_tower = config[key]
                        break
                
                # Check nested configs
                if not vision_tower:
                    for nested in ['vision_tower_cfg', 'vision_config', 'multimodal_config']:
                        if nested in config and isinstance(config[nested], dict):
                            for key in ['mm_vision_tower', 'vision_tower']:
                                if key in config[nested] and 'imagebind_huge' in str(config[nested][key]).lower():
                                    vision_tower = config[nested][key]
                                    break
                            if vision_tower:
                                break
                
                if vision_tower:
                    downloads = model.get('downloads', 0) or 0
                    likes = model.get('likes', 0) or 0
                    score = downloads + (likes * 10)
                    
                    results.append((model_id, downloads, likes, score, vision_tower))
                    print(f'✅ {model_id} - Downloads: {downloads:,} - Vision: {vision_tower}')
            
            if i % 10 == 0:
                print(f'   ... checked {i}/{len(models)} models')
                
        except Exception as e:
            print(f'❌ Error with {model.get("id", "unknown")}: {e}')
            continue
        
        time.sleep(0.05)

    # Sort and show top results
    if results:
        results.sort(key=lambda x: x[3], reverse=True)
        print(f'\n🏆 TOP {min(10, len(results))} IMAGEBIND MODELS BY POPULARITY:')
        print('=' * 70)
        
        for i, (model_id, downloads, likes, score, vision_tower) in enumerate(results[:10], 1):
            print(f'{i}. {model_id}')
            print(f'   📊 {downloads:,} downloads, ❤️ {likes} likes (Score: {score:,.0f})')
            print(f'   👁️  {vision_tower}')
            print(f'   🔗 https://huggingface.co/{model_id}')
            print()
    else:
        print('❌ No models found with imagebind_huge in vision tower config')

if __name__ == "__main__":
    main()