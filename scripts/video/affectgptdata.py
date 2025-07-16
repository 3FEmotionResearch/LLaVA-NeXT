from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd
import os
import json
import subprocess
from tqdm import tqdm
import glob

def create_enhanced_merged_dataset(local_path=None):
    """
    Create enhanced merged dataset with emotions, reasoning, and subtitle data.
    
    Args:
        local_path (str, optional): Path to local folder containing CSV files.
                                   If provided and all required files exist, 
                                   will read from local files instead of Hugging Face.
    
    Returns:
        dict: Dictionary containing:
            - enhanced_merged_by_name: Complete dataset with all merged data
            - merged_by_name: Basic merged dataset (track2 + track3)
            - subtitle_by_name: Subtitle data indexed by name
            - datasets: All loaded individual datasets
    """
    
    # Required CSV files
    required_files = [
        "subtitle_chieng.csv",
        "track2_train_mercaptionplus.csv", 
        "track2_train_ovmerd.csv",
        "track3_train_mercaptionplus.csv",
        "track3_train_ovmerd.csv"
    ]
    
    # Check if we should use local files
    use_local = False
    if local_path and os.path.exists(local_path):
        # Check if all required files exist in the local path
        local_files_exist = all(
            os.path.exists(os.path.join(local_path, file)) 
            for file in required_files
        )
        if local_files_exist:
            use_local = True
            csv_files = required_files
            print(f"✓ Using local files from: {local_path}")
        else:
            missing_files = [
                file for file in required_files 
                if not os.path.exists(os.path.join(local_path, file))
            ]
            print(f"❌ Missing files in {local_path}: {missing_files}")
            print("Falling back to Hugging Face...")
    
    # If not using local files, discover files using Hugging Face Hub API
    if not use_local:
        try:
            # List all files in the dataset repository
            files = list_repo_files("MERChallenge/MER2025", repo_type="dataset")
            
            # Filter for CSV files
            csv_files = [f for f in files if f.endswith('.csv')]
            
        except Exception as e:
            # Fallback to the required files list
            csv_files = required_files

    datasets = {}
    for csv_file in csv_files:
        try:
            dataset_name = csv_file.replace('.csv', '')
            
            if use_local:
                # Load from local file
                file_path = os.path.join(local_path, csv_file)
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Convert to datasets format
                from datasets import Dataset
                dataset_dict = {"train": Dataset.from_pandas(df)}
                datasets[dataset_name] = dataset_dict
                
            else:
                # Load from Hugging Face
                if csv_file == "subtitle_chieng.csv":
                    # Special handling for subtitle_chieng.csv which has encoding issues with datasets library
                    from huggingface_hub import hf_hub_download
                    
                    # Download the file first
                    file_path = hf_hub_download(
                        repo_id="MERChallenge/MER2025",
                        filename=csv_file,
                        repo_type="dataset"
                    )
                    
                    # Load with pandas
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Convert to datasets format
                    from datasets import Dataset
                    dataset_dict = {"train": Dataset.from_pandas(df)}
                    datasets[dataset_name] = dataset_dict
                    
                else:
                    # Load without specifying split to see all available splits
                    dataset_dict = load_dataset(
                        "MERChallenge/MER2025", 
                        data_files=csv_file
                    )
                    datasets[dataset_name] = dataset_dict
                
        except Exception as e:
            pass  # Silently skip failed loads

    merged_by_name = {}
    
    # Check if we have both datasets
    if ('track2_train_mercaptionplus' in datasets and 
        'track3_train_mercaptionplus' in datasets):
        
        # Get the train splits from both datasets
        track2_data = datasets['track2_train_mercaptionplus']['train']
        track3_data = datasets['track3_train_mercaptionplus']['train']
        
        # Convert to pandas DataFrames for easier merging
        track2_df = track2_data.to_pandas()
        track3_df = track3_data.to_pandas()
        
        # Check if both have 'name' column
        if 'name' in track2_df.columns and 'name' in track3_df.columns:
            # Merge on 'name' column
            merged_df = pd.merge(
                track2_df, 
                track3_df, 
                on='name', 
                how='inner',  # Only keep rows that exist in both datasets
                suffixes=('_track2', '_track3')
            )
            
            # Create a list of dictionaries with combined data
            merged_data = []
            for _, row in merged_df.iterrows():
                combined_row = {
                    'name': row['name'],
                    'openset': row.get('openset', None),  # from track2
                    'reason': row.get('reason', None)     # from track3
                }
                
                # Add all other columns from both datasets
                for col in merged_df.columns:
                    if col not in ['name', 'openset', 'reason']:
                        combined_row[col] = row[col]
                
                merged_data.append(combined_row)
            
            # Create a dictionary indexed by name for easy access
            merged_by_name = {}
            for entry in merged_data:
                name = entry['name']
                merged_by_name[name] = entry

    subtitle_by_name = {}
    enhanced_merged_by_name = {}
    
    if 'subtitle_chieng' in datasets:
        subtitle_data = datasets['subtitle_chieng']['train']
        
        # Convert to pandas DataFrame
        subtitle_df = subtitle_data.to_pandas()
        
        # Create subtitle dictionary indexed by name
        subtitle_by_name = {}
        for _, row in subtitle_df.iterrows():
            name = row['name']
            subtitle_by_name[name] = {
                'name': name,
                'chinese': row['chinese'] if pd.notna(row['chinese']) else None,
                'english': row['english'] if pd.notna(row['english']) else None
            }
        
        # Try to merge subtitle data with existing merged data if available
        if merged_by_name:
            # Add subtitle data to existing merged data
            enhanced_merged_by_name = {}
            for name, data in merged_by_name.items():
                enhanced_data = data.copy()
                
                # Add subtitle information if available
                if name in subtitle_by_name:
                    subtitle_info = subtitle_by_name[name]
                    enhanced_data.update({
                        'chinese_subtitle': subtitle_info['chinese'],
                        'english_subtitle': subtitle_info['english']
                    })
                else:
                    enhanced_data.update({
                        'chinese_subtitle': None,
                        'english_subtitle': None
                    })
                
                enhanced_merged_by_name[name] = enhanced_data
            
            # Show sample of enhanced data
            print(f"Sample of enhanced data with subtitles:")
            sample_names_with_subtitles = [name for name in list(enhanced_merged_by_name.keys())[:5] 
                                         if enhanced_merged_by_name[name]['chinese_subtitle'] or enhanced_merged_by_name[name]['english_subtitle']]
            
            for name in sample_names_with_subtitles[:2]:
                data = enhanced_merged_by_name[name]
                print(f"  Name: {name}")
                print(f"    openset: {data.get('openset', 'N/A')}")
                print(f"    reason: {data.get('reason', 'N/A')[:100]}..." if data.get('reason') and len(data.get('reason', '')) > 100 else f"    reason: {data.get('reason', 'N/A')}")
                print(f"    chinese_subtitle: {data.get('chinese_subtitle', 'N/A')}")
                print(f"    english_subtitle: {data.get('english_subtitle', 'N/A')}")
                print()
            
    else:
        # If no subtitle data, use merged data as enhanced data
        if merged_by_name:
            enhanced_merged_by_name = merged_by_name.copy()
    
    # Return all the created data structures
    return {
        'enhanced_merged_by_name': enhanced_merged_by_name,
        'merged_by_name': merged_by_name,
        'subtitle_by_name': subtitle_by_name,
        'datasets': datasets
    }

def load_from_local(local_path):
    """
    Convenience function to load dataset from local CSV files.
    
    Args:
        local_path (str): Path to folder containing the required CSV files:
                         - subtitle_chieng.csv
                         - track2_train_mercaptionplus.csv
                         - track2_train_ovmerd.csv
                         - track3_train_mercaptionplus.csv
                         - track3_train_ovmerd.csv
    
    Returns:
        dict: Same as create_enhanced_merged_dataset()
    """
    return create_enhanced_merged_dataset(local_path=local_path)

def save_ground_truth_labels(enhanced_merged_by_name, output_file):
    """
    Save ground truth labels to a JSON file.
    
    Args:
        enhanced_merged_by_name (dict): Enhanced merged dataset
        output_file (str): Path to save ground truth labels
    """
    ground_truth = {}
    
    for name, data in enhanced_merged_by_name.items():
        ground_truth[name] = {
            'video_name': name,
            'openset': data.get('openset', None),  # emotion label
            'reason': data.get('reason', None),    # emotion reasoning
            'chinese_subtitle': data.get('chinese_subtitle', None),
            'english_subtitle': data.get('english_subtitle', None)
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Ground truth labels saved to: {output_file}")
    print(f"   Total samples: {len(ground_truth)}")

def batch_inference_videos(enhanced_merged_by_name, video_base_path, model_path, output_dir, max_samples=None):
    """
    Batch inference for all video samples using the video_demo.sh script.
    
    Args:
        enhanced_merged_by_name (dict): Enhanced merged dataset
        video_base_path (str): Base path to video files
        model_path (str): Model path for inference
        output_dir (str): Directory to save inference results
        max_samples (int, optional): Maximum number of samples to process. If None, process all.
    """
    print(f"\n🚀 Starting batch inference...")
    print(f"   Model: {model_path}")
    print(f"   Video base path: {video_base_path}")
    print(f"   Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare predictions storage
    predictions = {}
    raw_outputs = {}
    failed_samples = []
    
    # Get all available video files
    video_files = glob.glob(os.path.join(video_base_path, "*.mp4"))
    available_videos = {os.path.basename(f).replace('.mp4', ''): f for f in video_files}
    
    print(f"   Found {len(available_videos)} video files")
    
    # Filter samples to only those with available videos
    samples_to_process = []
    for name in enhanced_merged_by_name.keys():
        if name in available_videos:
            samples_to_process.append(name)
        else:
            print(f"⚠️  Video not found for sample: {name}")
    
    # Limit samples if max_samples is specified
    if max_samples:
        samples_to_process = samples_to_process[:max_samples]
        print(f"   Limited to first {max_samples} samples")
    
    print(f"   Processing {len(samples_to_process)} samples with available videos")
    
    # Process each sample one by one
    for i, sample_name in enumerate(samples_to_process):
        print(f"\n{'='*60}")
        print(f"🎬 Processing {i+1}/{len(samples_to_process)}: {sample_name}")
        print(f"{'='*60}")
        
        try:
            video_path = available_videos[sample_name]
            print(f"📁 Video: {video_path}")
            
            # Show ground truth for reference
            if sample_name in enhanced_merged_by_name:
                gt_data = enhanced_merged_by_name[sample_name]
                print(f"🎯 Ground Truth: {gt_data.get('openset', 'N/A')}")
                if gt_data.get('reason'):
                    print(f"💭 Reason: {gt_data.get('reason', '')[:100]}...")
            
            # Run inference using video_demo.sh
            cmd = [
                "bash", "scripts/video/demo/video_demo.sh",
                model_path,                    # model path
                "vicuna_v1",                  # conv_mode
                "32",                         # frames
                "2",                          # pool_stride
                "average",                    # pool_mode
                "grid",                       # newline_position
                "True",                       # overwrite
                video_path,                    # video_path
                "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.", # prompt
            ]
            
            print(f"🔄 Running inference...")
            start_time = __import__('time').time()
            
            # Run the command
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  cwd="/home/paperspace/ReposPublic/25-07-14-LLaVA-NeXT",
                                  timeout=300)  # 5 minute timeout
            
            end_time = __import__('time').time()
            print(f"⏱️  Inference took {end_time - start_time:.1f} seconds")
            
            # Save raw output
            raw_output_info = {
                'sample_name': sample_name,
                'video_path': video_path,
                'command': ' '.join(cmd),
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'inference_time': end_time - start_time
            }
            raw_outputs[sample_name] = raw_output_info
            
            if result.returncode == 0:
                # Try to extract prediction from the output
                output_lines = result.stdout.split('\n')
                question_line = None
                response_line = None
                
                for line in output_lines:
                    if line.startswith('Question:'):
                        question_line = line.replace('Question:', '').strip()
                    elif line.startswith('Response:'):
                        response_line = line.replace('Response:', '').strip()
                
                if response_line:
                    # Clean up the response
                    cleaned_emotion = clean_emotion_response(response_line)
                    
                    predictions[sample_name] = {
                        'video_name': sample_name,
                        'raw_response': response_line,
                        'cleaned_emotion': cleaned_emotion,
                        'question': question_line,
                        'video_path': video_path,
                        'inference_time': end_time - start_time
                    }
                    
                    print(f"❓ Question: {question_line}")
                    print(f"📝 Raw Response: {response_line}")
                    print(f"✨ Cleaned: {cleaned_emotion}")
                    print(f"✅ SUCCESS")
                else:
                    failed_samples.append({
                        'sample_name': sample_name,
                        'reason': 'Could not extract response from output',
                        'stdout': result.stdout[:500] + '...' if len(result.stdout) > 500 else result.stdout
                    })
                    print(f"❌ Could not extract prediction from output")
                    print(f"📝 First 200 chars of stdout: {result.stdout[:200]}...")
            else:
                failed_samples.append({
                    'sample_name': sample_name,
                    'reason': f'Command failed with return code {result.returncode}',
                    'stderr': result.stderr[:500] + '...' if len(result.stderr) > 500 else result.stderr
                })
                print(f"❌ Inference failed (code {result.returncode})")
                print(f"📝 Error: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            failed_samples.append({
                'sample_name': sample_name,
                'reason': 'Inference timed out after 5 minutes'
            })
            print(f"⏰ TIMEOUT: Inference took too long")
        except Exception as e:
            failed_samples.append({
                'sample_name': sample_name,
                'reason': f'Exception: {str(e)}'
            })
            print(f"❌ Error processing {sample_name}: {str(e)}")
        
        # Save intermediate results after each sample
        predictions_file = os.path.join(output_dir, "predictions.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        raw_outputs_file = os.path.join(output_dir, "raw_outputs.json")
        with open(raw_outputs_file, 'w', encoding='utf-8') as f:
            json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Progress saved: {len(predictions)} successful, {len(failed_samples)} failed")
    
    # Save failed samples
    if failed_samples:
        failed_file = os.path.join(output_dir, "failed_samples.json")
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"⚠️  {len(failed_samples)} failed samples saved to: {failed_file}")
    
    print(f"\n🎯 Batch inference completed!")
    print(f"   Successful predictions: {len(predictions)}")
    print(f"   Failed samples: {len(failed_samples)}")
    print(f"   Predictions saved to: {predictions_file}")
    print(f"   Raw outputs saved to: {raw_outputs_file}")
    
    return predictions, failed_samples

def clean_emotion_response(response):
    """
    Clean the emotion response while preserving emotion phrases.
    
    Args:
        response (str): Raw response from the model
        
    Returns:
        str: Cleaned emotion response (words and phrases)
    """
    # Basic cleaning
    cleaned = response.strip()
    
    # Remove common prefixes that might appear
    prefixes_to_remove = [
        "The emotion is", "I see", "The person appears", "The video shows",
        "Based on the video", "From the video", "The main emotion", 
        "This video expresses", "The emotions are", "I can identify",
        "The dominant emotion", "The facial expression shows", "I would say",
        "In this video", "The person", "Looking at", "From what I can see",
        "The individual", "It appears", "I observe", "The subject",
        "I can see", "The emotion expressed", "The primary emotion"
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            # Remove common connectors after prefix removal
            if cleaned.lower().startswith(('is', 'are', 'seems', 'appears')):
                words = cleaned.split()
                if words:
                    cleaned = ' '.join(words[1:]).strip()
            break
    
    # Remove common suffixes
    suffixes_to_remove = [
        "in the video", "can be seen", "is evident", "is apparent", 
        "are visible", "are present", "is displayed", "is shown",
        "throughout the video", "in this clip"
    ]
    
    for suffix in suffixes_to_remove:
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[:-len(suffix)].strip()
            break
    
    # Remove ending punctuation
    cleaned = cleaned.rstrip('.,!?;:')
    
    # Only do major trimming if response is extremely long (>200 chars)
    if len(cleaned) > 200:
        # Take first sentence or first part before major punctuation
        sentences = cleaned.split('.')
        cleaned = sentences[0].strip()
        
        # If still too long, look for comma-separated patterns
        if len(cleaned) > 200:
            import re
            # Try to find emotion phrases separated by commas
            comma_parts = cleaned.split(',')
            if len(comma_parts) > 1:
                # Take first few parts that seem like emotions
                emotion_parts = []
                for part in comma_parts[:4]:  # Max 4 parts
                    part = part.strip()
                    if len(part) < 50 and part:  # Keep reasonable length parts
                        emotion_parts.append(part)
                if emotion_parts:
                    cleaned = ', '.join(emotion_parts)
    
    # Final cleanup - remove any remaining leading articles
    words = cleaned.split()
    if words and words[0].lower() in ['the', 'a', 'an']:
        cleaned = ' '.join(words[1:])
    
    return cleaned.strip()

def run_limited_inference(local_path, video_base_path, model_path, output_base_dir, max_samples=5):
    """
    Run inference on a limited number of samples for testing.
    
    Args:
        local_path (str): Path to CSV files
        video_base_path (str): Path to video files  
        model_path (str): Model path for inference
        output_base_dir (str): Base directory for all outputs
        max_samples (int): Number of samples to process
    """
    print(f"🧪 Running limited inference on {max_samples} samples...")
    
    # Step 1: Load and merge data
    print("\n📊 Step 1: Loading and merging data...")
    result = create_enhanced_merged_dataset(local_path=local_path)
    enhanced_merged_by_name = result['enhanced_merged_by_name']
    
    print(f"   Loaded {len(enhanced_merged_by_name)} samples")
    
    # Step 2: Save ground truth labels
    print("\n💾 Step 2: Saving ground truth labels...")
    gt_output_file = os.path.join(output_base_dir, "ground_truth_labels.json")
    save_ground_truth_labels(enhanced_merged_by_name, gt_output_file)
    
    # Step 3: Run limited inference
    print(f"\n🤖 Step 3: Running inference on {max_samples} samples...")
    inference_output_dir = os.path.join(output_base_dir, "inference_results")
    predictions, failed_samples = batch_inference_videos(
        enhanced_merged_by_name, 
        video_base_path, 
        model_path, 
        inference_output_dir,
        max_samples=max_samples
    )
    
    print(f"\n🎉 Limited inference completed!")
    print(f"   Ground truth: {gt_output_file}")
    print(f"   Predictions: {os.path.join(inference_output_dir, 'predictions.json')}")
    
    return {
        'enhanced_merged_by_name': enhanced_merged_by_name,
        'predictions': predictions,
        'failed_samples': failed_samples,
        'ground_truth_file': gt_output_file,
        'predictions_file': os.path.join(inference_output_dir, 'predictions.json')
    }

def run_complete_pipeline(local_path, video_base_path, model_path, output_base_dir):
    """
    Run the complete pipeline: data loading, ground truth saving, and batch inference.
    
    Args:
        local_path (str): Path to CSV files
        video_base_path (str): Path to video files
        model_path (str): Model path for inference
        output_base_dir (str): Base directory for all outputs
    """
    print("🔄 Starting complete pipeline...")
    
    # Step 1: Load and merge data
    print("\n📊 Step 1: Loading and merging data...")
    result = create_enhanced_merged_dataset(local_path=local_path)
    enhanced_merged_by_name = result['enhanced_merged_by_name']
    
    print(f"   Loaded {len(enhanced_merged_by_name)} samples")
    
    # Step 2: Save ground truth labels
    print("\n💾 Step 2: Saving ground truth labels...")
    gt_output_file = os.path.join(output_base_dir, "ground_truth_labels.json")
    save_ground_truth_labels(enhanced_merged_by_name, gt_output_file)
    
    # Step 3: Run batch inference
    print("\n🤖 Step 3: Running batch inference...")
    inference_output_dir = os.path.join(output_base_dir, "inference_results")
    predictions, failed_samples = batch_inference_videos(
        enhanced_merged_by_name, 
        video_base_path, 
        model_path, 
        inference_output_dir
    )
    
    print(f"\n🎉 Pipeline completed!")
    print(f"   Ground truth: {gt_output_file}")
    print(f"   Predictions: {os.path.join(inference_output_dir, 'predictions.json')}")
    
    return {
        'enhanced_merged_by_name': enhanced_merged_by_name,
        'predictions': predictions,
        'failed_samples': failed_samples,
        'ground_truth_file': gt_output_file,
        'predictions_file': os.path.join(inference_output_dir, 'predictions.json')
    }

# Main execution
if __name__ == "__main__":
    # Configuration
    LOCAL_CSV_PATH = "/home/paperspace/Downloads/affectgpt-dataset-mini100"
    VIDEO_BASE_PATH = "/home/paperspace/Downloads/affectgpt-dataset-mini100/video"
    MODEL_PATH = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
    OUTPUT_BASE_DIR = "./.work_dirs/batch_inference_results"
    
    # Run complete pipeline for all samples
    print("🚀 Running complete inference pipeline...")
    pipeline_result = run_complete_pipeline(
        local_path=LOCAL_CSV_PATH,
        video_base_path=VIDEO_BASE_PATH,
        model_path=MODEL_PATH,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    # Make results globally accessible
    globals().update({
        'enhanced_merged_by_name': pipeline_result['enhanced_merged_by_name'],
        'predictions': pipeline_result['predictions'],
        'failed_samples': pipeline_result['failed_samples']
    })
    
    print(f"\n📈 Ready for accuracy calculation!")
    print(f"Use: enhanced_merged_by_name, predictions, failed_samples")
