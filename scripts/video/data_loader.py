import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files


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
    """

    # Required CSV files
    required_files = ["subtitle_chieng.csv", "track2_train_mercaptionplus.csv", "track2_train_ovmerd.csv", "track3_train_mercaptionplus.csv", "track3_train_ovmerd.csv"]

    # Check if we should use local files
    use_local = False
    if local_path and os.path.exists(local_path):
        # Check if all required files exist in the local path
        local_files_exist = all(os.path.exists(os.path.join(local_path, file)) for file in required_files)
        if local_files_exist:
            use_local = True
            csv_files = required_files
            print(f"✓ Using local files from: {local_path}")
        else:
            missing_files = [file for file in required_files if not os.path.exists(os.path.join(local_path, file))]
            print(f"❌ Missing files in {local_path}: {missing_files}")
            print("Falling back to Hugging Face...")

    # If not using local files, discover files using Hugging Face Hub API
    if not use_local:
        try:
            # List all files in the dataset repository
            files = list_repo_files("MERChallenge/MER2025", repo_type="dataset")

            # Filter for CSV files
            csv_files = [f for f in files if f.endswith(".csv")]

        except Exception as e:
            # Fallback to the required files list
            csv_files = required_files

    datasets = {}
    for csv_file in csv_files:
        try:
            dataset_name = csv_file.replace(".csv", "")

            if use_local:
                # Load from local file
                file_path = os.path.join(local_path, csv_file)
                df = pd.read_csv(file_path, encoding="utf-8")

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
                    file_path = hf_hub_download(repo_id="MERChallenge/MER2025", filename=csv_file, repo_type="dataset")

                    # Load with pandas
                    df = pd.read_csv(file_path, encoding="utf-8")

                    # Convert to datasets format
                    from datasets import Dataset

                    dataset_dict = {"train": Dataset.from_pandas(df)}
                    datasets[dataset_name] = dataset_dict

                else:
                    # Load without specifying split to see all available splits
                    dataset_dict = load_dataset("MERChallenge/MER2025", data_files=csv_file)
                    datasets[dataset_name] = dataset_dict

        except Exception as e:
            pass  # Silently skip failed loads

    merged_by_name = {}

    # Check if we have both datasets
    if "track2_train_mercaptionplus" in datasets and "track3_train_mercaptionplus" in datasets:

        # Get the train splits from both datasets
        track2_data = datasets["track2_train_mercaptionplus"]["train"]
        track3_data = datasets["track3_train_mercaptionplus"]["train"]

        # Convert to pandas DataFrames for easier merging
        track2_df = track2_data.to_pandas()
        track3_df = track3_data.to_pandas()

        # Check if both have 'name' column
        if "name" in track2_df.columns and "name" in track3_df.columns:
            # Merge on 'name' column
            merged_df = pd.merge(track2_df, track3_df, on="name", how="inner", suffixes=("_track2", "_track3"))  # Only keep rows that exist in both datasets

            # Create a list of dictionaries with combined data
            merged_data = []
            for _, row in merged_df.iterrows():
                combined_row = {"name": row["name"], "openset": row.get("openset", None), "reason": row.get("reason", None)}  # from track2  # from track3

                # Add all other columns from both datasets
                for col in merged_df.columns:
                    if col not in ["name", "openset", "reason"]:
                        combined_row[col] = row[col]

                merged_data.append(combined_row)

            # Create a dictionary indexed by name for easy access
            merged_by_name = {}
            for entry in merged_data:
                name = entry["name"]
                merged_by_name[name] = entry

    subtitle_by_name = {}
    enhanced_merged_by_name = {}

    if "subtitle_chieng" in datasets:
        subtitle_data = datasets["subtitle_chieng"]["train"]

        # Convert to pandas DataFrame
        subtitle_df = subtitle_data.to_pandas()

        # Create subtitle dictionary indexed by name
        subtitle_by_name = {}
        for _, row in subtitle_df.iterrows():
            name = row["name"]
            subtitle_by_name[name] = {"name": name, "chinese": row["chinese"] if pd.notna(row["chinese"]) else None, "english": row["english"] if pd.notna(row["english"]) else None}

        # Try to merge subtitle data with existing merged data if available
        if merged_by_name:
            # Add subtitle data to existing merged data
            enhanced_merged_by_name = {}
            for name, data in merged_by_name.items():
                enhanced_data = data.copy()

                # Add subtitle information if available
                if name in subtitle_by_name:
                    subtitle_info = subtitle_by_name[name]
                    enhanced_data.update({"chinese_subtitle": subtitle_info["chinese"], "english_subtitle": subtitle_info["english"]})
                else:
                    enhanced_data.update({"chinese_subtitle": None, "english_subtitle": None})

                enhanced_merged_by_name[name] = enhanced_data

            # Show sample of enhanced data
            print(f"Sample of enhanced data with subtitles:")
            sample_names_with_subtitles = [name for name in list(enhanced_merged_by_name.keys())[:5] if enhanced_merged_by_name[name]["chinese_subtitle"] or enhanced_merged_by_name[name]["english_subtitle"]]

            for name in sample_names_with_subtitles[:1]:
                data = enhanced_merged_by_name[name]
                print(f"  Name: {name}")
                print(f"    openset: {data.get('openset', 'N/A')}")
                print(f"    reason: {data.get('reason', 'N/A')[:100]}..." if data.get("reason") and len(data.get("reason", "")) > 100 else f"    reason: {data.get('reason', 'N/A')}")
                print(f"    chinese_subtitle: {data.get('chinese_subtitle', 'N/A')}")
                print(f"    english_subtitle: {data.get('english_subtitle', 'N/A')}")
                print()

    else:
        # If no subtitle data, use merged data as enhanced data
        if merged_by_name:
            enhanced_merged_by_name = merged_by_name.copy()

    # Return all the created data structures
    print(f"Number of samples in enhanced_merged_by_name: {len(enhanced_merged_by_name)}")
    return enhanced_merged_by_name
