# Env Steps
* Run
```bash
# Step 1: Install torch first
pip install torch==2.1.2 torchvision==0.16.2 numpy==1.26.4

# Step 2: Install flash-attn (now that torch is available)
pip install flash-attn==2.5.7

# Step 3: Install the rest
pip install -r requirements_my.txt
```
* Follow "#### 2. **Install the inference package:**" to install currnet repo as a pkg.
* Run the demo
```bash
bash scripts/video/demo/video_demo.sh lmms-lab/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average grid True /home/paperspace/Downloads/25-07-11-affectgpt-dataset-mini100/video/sample_00000007.mp4
```

# Batch Inference on AffectGPT Dataset
* Put the affectgpt dataset in the path:
```bash
/home/paperspace/LLaVA-NeXT/scripts/video/affectgpt-dataset-mini100
```
* The dataset should contain the following files:
  - CSV files: `subtitle_chieng.csv`, `track2_train_mercaptionplus.csv`, `track2_train_ovmerd.csv`, `track3_train_mercaptionplus.csv`, `track3_train_ovmerd.csv`
  - Video files in: `video/` folder with `.mp4` files

* Run batch inference on all 50 samples:
```bash
cd /home/paperspace/LLaVA-NeXT/scripts/video
python affectgptdata.py
```

* This will:
  - Load and merge all CSV data files
  - Save ground truth labels to: `work_dirs/batch_inference_results/ground_truth_labels.json`
  - Run inference on all available video samples using the same method as `video_demo.sh`
  - Save predictions to: `work_dirs/batch_inference_results/inference_results/predictions.json`
  - Save raw outputs to: `work_dirs/batch_inference_results/inference_results/raw_outputs.json`
  - Save failed samples (if any) to: `work_dirs/batch_inference_results/inference_results/failed_samples.json`

# Multimodal Demo
* Install imagebind
```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
```