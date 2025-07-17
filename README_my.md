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
* Test the demo
```bash
bash scripts/video/demo/video_demo.sh lmms-lab/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 average grid True /home/paperspace/Downloads/25-07-11-affectgpt-dataset-mini100/video/sample_00000007.mp4
```

# Our scripts
## Run emotion detection inference by using mercaptionplus dataset's video and Chinese, English subtitles. (Audio modality is TBD)
At this repo's root path. Extract mercaptionplus dataset to some path. The cmd below will generate predictions.json at `./work_dirs`.
```
python -m playground.demo.emotion_detect_infer_my --dataset_path_mercaptionplus=/home/paperspace/Downloads/affectgpt-dataset-mini100
```
We can run evaluation metrics based on AffectGPT emotion wheel:
```
python -m scripts.video.eval.mercaptionplus_metrics_cal_my \
  --dataset_path /home/paperspace/Downloads/affectgpt-dataset-mini100 \
  --predictions_path /home/paperspace/ReposPublic/25-07-14-LLaVA-NeXT/work_dirs/emotion_detect_infer_outputs/LLaVA-NeXT-Video-7B-DPO_vicuna_v1_frames_8_stride_2/predictions.json
```

# Multimodal Setup TBD
* Install imagebind
```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install .
```