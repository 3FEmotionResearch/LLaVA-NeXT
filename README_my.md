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
# Multimodal Demo
* Install imagebind
```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
```