# Env Setup

Remove torch version requirement in /home/paperspace/ReposPublic/25-07-14-ImageBind/requirements.txt .

```
576  pip uninstall -y torch torchvision torchaudio transformers accelerate flash-attn
577  pip cache purge
578  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
579  pip install -e ".[train]"
580  pip uninstall -y torch torchvision torchaudio
581  # Make sure you are in the LLaVA-NeXT repository folder
582  pip install --index-url https://download.pytorch.org/whl/cu121 -e ".[train]"
583  pip install torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
584  pip install torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
585  pip install flash-attn --no-build-isolation
586  pip install flash-attn==2.5.7
```
# Download encoder pretrain dataset.
```bash
python scripts/train/download_llava_pretrain.py \
>     --output_dir /home/paperspace/Downloads/blip_558k \
>     --download_images \
>     --use_zip \
>     --sample_size 100
```

# Run encoder embedding space alignment pretrain.
```
bash scripts/train/pretrain_imagebind_huge.sh
```

# Notes
## Files
* .checkpoints/ (with dot) - ImageBind Model Weights. 
* checkpoints/ (without dot) - LLaVA Training Progress.