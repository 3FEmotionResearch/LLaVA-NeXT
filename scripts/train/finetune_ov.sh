export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9" 
PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_ov_stage_july21.yaml \
    --image_folder /mnt/bn/vl-research/data/llava_data \
    --video_folder /mnt/bn/vl-research/data/llava_video \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /mnt/bn/vl-research/checkpoints/onevision/$RUN_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 8 \
    2>&1 | tee logs/training_log_$(date +%Y%m%d_%H%M%S).log

# Command to monitor loss in real-time (run in another terminal):
# tail -f training_log_*.log | grep -E "(loss|epoch|step)"

exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
