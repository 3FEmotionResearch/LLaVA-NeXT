'''
Basic Usage:
python -m playground.demo.emotion_detect_infer_my --dataset_path_mercaptionplus=/home/paperspace/Downloads/affectgpt-dataset-mini100
'''

# S: Warning suppression - must be at the very beginning
import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
# E: Warning suppression

import argparse
import base64
import json
import math
import time

import cv2
import numpy as np
import openai
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_anyres_image, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from scripts.video.data_loader import create_enhanced_merged_dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # Dataset loading flags.
    parser.add_argument("--dataset_path_mercaptionplus", type=str, required=True, help="Path to the MERCaptionPlus dataset folder")
    # The default of `output_dir` is calculated based on the model path, conv_mode, for_get_frames_num, mm_spatial_pool_stride, and overwrite. See "__main__".
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default=None)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default="predictions")
    # HuggingFace model path.
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == "true"), default=True)
    # Whether to force sample frames.
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == "true"), default=False)
    # Number of frames to get. Only used when force_sample is True.
    parser.add_argument("--for_get_frames_num", type=int, default=8)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument(
        "--prompt_question",
        type=str,
        default="""Analyze the emotion in the video, only respond with English emotion words seperated by commas as a list. For example, "happy, sad, angry".
""",
    )
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    """
    mm_newline_position
    Available Options:
        "grid" (default in training) - Grid-wise processing
        Adds newline tokens at the grid level
        Uses add_token_per_grid() method
        Processes image patches in a grid format and adds newline tokens to separate grid regions
        "frame" - Frame-wise processing
        Adds newline tokens at the frame level for video
        Uses add_token_per_frame() method
        Useful for video understanding where you want to separate different temporal frames
        "one_token" - Single newline token
        Adds just one newline token at the end
        Simplest approach for basic image/video processing
        "no_token" (default in demo scripts) - No newline tokens
        No special newline tokens are added
        Raw image/video features are flattened without structural markers
        Purpose:
        The newline tokens serve as structural separators that help the language model understand:
        Where one image patch ends and another begins
        How visual content is organized spatially or temporally
        The boundaries between different visual elements
    """
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--add_time_instruction", type=str, default=False)

    return parser.parse_args()


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def run_inference(args):
    """
    Run inference on emotion detection dataset using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Load the dataset
    print("Loading dataset...")
    enhanced_merged_by_name = create_enhanced_merged_dataset(args.dataset_path_mercaptionplus)
    
    if not enhanced_merged_by_name:
        print("Error: Could not load dataset or dataset is empty")
        return
    
    print(f"Loaded dataset with {len(enhanced_merged_by_name)} entries")

    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass

    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample

    # import pdb;pdb.set_trace()

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file_path = os.path.join(args.output_dir, f"{output_name}.json")
    error_log_path = os.path.join(args.output_dir, f"{output_name}_errors.log")
    
    # Clear the output file to ensure clean start
    with open(answers_file_path, "w", encoding="utf-8") as f:
        f.write("{\n}")  # Start with empty JSON object
    
    # Initialize counters
    processed_count = 0
    total_entries = len(enhanced_merged_by_name)
    error_count = 0
    
    # Process each entry in the dataset
    for entry_key, entry_val in tqdm(enhanced_merged_by_name.items(), desc="Processing videos"):
        try:
            # Start timing
            start_time = time.time()
            
            # Construct video path
            video_path = os.path.join(args.dataset_path_mercaptionplus, "video", f"{entry_key}.mp4")
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            question = args.prompt_question
            
            # Load and process video
            if "gpt4v" != args.model_path:
                """
                Data Type: torch.Tensor
                Shape: (num_frames, channels, height, width) - typically (4, 3, 336, 336)
                Precision: torch.float16 (half precision)
                Location: GPU memory (CUDA)

                The preprocessing pipeline:
                image_processor.preprocess() normalizes and converts to PyTorch format
                return_tensors="pt" ensures PyTorch tensors
                ["pixel_values"] extracts the tensor from the returned dictionary
                .half() converts to float16 for efficiency
                .cuda() moves to GPU

                After list wrapping, it adds the dimension of batch size.
                """
                video, frame_time, video_time = load_video(video_path, args)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
            else:
                spare_frames, frame_time, video_time = load_video_base64(video_path)
                interval = int(len(video) / args.for_get_frames_num)

            # Run inference on the video
            if "gpt4v" != args.model_path:
                qs = question
                if args.add_time_instruction:
                    time_instruciton = (
                        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    )
                    qs = f"{time_instruciton}\n{qs}"
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    question_text = qs  # Save original question
                    qs = DEFAULT_IMAGE_TOKEN + "\n"
                    if entry_val["chinese_subtitle"] is not None:
                        qs += '<chinese_subtitle>\n' + entry_val["chinese_subtitle"] + '\n</chinese_subtitle>\n'
                    if entry_val["english_subtitle"] is not None:
                        qs += '<english_subtitle>\n' + entry_val["english_subtitle"] + '\n</english_subtitle>\n'
                    qs += question_text

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                if isinstance(input_ids, list):
                    input_ids = torch.tensor(input_ids)
                input_ids = input_ids.unsqueeze(0).cuda()
                if tokenizer.pad_token_id is None:
                    if "qwen" in tokenizer.name_or_path.lower():
                        print("Setting pad token to bos token for qwen model.")
                        tokenizer.pad_token_id = 151643

                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                cur_prompt = question
            else:
                prompt = question

            system_error = ""

            if "gpt4v" != args.model_path:
                with torch.inference_mode():
                    if "mistral" not in cfg_pretrained._name_or_path.lower():
                        output_ids = model.generate(
                            inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.7, max_new_tokens=1024, top_p=0.95, num_beams=1, use_cache=True, stopping_criteria=[stopping_criteria]
                        )
                    else:
                        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
            else:
                openai.api_key = args.api_key  # Your API key here

                max_num_retries = 0
                retry = 5
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
                            *map(lambda x: {"image": x, "resize": 336}, video[0::interval]),
                        ],
                    },
                ]
                params = {
                    "model": "gpt-4-vision-preview",  # gpt-4-1106-vision-preview
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1024,
                }
                sucess_flag = False
                while max_num_retries < retry:
                    try:
                        result = openai.ChatCompletion.create(**params)
                        outputs = result.choices[0].message.content
                        sucess_flag = True
                        break
                    except Exception as inst:
                        if "error" in dir(inst):
                            # import pdb;pdb.set_trace()
                            if inst.error.code == "rate_limit_exceeded":
                                if "TPM" in inst.error.message:
                                    time.sleep(30)
                                    continue
                                else:
                                    import pdb

                                    pdb.set_trace()
                            elif inst.error.code == "insufficient_quota":
                                print(f"insufficient_quota key")
                                exit()
                            elif inst.error.code == "content_policy_violation":
                                print(f"content_policy_violation")
                                system_error = "content_policy_violation"

                                break
                            print("Find error message in response: ", str(inst.error.message), "error code: ", str(inst.error.code))

                        continue
                if not sucess_flag:
                    print(f"Calling OpenAI failed after retrying for {max_num_retries} times. Check the logs for details.")
                    continue  # Skip this entry instead of exiting

            if "gpt4v" != args.model_path:
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            else:
                print(len(video[0::interval]))

            print(f"Entry: {entry_key}")
            print(f"Prompt: {prompt}\n")
            print(f"Response: {outputs}\n")

            if "gpt4v" == args.model_path:
                if system_error == "content_policy_violation":
                    continue
                elif system_error == "":
                    continue
                else:
                    import pdb

                    pdb.set_trace()

            # import pdb;pdb.set_trace()
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]

            outputs = outputs.strip()

            # Calculate inference time
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Write result immediately to file
            result_entry = {
                "prompt_question": prompt,
                "emotion_labels": outputs,
                "inference_time": f"{inference_time:.1f}s"
            }
            
            # Append to JSON file immediately while keeping it valid
            with open(answers_file_path, "r+", encoding="utf-8") as f:
                # Remove the last closing brace
                f.seek(0, 2)  # Go to end of file
                f.seek(f.tell() - 1)  # Go back 1 character
                f.truncate()  # Remove the '}'
                
                if processed_count > 0:
                    f.write(",\n")
                else:
                    f.write("\n")  # First entry, just add newline
                
                f.write(f'  "{entry_key}": {json.dumps(result_entry, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + "  ")}')
                f.write("\n}")  # Always close the JSON properly
                f.flush()  # Ensure data is written immediately
            
            processed_count += 1
            
        except Exception as e:
            error_msg = f"Error processing entry {entry_key}: {str(e)}"
            print(error_msg)
            
            # Create error log file on first error
            if error_count == 0:
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"Error log for {output_name} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
            
            # Log error to file
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
                f.write(f"Entry: {entry_key}\n")
                f.write(f"Video path: {video_path}\n")
                f.write(f"Error details: {repr(e)}\n")
                f.write("-" * 30 + "\n\n")
                f.flush()
            
            error_count += 1
            continue  # Skip this entry and continue with the next one

    print(f"Processed {processed_count} entries successfully out of {total_entries} total entries")
    print(f"Results saved to: {answers_file_path}")
    if error_count > 0:
        print(f"Error log saved to: {error_log_path} ({error_count} errors)")
    else:
        print("No errors encountered during processing")


if __name__ == "__main__":
    args = parse_args()

    # Calculate default output directory if not provided
    if args.output_dir is None:
        ckpt_basename = os.path.basename(args.model_path)
        conv_mode = args.conv_mode if args.conv_mode else "None"
        frames = args.for_get_frames_num
        pool_stride = args.mm_spatial_pool_stride
        overwrite = args.overwrite

        if overwrite == False:
            save_dir = f"{ckpt_basename}_{conv_mode}_frames_{frames}_stride_{pool_stride}_overwrite_{overwrite}"
        else:
            save_dir = f"{ckpt_basename}_{conv_mode}_frames_{frames}_stride_{pool_stride}"

        args.output_dir = f"./work_dirs/emotion_detect_infer_outputs/{save_dir}"

    run_inference(args)
