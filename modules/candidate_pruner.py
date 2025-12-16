import os
import sys
import json
import time
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

"""
STELLA: From Searching to Structuring (CVPR 2026 Submission)
File: candidate_pruner.py

Description:
    Implementation of the Similarity Scoring for Candidate Pruning (Stage 2.3, Step 1).
    This script calculates the similarity between each answer option (a0-a4) and 
    video clips.
    
    These scores are used to calculate the 'Option-based Inverse Filtering' score
    to remove irrelevant clips (outliers) before the Fine-grained Narrative generation.
"""

# --- 1. Environment & Logging Setup ---

def setup_environment():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(f"[*] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

def setup_logging(gpu_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f'pruning_scoring_gpu_{gpu_index}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"[*] Logging started: {log_filename}")

# --- 2. Model Loading (Same as Anchor Scorer) ---

try:
    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
    from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
except ImportError as e:
    print("\n[Critical Error] VLM2Vec modules not found.")
    exit()

def load_model():
    logging.info("🧠 Loading VLM2Vec-2B Model...")
    model_args = ModelArguments(
        model_name='Qwen/Qwen2-VL-2B-Instruct',
        checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = load_processor(model_args, data_args)
        model = MMEBModel.load(model_args)
        model = model.to(device, dtype=torch.bfloat16)
        model.eval()
        logging.info(f"✅ Model loaded on {device}")
        return model, processor, device
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}", exc_info=True)
        exit()

# --- 3. Similarity Calculation ---

@torch.inference_mode()
def calculate_similarity(model, processor, device, clip_path, query_text):
    """Calculates cosine similarity between a video clip and a text query."""
    video_messages = [
        {"role": "user", "content": [
            {"type": "video", "video": str(clip_path), "fps": 1.0},
            {"type": "text", "text": "Represent the given video."},
        ]}
    ]
    _, video_inputs = process_vision_info(video_messages)
    video_inputs_processed = processor(
        text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
        videos=video_inputs, return_tensors="pt"
    )
    video_inputs_processed = {k: v.to(device) for k, v in video_inputs_processed.items()}
    video_inputs_processed['pixel_values_videos'] = video_inputs_processed['pixel_values_videos'].unsqueeze(0)
    video_inputs_processed['video_grid_thw'] = video_inputs_processed['video_grid_thw'].unsqueeze(0)
    
    video_output = model(qry=video_inputs_processed)["qry_reps"]

    text_inputs = processor(text=query_text, images=None, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_output = model(tgt=text_inputs)["tgt_reps"]
    
    similarity_score = model.compute_similarity(video_output, text_output).item()
    return similarity_score

# --- 4. Main Processing Logic (Options Processing) ---

def process_tasks(model, processor, device, tasks, output_file, clips_base_dir):
    """Processes tasks by calculating similarity for all Answer Options (a0-a4)."""
    
    results = []
    results_map = {}
    
    # Resume Logic
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            logging.info(f"🔄 Resuming: Loaded {len(results)} existing records.")
            for item in results:
                # Check for 'answer_option_status' to distinguish from anchor scoring
                if item.get('answer_option_status') == 'success':
                    results_map[item['problem_id']] = item
        except json.JSONDecodeError:
            logging.warning("⚠️ Existing file corrupted. Starting fresh.")

    tasks_to_run = [t for t in tasks if t['problem_id'] not in results_map]
    
    if not tasks_to_run:
        logging.info("🎉 All assigned tasks are already completed.")
        return

    logging.info(f"🚀 Processing {len(tasks_to_run)} items for Option Pruning...")

    pbar = tqdm(tasks_to_run, desc=f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}", unit="item")

    for i, item in enumerate(pbar):
        video_id = item.get('video')
        pid = item['problem_id']
        
        status = 'fail'
        answer_option_scores = {} # Dictionary to store scores for each option
        
        for attempt in range(1, 4):
            try:
                clips_dir = Path(clips_base_dir) / f"{video_id}_clips"
                if not clips_dir.is_dir():
                    raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

                clip_paths = sorted(list(clips_dir.glob(f"*{video_id}*clip_*.mp4")))
                if not clip_paths:
                    raise FileNotFoundError(f"No clips found in {clips_dir}")

                # --- Iterate through Options a0 to a4 ---
                temp_all_scores = {}
                for j in range(5):
                    option_key = f'a{j}'
                    # Flexible key access (handle case sensitivity if needed)
                    option_text = item.get(option_key)
                    
                    if not option_text:
                        continue # Skip empty options

                    clip_scores = []
                    for clip_path in clip_paths:
                        score = calculate_similarity(model, processor, device, clip_path, option_text)
                        clip_scores.append(score)
                    
                    temp_all_scores[option_key] = {
                        "text": option_text,
                        "scores": clip_scores
                    }
                # ----------------------------------------

                answer_option_scores = temp_all_scores
                status = 'success'
                break

            except Exception as e:
                if attempt == 3:
                    logging.error(f"❌ Failed PID {pid}: {e}")
                time.sleep(1)
        
        # Save results
        item['answer_option_scores'] = answer_option_scores
        item['answer_option_status'] = status
        results_map[pid] = item

        if (i + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(list(results_map.values()), f, indent=4)
    
    with open(output_file, 'w') as f:
        json.dump(list(results_map.values()), f, indent=4)
    logging.info(f"✅ Processing Complete. Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="STELLA: Candidate Pruner (Option Scorer)")
    parser.add_argument("--gpu_index", type=int, required=True, help="GPU Index")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output_dir", type=str, default="./results_pruning", help="Output directory")
    parser.add_argument("--clips_dir", type=str, required=True, help="Base directory of clips")
    
    args = parser.parse_args()

    setup_environment()
    setup_logging(args.gpu_index, args.output_dir)

    model, processor, device = load_model()

    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        return
        
    with open(args.input_file, 'r') as f:
        all_data = json.load(f)
    
    assigned_tasks = [
        item for i, item in enumerate(all_data) 
        if i % args.total_gpus == args.gpu_index
    ]
    logging.info(f"assigned {len(assigned_tasks)} tasks to GPU {args.gpu_index}")

    output_json_path = os.path.join(args.output_dir, f"option_scores_gpu_{args.gpu_index}.json")
    
    process_tasks(model, processor, device, assigned_tasks, output_json_path, args.clips_dir)

if __name__ == "__main__":
    main()