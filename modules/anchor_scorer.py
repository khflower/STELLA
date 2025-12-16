import sys
import os

# Add the current directory to sys.path to allow importing from 'src' in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Import VLM2Vec modules (Assuming 'src' folder is in the same directory)
from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

# --- 1. Environment & Logging Setup ---

def setup_environment():
    """Configures environment variables."""
    # Disable tokenizers parallelism warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup_logging(gpu_index, output_dir):
    """Configures logging settings."""
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f'anchor_scoring_gpu_{gpu_index}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"[*] Logging started: {log_filename}")

# --- 2. Model Loading (VLM2Vec) ---

def load_model():
    """
    Loads the VLM2Vec-2B model and processor.
    """
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
        if device == 'cpu':
            logging.warning("⚠️ CUDA not available. Running on CPU (slow).")
        
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
    """
    Calculates the cosine similarity between a video clip and a text query.
    """
    
    # 1. Video Encoding
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
    
    # Fix dimensions for model input
    video_inputs_processed['pixel_values_videos'] = video_inputs_processed['pixel_values_videos'].unsqueeze(0)
    video_inputs_processed['video_grid_thw'] = video_inputs_processed['video_grid_thw'].unsqueeze(0)
    
    video_output = model(qry=video_inputs_processed)["qry_reps"]

    # 2. Text Encoding
    text_inputs = processor(text=query_text, images=None, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_output = model(tgt=text_inputs)["tgt_reps"]
    
    # 3. Compute Similarity
    similarity_score = model.compute_similarity(video_output, text_output).item()
    return similarity_score

# --- 4. Main Processing Logic ---

def process_tasks(model, processor, device, tasks, output_file, clips_base_dir):
    """
    Processes the assigned tasks (calculating Anchor Query Similarity for each clip).
    """
    
    # Load existing results for resumption
    results = []
    results_map = {}
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            logging.info(f"🔄 Resuming: Loaded {len(results)} existing records.")
            # Map existing successful records
            for item in results:
                if item.get('status') == 'success':
                    results_map[item['problem_id']] = item
        except json.JSONDecodeError:
            logging.warning("⚠️ Existing file is corrupted. Starting fresh.")

    # Filter tasks to run
    tasks_to_run = [t for t in tasks if t['problem_id'] not in results_map]
    
    if not tasks_to_run:
        logging.info("🎉 All assigned tasks are already completed.")
        return

    logging.info(f"🚀 Starting processing for {len(tasks_to_run)} items...")

    # Progress bar
    pbar = tqdm(tasks_to_run, desc=f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}", unit="item")

    for i, item in enumerate(pbar):
        video_id = item['video']
        # Note: Adjust key 'Query' if input JSON structure differs
        query_text = item['query'][0]['Query'] if isinstance(item['query'], list) else item['query']
        pid = item['problem_id']
        
        status = 'fail'
        scores = []
        
        # Retry logic (up to 3 attempts)
        for attempt in range(1, 4):
            try:
                # Locate clips directory
                clips_dir = Path(clips_base_dir) / f"{video_id}_clips"
                
                if not clips_dir.is_dir():
                    raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

                # Find all mp4 clips
                clip_paths = sorted(list(clips_dir.glob(f"*{video_id}*clip_*.mp4")))
                if not clip_paths:
                    raise FileNotFoundError(f"No .mp4 clips found in {clips_dir}")
                
                # Calculate similarity scores for all clips
                temp_scores = []
                for clip_path in clip_paths:
                    score = calculate_similarity(model, processor, device, clip_path, query_text)
                    temp_scores.append(score)
                
                scores = temp_scores
                status = 'success'
                break 

            except Exception as e:
                if attempt == 3:
                    logging.error(f"❌ Failed PID {pid}: {e}")
                time.sleep(1)
        
        # Update result item
        item['anchor_similarity_scores'] = scores
        item['status'] = status
        results_map[pid] = item

        # Periodic Save
        if (i + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(list(results_map.values()), f, indent=4)
    
    # Final Save
    with open(output_file, 'w') as f:
        json.dump(list(results_map.values()), f, indent=4)
    logging.info(f"✅ Processing Complete. Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="STELLA: Anchor Event Scorer")
    parser.add_argument("--gpu_index", type=int, required=True, help="Index of the GPU to use (for data splitting)")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total number of GPUs used")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON with questions")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--clips_dir", type=str, required=True, help="Base directory containing video clips")
    
    args = parser.parse_args()

    # 1. Setup
    setup_environment()
    setup_logging(args.gpu_index, args.output_dir)

    # 2. Load Model
    model, processor, device = load_model()

    # 3. Load Data & Split
    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        return

    with open(args.input_file, 'r') as f:
        all_data = json.load(f)
    
    # Distributed processing logic: Assign tasks based on GPU index
    assigned_tasks = [
        item for i, item in enumerate(all_data) 
        if i % args.total_gpus == args.gpu_index
    ]
    logging.info(f"Assigned {len(assigned_tasks)} / {len(all_data)} tasks to GPU {args.gpu_index}")

    # 4. Run
    output_json_path = os.path.join(args.output_dir, f"anchor_scores_gpu_{args.gpu_index}.json")
    
    process_tasks(model, processor, device, assigned_tasks, output_json_path, args.clips_dir)

if __name__ == "__main__":
    main()