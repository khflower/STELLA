import numpy as np
from pathlib import Path
from typing import List, Tuple
import cv2
import subprocess
import shutil
import os
import argparse
import sys

"""
STELLA: From Searching to Structuring (CVPR 2026 Submission)
File: video_processor.py

Description:
    Implementation of the Video Segmentation module (Algorithm 1: SEGMENT(V)).
    This script detects significant motion changes in a video to split it into 
    candidate clips for the subsequent Scoping stage.
"""

class ActionBasedVideoSplitter:
    """
    Splits video into segments based on motion change detection.
    """
    
    def __init__(self):
        self.splits = []
    
    def detect_motion_changes(self, video_path: str, sensitivity: float = 0.3) -> List[Tuple[float, float]]:
        """
        Detects action segments based on motion changes.
        
        Args:
            video_path (str): Path to the input video file.
            sensitivity (float): Threshold sensitivity for motion detection.
            
        Returns:
            List[Tuple[float, float]]: A list of (start_time, end_time) tuples.
        """
        print(f"🏃 [Detection] Starting Motion-based Segmentation (Sensitivity: {sensitivity})")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ [Error] Cannot open video file.")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("❌ [Error] Cannot read FPS. Skipping file.")
            cap.release()
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return []
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_scores = []
        frame_count = 0
        
        print("📊 Analyzing motion per frame...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)
            
            prev_gray = gray
            frame_count += 1
            
            if frame_count % 100 == 0 and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                sys.stdout.write(f"\r    ... Progress: {progress:.1f}%")
                sys.stdout.flush()
        
        print()
        cap.release()
        
        motion_changes = self._find_motion_segments(motion_scores, sensitivity, fps)
        
        segments = []
        for start_frame, end_frame in motion_changes:
            start_time = start_frame / fps
            end_time = end_frame / fps
            segments.append((start_time, end_time))
        
        self.splits = segments
        
        print(f"🎯 Total {len(segments)} clips detected:")
        for i, (start, end) in enumerate(segments):
            print(f"    🎬 Clip {i+1}: {start:.2f}s ~ {end:.2f}s (Duration: {end-start:.2f}s)")
        
        return segments
    
    def _find_motion_segments(self, motion_scores: List[float], sensitivity: float, fps: float) -> List[Tuple[int, int]]:
        """
        Internal method to determine segmentation points based on statistical thresholds.
        """
        if not motion_scores or fps == 0:
            return []

        mean_motion = np.mean(motion_scores)
        std_motion = np.std(motion_scores)
        threshold = mean_motion + (sensitivity * std_motion * 2.0)
        
        change_points = [0]
        current_state = motion_scores[0] > threshold
        min_segment_length = int(5.0 * fps) # Minimum clip length constraint (e.g., 5 seconds)
        
        for frame, score in enumerate(motion_scores[1:], 1):
            new_state = score > threshold
            if new_state != current_state:
                if frame - change_points[-1] >= min_segment_length:
                    change_points.append(frame)
                    current_state = new_state
        
        if len(motion_scores) - change_points[-1] >= min_segment_length:
            change_points.append(len(motion_scores))
        elif len(change_points) > 1:
            change_points[-1] = len(motion_scores)
        
        segments = []
        for i in range(len(change_points) - 1):
            start_frame = change_points[i]
            end_frame = change_points[i + 1]
            segments.append((start_frame, end_frame))
        
        if not segments:
            segments = [(0, len(motion_scores))]
        
        return segments
    
    def extract_clips(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extracts video clips using FFmpeg based on detected segments.
        """
        if not self.splits:
            print("❌ [Error] No segments found. Run detect_motion_changes() first.")
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        clip_paths = []
        
        for i, (start_time, end_time) in enumerate(self.splits):
            output_file = output_path / f"{video_name}_clip_{i+1:03d}.mp4"
            duration = end_time - start_time
            
            # Skip extremely short clips (< 0.5s)
            if duration < 0.5:
                continue

            # Construct FFmpeg command
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_path),
                '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast', '-crf', '23', str(output_file)
            ]
            
            print(f"✂️  Extracting Clip {i+1}: {start_time:.2f}s ~ {end_time:.2f}s")
            
            # Suppress FFmpeg logs unless error
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 1024:
                clip_paths.append(str(output_file))
                # file_size = output_file.stat().st_size / (1024 * 1024)
                # print(f"    ✅ Saved: {output_file.name} ({file_size:.2f}MB)")
            else:
                print(f"❌ [Error] Failed to save {output_file.name}")
                if result.stderr:
                    print(f"    Error Log: {result.stderr.strip()}")
        
        print(f"\n✅ Extraction Complete! {len(clip_paths)} clips saved in '{output_dir}'.")
        return clip_paths


if __name__ == "__main__":
    # Argument Parser for flexibility
    parser = argparse.ArgumentParser(description="STELLA Video Segmentation Tool")
    parser.add_argument('--input_dir', type=str, default='./data/input_videos', help='Directory containing input video files (.mp4)')
    parser.add_argument('--output_dir', type=str, default='./data/output_clips', help='Base directory to save extracted clips')
    parser.add_argument('--sensitivity', type=float, default=0.5, help='Motion detection sensitivity (higher = more segments)')
    
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_base_path = Path(args.output_dir)

    print("🎬 STELLA Video Segmenter (Motion-based)")
    print("=" * 60)
    print(f"📂 Input Directory:  {input_path}")
    print(f"📂 Output Directory: {output_base_path}")
    print(f"⚙️  Sensitivity:     {args.sensitivity}")
    print("=" * 60)

    if not input_path.exists():
        print(f"❌ [Error] Input directory does not exist: {input_path}")
        exit()

    # Find all MP4 files
    video_files = list(input_path.glob("*.mp4"))
    
    if not video_files:
        print("⚠️  No .mp4 files found in the input directory.")
        exit()
        
    print(f"📄 Found {len(video_files)} video files.")

    # Process each video
    for video_file in video_files:
        print(f"\n\n🚀 Processing: {video_file.name}")
        
        # Create a specific folder for this video's clips
        output_dir_for_video = output_base_path / f"{video_file.stem}_clips"
        
        # Clean up existing directory if needed (Safe remove)
        if output_dir_for_video.exists():
            print(f"    ⚠️  Cleaning up existing directory: {output_dir_for_video}")
            shutil.rmtree(output_dir_for_video)
            
        splitter = ActionBasedVideoSplitter()
        
        # 1. Detect Motion
        segments = splitter.detect_motion_changes(str(video_file), sensitivity=args.sensitivity)
        
        # 2. Extract Clips
        if segments:
            clips = splitter.extract_clips(str(video_file), str(output_dir_for_video))
        else:
            print("❌ No segments detected.")
            
        print("-" * 60)

    print("\n\n🎉 All tasks completed successfully! 🎉")