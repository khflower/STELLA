import os
import time
import json
import argparse
import glob
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

"""
STELLA: From Searching to Structuring (CVPR 2026 Submission)
File: logic_extractor.py

Description:
    Implementation of Stage 1: Temporal Logic Extraction.
    This script utilizes an LLM to parse natural language questions into 
    structured Temporal Logic (TL), Atomic Actions, and Anchor Event Queries.
    
    Paper Section: 3.1. Step 1: Temporal Logic Extraction
"""

# --- Configuration ---

# Directory Settings
# Ensure your input data is placed in './data/input' and results will be saved to './data/output'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "data", "output")

# Column Configurations
UNIQUE_ID_COLUMN = 'problem_id'

# Model Configuration
# Note: As described in the paper, we use a high-capacity LLM for logic extraction.
MODEL_NAME = "gpt-5-chat-latest" 

# Error Handling
MAX_RETRIES = 5
RETRY_DELAY = 1.0  # seconds


# --- 1. OpenAI Client Initialization ---

def initialize_openai_client():
    """
    Initializes the OpenAI client.
    Ensure 'OPENAI_API_KEY' is set in your environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Error: OPENAI_API_KEY is not set in environment variables.")
    
    client = OpenAI(api_key=api_key)
    return client


# --- 2. Prompt Generation (Paper Sec 3.1) ---

def create_prompt(question: str) -> list:
    """
    Constructs the prompt for extracting Temporal Logic and Anchor Events.
    Matches the 'Temporal Logic-Guided Narrative Generation Prompt' structure.
    """

    system_prompt = (
        "You are an expert in temporal reasoning and video event understanding. "
        "Your task is to identify atomic actions (minimal independent events), "
        "temporal logic (how those actions are temporally related), "
        "and generate an anchor event query suitable for video retrieval. "
        "The anchor event is the reference action that defines the timing or sequence of other actions "
        "(e.g., in 'What did the man do after putting the black dog down?', "
        "the anchor event is 'the man putting down the black dog'). "
        "The anchor_event_query should be a short, clear natural-language description "
        "that can be used directly as a retrieval query for finding the clip containing that event."
    )

    user_prompt = f"""
    Analyze the following question about actions and their temporal relationships.

    Question: "{question}"

    1. Extract all atomic actions as simple predicates in the form of action(subject, object).
       Example: play(man1, guitar), go_near(man1, man2)

    2. Describe their temporal logic using standard temporal operators such as:
       'before', 'after', 'during', 'until', or 'simultaneous'.

    3. Identify the **anchor event** — the key reference action for temporal reasoning.

    4. Generate a **natural-language query** (anchor_event_query) that can be used
       to retrieve the video clip corresponding to that anchor event.
       Example: "man putting down the black dog"

    Return the result **only** in valid JSON format like this:
    {{
        "atomic_actions": [
            "action(subject, object)",
            "action(subject, object)"
        ],
        "temporal_logic": "description or symbolic form",
        "anchor_event_query": "natural-language description of the anchor event"
    }}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages


# --- 3. Batch Processing Logic ---

def process_batch(client, target_dataset, total_batches, current_batch):
    """
    Processes a specific batch of the dataset to extract temporal logic.
    """
    
    # Define file paths
    input_file_name = f"test_{target_dataset}_with_id.csv"
    input_file_path = os.path.join(INPUT_DATA_DIR, target_dataset, input_file_name)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    output_file_name = f"result_{target_dataset}_batch_{current_batch}_of_{total_batches}_test.csv"
    output_file_path = os.path.join(OUTPUT_DATA_DIR, output_file_name)

    print(f"[*] Starting Batch Processing: {current_batch}/{total_batches}")
    print(f"    Input: {input_file_path}")
    print(f"    Output: {output_file_path}")

    # Load Data
    if not os.path.exists(input_file_path):
        print(f"[Error] Input file not found: {input_file_path}")
        return

    source_df = pd.read_csv(input_file_path)
    print(f"    Total samples in source: {len(source_df)}")

    # Resume capability: Check for existing results
    completed_ids = set()
    if os.path.exists(output_file_path):
        try:
            result_df = pd.read_csv(output_file_path)
            if 'status' in result_df.columns and UNIQUE_ID_COLUMN in result_df.columns:
                completed_ids = set(result_df[result_df['status'] == 'success'][UNIQUE_ID_COLUMN])
                print(f"    Resuming... {len(completed_ids)} samples already completed.")
        except Exception:
            print("    Warning: Could not read existing output file. Starting fresh.")

    # Calculate Batch Range
    batch_size = len(source_df) // total_batches
    start_index = (current_batch - 1) * batch_size
    # Handle the last batch to cover the remainder
    end_index = start_index + batch_size if current_batch < total_batches else len(source_df)
    
    batch_df = source_df.iloc[start_index:end_index].copy()
    
    # Filter out already completed items
    todo_df = batch_df[~batch_df[UNIQUE_ID_COLUMN].isin(completed_ids)]

    # --- [Optional] Debugging / Testing Filter ---
    # Uncomment the lines below to test with specific questions only.
    # target_question = "what does the man do after putting the black dog down"
    # todo_df = todo_df[todo_df['question'].str.strip().str.lower() == target_question.lower()]
    # ---------------------------------------------

    if todo_df.empty:
        print("    No new items to process in this batch.")
        return

    print(f"    Processing {len(todo_df)} samples...")

    # Initialize Output File with Headers if it doesn't exist
    if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
        header_df = source_df.head(0)
        new_columns = ['temporal_logic', 'atomic_action', 'anchor_event_query', 'model_output', 'status', 'error_message']
        for col in new_columns:
            header_df[col] = pd.Series(dtype='object')
        header_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    # Main Inference Loop
    for index, row in tqdm(todo_df.iterrows(), total=len(todo_df), desc=f"Batch {current_batch}"):
        result = {
            'temporal_logic': '', 
            'atomic_action': '', 
            'anchor_event_query': '', # Added specific column for anchor query
            'model_output': '',
            'status': 'fail', 
            'error_message': ''
        }
        
        # Prepare row data
        final_row_data_dict = row.to_dict()

        for attempt in range(MAX_RETRIES):
            try:
                question = row['question']
                messages = create_prompt(question)
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1, # Low temperature for deterministic logic extraction
                    response_format={"type": "json_object"}
                )
                
                model_output_text = response.choices[0].message.content
                result['model_output'] = model_output_text
                
                # Parse JSON output
                output_data = json.loads(model_output_text)
                result['temporal_logic'] = output_data.get('temporal_logic', '')
                result['atomic_action'] = json.dumps(output_data.get('atomic_actions', []))
                result['anchor_event_query'] = output_data.get('anchor_event_query', '')
                
                result['status'] = 'success'
                break # Success, exit retry loop

            except Exception as e:
                # print(f"    [Retry {attempt+1}] Error for ID {row[UNIQUE_ID_COLUMN]}: {e}")
                result['error_message'] = str(e)
                time.sleep(RETRY_DELAY)
        
        # Save results incrementally
        final_row_data_dict.update(result)
        final_row_df = pd.DataFrame([final_row_data_dict])
        final_row_df.to_csv(output_file_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    print(f"[*] Batch {current_batch} completed.")


# --- 4. Result Merging Utility ---

def merge_results(target_dataset):
    """
    Merges all batch result files into a single CSV.
    """
    print(f"\n[*] Merging results for {target_dataset}...")
    
    file_pattern = os.path.join(OUTPUT_DATA_DIR, f"result_{target_dataset}_batch_*.csv")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print("[Error] No result files found to merge.")
        return

    all_data = []
    for f in sorted(file_list):
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except Exception as e:
            print(f"    Skipping corrupted file {f}: {e}")
    
    if not all_data:
        return

    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Deduplication: Keep the latest 'success' or latest attempt
    merged_df.sort_values(by='status', ascending=False, inplace=True) # Ensure 'success' comes first if duplicates exist
    merged_df.drop_duplicates(subset=[UNIQUE_ID_COLUMN], keep='first', inplace=True)
    merged_df.sort_values(by=UNIQUE_ID_COLUMN, inplace=True)
    
    final_output_path = os.path.join(OUTPUT_DATA_DIR, f"FINAL_result_{target_dataset}.csv")
    merged_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
    
    print(f"[*] Merge Complete. Saved to: {final_output_path}")
    print(f"    Total unique records: {len(merged_df)}")


# --- 5. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STELLA Stage 1: Temporal Logic Extraction")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['intent_qa', 'next_qa'],
                        help="Target dataset name (e.g., intent_qa, next_qa)")
    parser.add_argument('--total_batches', type=int, default=1,
                        help="Total number of batches to split the dataset into")
    parser.add_argument('--current_batch', type=int, default=1,
                        help="Current batch index to process (1-based)")
    parser.add_argument('--merge_only', action='store_true',
                        help="If set, only runs the merge process without inference")

    args = parser.parse_args()

    if args.merge_only:
        merge_results(args.dataset)
    else:
        try:
            client = initialize_openai_client()
            process_batch(
                client=client,
                target_dataset=args.dataset,
                total_batches=args.total_batches,
                current_batch=args.current_batch
            )
        except Exception as e:
            print(f"[Critical Error] {e}")