import time
import os
from typing import List
import google.generativeai as genai 
from prompts import NARRATIVE_SYSTEM_PROMPT

class NarrativeGenerator:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def generate(self, clips: List[str], temporal_logic: str, fps: int = 1) -> str:
        """
        Generates a narrative using Multi-turn conversation.
        """
        if not clips:
            return "No visual evidence found."

        formatted_system_prompt = NARRATIVE_SYSTEM_PROMPT.format(temporal_logic=temporal_logic)
        
        model = genai.GenerativeModel(self.model_name)
        
        chat = model.start_chat(history=[
            {'role': 'user', 'parts': [formatted_system_prompt]},
            {'role': 'model', 'parts': ["Understood. I am ready."]}
        ])

        uploaded_files = []
        
        try:
            for clip_path in clips:
                if not os.path.exists(clip_path): continue
                
                print(f"⏳ Uploading clip: {os.path.basename(clip_path)}")
                video_file = genai.upload_file(path=clip_path)
                
                while video_file.state.name == "PROCESSING":
                    time.sleep(1)
                    video_file = genai.get_file(video_file.name)
                
                uploaded_files.append(video_file)
                
                chat.send_message(
                    ["Describe this clip in detail based on the query.", video_file],
                    request_options={"timeout": 600}
                )
                time.sleep(1)
            final_response = chat.send_message(
                "Now, synthesize all the above observations into a single, continuous narrative resolving the query."
            )
            return final_response.text

        finally:
            for f in uploaded_files:
                try:
                    genai.delete_file(f.name)
                except Exception:
                    pass