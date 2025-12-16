import time
import json
import os
import google.generativeai as genai 
from prompts import QA_REASONING_PROMPT, LAS_SCORING_PROMPT

class Reasoner:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def _upload_file(self, path: str):
        """Uploads file using the original google.generativeai API."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        print(f"⏳ Uploading {os.path.basename(path)}...", end="", flush=True)
        
        video_file = genai.upload_file(path=path)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"File upload failed: {video_file.state.name}")
        
        print(" Done.")
        return video_file

    def predict_answer(self, video_path: str, description: str, question: str, options: str) -> str:
        """Performs QA Reasoning."""
        video_file = None
        try:
            video_file = self._upload_file(video_path)
            model = genai.GenerativeModel(self.model_name)
            
            prompt = QA_REASONING_PROMPT.format(
                description=description,
                question=question,
                options=options
            )


            response = model.generate_content(
                [video_file, prompt],
                request_options={"timeout": 600}
            )
            
            if not response.text:
                return "A"
                
            return response.text.strip().upper()[:1] 

        finally:
            if video_file:
                try:
                    genai.delete_file(video_file.name)
                except Exception:
                    pass

    def calculate_las(self, video_path: str, description: str, question: str, options: str, predicted_answer: str) -> int:
        """Calculates Logical Alignment Score (LAS)."""
        video_file = None
        try:
            video_file = self._upload_file(video_path)
            model = genai.GenerativeModel(self.model_name)
            
            prompt = LAS_SCORING_PROMPT.format(
                question=question,
                options=options,
                description=description,
                predicted_answer=predicted_answer
            )

            response = model.generate_content(
                [video_file, prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                result = json.loads(response.text)
                return int(result.get("score", 1))
            except (json.JSONDecodeError, ValueError, AttributeError):
                return 1
                
        except Exception:
            return 1
        finally:
            if video_file:
                try:
                    genai.delete_file(video_file.name)
                except Exception:
                    pass