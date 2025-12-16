import sys
import os

# ==============================================================================
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ==============================================================================
# [Imports]
# ==============================================================================
import json
import google.generativeai as genai

try:
    from modules.logic_extractor import create_prompt as get_logic_prompt
    from modules.video_processor import ActionBasedVideoSplitter
    from modules.anchor_scorer import calculate_similarity
    from modules.candidate_pruner import calculate_similarity as calc_option_sim
    from modules.narrative_generator import NarrativeGenerator
    from modules.reasoner import Reasoner
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class STELLA:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
        self.narrator = NarrativeGenerator()
        self.reasoner = Reasoner()
        
        # from modules.anchor_scorer import load_model
        # self.vlm_model, self.vlm_processor, self.device = load_model()
        print("✅ STELLA Framework Initialized (Legacy API Version).")

    def run(self, video_path: str, question: str, options_dict: dict):
        print(f"🚀 [Start] Processing Question: {question}")
        
        # 1. Logic Extraction (Dummy for test)
        tl_result = {
            "temporal_logic": "do(man, ?) after put_down(man, dog)",
            "anchor_event_query": "man putting down the black dog" 
        }
        
        # 2. Scoping (Video Splitting)
        splitter = ActionBasedVideoSplitter()
        scope_clips = [video_path] 

        # 3. Coarse Stage
        print("   -> Entering Coarse Stage...")
        coarse_narrative = self.narrator.generate(scope_clips, tl_result['temporal_logic'])
        
        options_text = "\n".join([f"{k.upper()}) {v}" for k, v in options_dict.items()])
        pred_answer = self.reasoner.predict_answer(video_path, coarse_narrative, question, options_text)
        las = self.reasoner.calculate_las(video_path, coarse_narrative, question, options_text, pred_answer)
        
        print(f"   -> Coarse LAS: {las} (Answer: {pred_answer})")
        
        if las >= 3:
            return pred_answer

        # 4. Fine Stage
        print("   -> Low Confidence. Entering Fine Stage...")
        fine_narrative = self.narrator.generate(scope_clips, tl_result['temporal_logic']) # FPS logic can be added inside generate
        final_answer = self.reasoner.predict_answer(video_path, fine_narrative, question, options_text)
        
        return final_answer

if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables.")
        sys.exit(1)

    stella = STELLA(api_key)
    
    video_path = ""
    q = "What does the man do after putting the black dog down?"
    opts = {'a0': 'walk away', 'a1': 'pet the dog', 'a2': 'pick up bag', 'a3': 'sit down', 'a4': 'run'}
    
    if os.path.exists(video_path):
        print(f"▶️ Testing with: {video_path}")
        try:
            result = stella.run(video_path, q, opts)
            print(f"\n🏆 Final Answer: {result}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Video file not found: {video_path}")