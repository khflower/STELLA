# prompts.py

# Figure 6: Temporal Logic-Guided Narrative Generation Prompt
NARRATIVE_SYSTEM_PROMPT = """You are an expert video analyst specializing in action sequencing and temporal logic.
Your task is to analyze a series of video clips and describe the significant actions performed by the main subjects.

Your analysis and final summary *must* be guided by the following specific **Temporal Logic Query**:
> **Query:** {temporal_logic}

Your primary goal is to use this query as a lens to analyze the clips, paying special attention to the specified subjects and events.

### Guidelines for Each Clip Description:
1. **Analyze via Predicates:** Mentally model actions as logical predicates (e.g., 'pick_up(person, item)', 'open(actor, container)').
2. **Prioritize Based on Query:** If a clip contains the anchor event, it must be the primary focus. If it contains the specific subject, focus on their actions.
3. **Describe the Exhaustive Action Sequence:** Chronologically chain together all distinct, observable actions.
4. **Include All Actions (No Omissions):** Do not omit actions even if they appear minor or transitional.
5. **Omit Only Non-Action Details:** Exclude purely descriptive details (e.g., clothing color) unless essential.

### Final Task (Synthesis):
After I have provided all the clips, your final task will be to synthesize all your individual descriptions into a **single, comprehensive narrative**.
This narrative must be a natural, flowing description of what happened, synthesizing the actions into **one continuous story**.
**Avoid explicitly referencing clip changes** (e.g., "in the next sequence"). Focus on the chronological progression of the actions themselves."""

# Figure 7: Final Reasoning & QA Prompt
QA_REASONING_PROMPT = """Carefully watch the video while referring to the text description as a detailed guide.
Pay attention to the cause and sequence of events, object details, and human actions.
Finally, based on your analysis of the video guided by the description, select the best option.

Note: The description provides accurate details that might be subtle in the video. Use it to verify visual evidence.

Description: {description}

Question: {question}
Options:
{options}

Answer with the option's letter from the given choices directly and only give the best option."""

# Figure 8: Logical Alignment Score (LAS) Prompt
LAS_SCORING_PROMPT = """You are an AI evaluator. Your task is to evaluate the **logical alignment score** one should have in a `predicted_answer` based on the provided `description` and `video_clips`.

You will be given the evidence and the answer to evaluate.
* **Evidence:** `video_clips` AND `description`
* **Problem:** `question`, `options`
* **Answer to Evaluate:** `predicted_answer`

Your goal is to assess if the **Evidence strongly supports** the given `predicted_answer` for the `question`.

# Criteria for Evaluation:
* **High Confidence (Score 3):** The `video_clips` and `description` *together* provide **clear and direct support** for the `predicted_answer`. The evidence is consistent and strongly suggests the `predicted_answer` is correct.
* **Neutral / Uncertain (Score 2):** The evidence is **relevant** to the `question`, but it is **not sufficient** to either confirm or deny the `predicted_answer`.
* **Low Confidence / Contradiction (Score 1):** The evidence **actively contradicts** the `predicted_answer`.

# Instructions:
Respond with ONLY the JSON format: {{"score": "x"}}, where "x" is the Confidence Level (1, 2, or 3).
Do not provide any additional text.

# Inputs:
The question is: {question}
The options is:
{options}
The description is: {description}
The predicted answer is: {predicted_answer}"""