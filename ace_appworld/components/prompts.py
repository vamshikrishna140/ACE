import json
from typing import Dict, List, Optional
from ace_appworld.components.models import Episode

"""
All prompt templates for the ACE AppWorld system.
Contains functions to build prompts for Generator, Reflector, and Curator.
"""

def get_generator_prompt(playbook_json: str, task: str, main_user: Dict[str, str], 
                         conversation_history: str = "") -> str:
    """
    Build the generator prompt with playbook and conversation history
    """
    base_prompt = """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using their associated APIs on my behalf. For this you will undertake a multi-step conversation using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal.

This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information:

```python
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. spotify
print(apis.api_docs.show_api_descriptions(app_name='spotify'))

# To get the specification of a particular api, e.g. spotify app's login api
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that I will execute, to solve the task.

You are also provided with a curated cheatsheet of strategies, API-specific information, common mistakes, and proven solutions to help you solve the task effectively.

ACE Playbook - Read the Playbook first, then execute the task by explicitly leveraging each relevant section:

PLAYBOOK_BEGIN
{playbook}
PLAYBOOK_END

Key instructions:
1. Make sure to end code blocks with ``` followed by a newline().
2. Remember you can use the variables in your code in subsequent code blocks.
3. You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.
4. Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.
5. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
6. Many APIs return items in "pages". Make sure to run through all the pages by looping over page_index.
7. Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>).
8. Treat the cheatsheet as a tool. Use only the parts that are relevant and applicable to your specific situation and task context, otherwise use your own judgement.

My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

Task: {task}

{history}

Generate the next step of code to solve this task. Think step by step.
"""
    
    return base_prompt.format(
        playbook=playbook_json,
        task=task,
        first_name=main_user.get("first_name", ""),
        last_name=main_user.get("last_name", ""),
        email=main_user.get("email", ""),
        phone_number=main_user.get("phone_number", ""),
        history=conversation_history
    )


def get_reflection_prompt(task_instruction: str,
                          trajectory: List[Dict],
                          final_answer: Optional[str],
                          ground_truth: Optional[Dict],
                          execution_feedback: str,
                          playbook_bullets: Optional[List[Dict]] = None) -> str:
    """
    Build reflection prompt for trajectory analysis
    """
    trajectory_json = json.dumps({
        'task': task_instruction,
        'steps': trajectory,
        'final_answer': final_answer
    }, indent=2)
    
    ground_truth_code = ""
    if ground_truth and ground_truth.get('code'):
        ground_truth_code = ground_truth['code']
    elif ground_truth and ground_truth.get('answer'):
        ground_truth_code = f"# Expected answer: {ground_truth['answer']}"
    
    playbook_json = ""
    if playbook_bullets:
        playbook_sections = {}
        for bullet in playbook_bullets:
            section = bullet.get('section', 'general')
            if section not in playbook_sections:
                playbook_sections[section] = []
            playbook_sections[section].append(
                f"[{bullet['id']}] (helpful={bullet.get('helpful', 0)}, harmful={bullet.get('harmful', 0)}): {bullet['content']}"
            )
        
        for section, bullets in playbook_sections.items():
            playbook_json += f"\n## {section}\n"
            for bullet in bullets:
                playbook_json += f"- {bullet}\n"
    
    prompt = f"""You are an expert AppWorld coding agent and educator. Your job is to diagnose the current trajectory: identify what went wrong (or could be better), grounded in execution feedback, API usage, unit test report, and ground truth when applicable.

Instructions: 
- Carefully analyze the model's reasoning trace to identify where it went wrong 
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap 
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies 
- Provide actionable insights that could help the model avoid this mistake in the future 
- Identify root causes: wrong source of truth, bad filters (timeframe/direction/identity), formatting issues, or missing authentication and how to correct them
- Provide concrete, step-by-step corrections the model should take in this task
- Be specific about what the model should have done differently 
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer) 
- Explicitly curate from the environment feedback the output format/schema of APIs used when unclear or mismatched with expectations (e.g., apis.blah.show_contents() returns a list of content_ids (strings), not content objects)

Inputs:

Ground truth code (reference, known-correct):
GROUND_TRUTH_CODE_START
{ground_truth_code}
GROUND_TRUTH_CODE_END

Test report (unit tests result for the task after the generated code was run):
TEST_REPORT_START
{execution_feedback}
TEST_REPORT_END

ACE playbook (playbook that's used by model for code generation):
PLAYBOOK_START
{playbook_json}
PLAYBOOK_END

Outputs: 
Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations 
- error_identification: what specifically went wrong in the reasoning? 
- root_cause_analysis: why did this error occur? What concept was misunderstood? 
- correct_approach: what should the model have done instead? 
- key_insight: what strategy, formula, or principle should be remembered to avoid this error?
- bullet_tags: A list of dicts, e.g., [{{"id": "gen-00001", "tag": "helpful"}}, {{"id": "api-00002", "tag": "harmful"}}]

Answer in this exact JSON format (no markdown code blocks):
{{
"reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
"error_identification": "[What specifically went wrong in the reasoning?]",
"root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
"correct_approach": "[What should the model have done instead?]",
"key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
"bullet_tags": [{{"id": "[bullet_id_1]", "tag": "[helpful/harmful/neutral]"}}]
}}

[FULL AGENT-ENVIRONMENT TRAJECTORY]
{trajectory_json}"""
    
    return prompt


def get_curation_prompt(task_context: str,
                       current_playbook: Dict,
                       reflection_result: Dict) -> str:
    """
    Build curation prompt for playbook updates
    """
    prompt = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook will be used by an AI agent to solve similar tasks.
- The reflection was generated using ground truth (which won't be available when using the playbook).
- You must extract insights that help the agent align with the ground truth *without* revealing the answers (e.g., "The agent should use `apis.phone.search_contacts()` to find roommates" is GOOD. "The answer is 'John Smith'" is BAD).

**Instructions:**
1.  **Review Existing Playbook & Reflection:** Identify what *new, actionable* insights are *missing* from the playbook.
2.  **Avoid Redundancy:** Do NOT add an insight if a similar one already exists. Only add complementary content.
3.  **Be Specific:** Insights must be concise and actionable. "Check for errors" is bad. "After calling `apis.venmo.pay()`, check the 'error' field in the response" is good.
4.  **Format Correctly:** Only provide ADD operations.

**Available Sections:**
- `strategies_and_hard_rules`: Core strategies and mandatory rules (e.g., "Always use `while True` for pagination").
- `apis_to_use_for_specific_information`: API usage patterns and gotchas (e.g., "To get user emails, use `apis.supervisor.get_user_account()` not `apis.phone.search_contacts()`").
- `common_mistakes`: Errors to avoid (e.g., "Do not assume `search_transactions` returns all items; check for `next_page` token").
- `verification_checklist`: Validation steps (e.g., "Before completing, verify the total number of items processed matches the `total` field").
- `domain_concepts`: Domain-specific knowledge (e.g., "A 'roommate' is defined in the `apis.phone.search_contacts()` app").
- `code_patterns`: Reusable code snippets (e.g., "Pagination loop: `page=0; while True: ...`").

---

**Task Context:**
"""
    prompt += f"{task_context}\n\n"
    
    prompt += "**Current Playbook (Sample):**\n"
    if current_playbook:
        for section, bullets in current_playbook.items():
            if bullets:
                prompt += f"\n### {section}\n"
                for bullet in bullets[:5]:  # Show first 5 per section
                    prompt += f"- [{bullet.get('id', '?')}] {bullet.get('content', '')}\n"
    else:
        prompt += "(Empty playbook)\n"
    
    prompt += f"""
**Reflection Analysis:**
- Error: {reflection_result.get('error_identification', '')}
- Root Cause: {reflection_result.get('root_cause_analysis', '')}
- Correct Approach: {reflection_result.get('correct_approach', '')}
- Key Insight: {reflection_result.get('key_insight', '')}

---

**Your Task:**
Output ONLY a valid JSON object with these fields:
- "reasoning": Your thought process on why you are adding/not-adding insights.
- "operations": A list of operations (ONLY "ADD" type).

**Response Format (no markdown, no code blocks):**
{{
    "reasoning": "[Your analysis of what's missing and why it should be added, or why nothing is new]",
    "operations": [
        {{
            "type": "ADD",
            "section": "common_mistakes",
            "content": "[New strategy or rule based on the reflection's key insight]"
        }}
    ]
}}
"""
    return prompt


def get_validation_prompt(episode: Episode, ground_truth: Dict) -> str:
    """
    Build validation prompt for LLM-based trajectory validation
    """
    prompt = f"""You are a validation agent for task completion verification.

**Task Instruction:**
{episode.instruction}

**Agent Execution Trajectory:**
"""
    
    for i, step in enumerate(episode.steps, 1):
        prompt += f"""
--- Step {i} ---
Reasoning: {step.thought}

Action:
```python
{step.action}
```

Observation: {step.observation}
Success: {step.success}
"""
    
    prompt += f"""
**Final Answer:** {episode.final_answer if episode.final_answer else "No explicit answer provided"}

**Ground Truth:**
"""
    
    if ground_truth['answer'] is not None:
        prompt += f"Expected Answer: {json.dumps(ground_truth['answer'], indent=2)}\n"
    else:
        prompt += "Expected Answer: Not explicitly specified\n"
    
    if ground_truth['private_data']:
        prompt += f"\nValidation Data: {json.dumps(ground_truth['private_data'], indent=2)}\n"
    
    prompt += """
**Validation Criteria:**
1. Analyze each step in the trajectory to understand the agent's reasoning and actions
2. Check if the sequence of actions logically accomplishes the task instruction
3. Compare the final answer (if provided) with the ground truth
4. If no explicit answer, verify that the actions successfully completed the task
5. Consider semantic equivalence (e.g., "yes"/"true", date format variations, equivalent phrasings)
6. Review observations to confirm correct data was retrieved or modified
7. Check if the final action indicates task completion

**Important Notes:**
- Exact string matching is NOT required; focus on semantic correctness
- The task may be completed through actions without an explicit answer
- Consider the overall trajectory, not just the final answer
- Verify that side effects (updates, deletions, creations) were properly executed

**Response Format:**
Respond with ONLY one of:
- SUCCESS: [concise reason why the task was completed correctly]
- FAILURE: [concise reason why the task was not completed or completed incorrectly]

Be decisive and clear in your judgment.
"""
    
    return prompt