import os
import re
import json
from agent import SHLAgent

def parse_markdown_trace(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        content = f.read()
        
    turns = []
    # Split content by Turn headers, e.g. "### Turn 1"
    turn_blocks = re.split(r"### Turn \d+", content)
    if len(turn_blocks) <= 1:
        return []
        
    conversation_so_far = []
    
    for block in turn_blocks[1:]:
        block = block.strip()
        if not block:
            continue
            
        # Extract User content
        # Pattern: **User**\n\n> (.*)
        user_match = re.search(r"\*\*User\*\*\s*\n+\s*>\s*(.*?)(?=\n+\s*\*\*Agent\*\*|\Z)", block, re.DOTALL)
        if not user_match:
            continue
        user_text = user_match.group(1).strip()
        # Remove markdown quotes from user text
        user_text = re.sub(r"^>\s*", "", user_text, flags=re.MULTILINE)
        
        # Extract Agent content
        # Pattern: **Agent**\n\n(.*)(?=\n+\s*_No recommendations|_`end_of_conversation`|$)
        agent_match = re.search(r"\*\*Agent\*\*\s*\n+\s*(.*?)(?=\n+\s*_[N`_]|$)", block, re.DOTALL)
        agent_text = agent_match.group(1).strip() if agent_match else ""
        
        # Extract Expected Recommendations
        # Check if contains tables
        # Example: | 1 | Occupational Personality Questionnaire OPQ32r | P | ...
        expected_recs = []
        table_match = re.findall(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([A-Z])\s*\|[^|]*\|[^|]*\|[^|]*\|\s*<([^>]+)>", block)
        for name, t_type, url in table_match:
            expected_recs.append(name.strip())
            
        # Extract end_of_conversation
        # Pattern: _`end_of_conversation`: **(true|false)**_
        eoc_match = re.search(r"_`end_of_conversation`:\s*\*\*(true|false)\*\*", block)
        expected_eoc = eoc_match.group(1) == "true" if eoc_match else False
        
        turns.append({
            "user": user_text,
            "expected_agent_reply": agent_text,
            "expected_recommendations": expected_recs,
            "expected_end_of_conversation": expected_eoc
        })
        
    return turns

def run_tests():
    agent = SHLAgent("/home/aryan/shl-assessment/shl_product_catalog.json")
    
    traces_dir = "/home/aryan/shl-assessment/sample_conversations/GenAI_SampleConversations"
    files = sorted([f for f in os.listdir(traces_dir) if f.endswith(".md")], key=lambda x: int(re.search(r"\d+", x).group()))
    
    print("=" * 60)
    print("SHL ASSESSMENT AGENT SIMULATOR & TESTER")
    print("=" * 60)
    
    api_key_set = any(os.environ.get(k) for k in ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"])
    if not api_key_set:
        print("WARNING: No LLM API keys set in current environment variables.")
        print("The simulator will run using mock responses or report status.")
        print("Please export GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY to test real LLM responses.")
        print("=" * 60)
        return
        
    for file in files[:3]: # Let's run first 3 files to save tokens and time
        filepath = os.path.join(traces_dir, file)
        print(f"\nRunning test trace: {file}")
        turns = parse_markdown_trace(filepath)
        
        history = []
        for i, turn in enumerate(turns):
            print(f"  Turn {i+1}:")
            print(f"    User: {turn['user']}")
            
            history.append({"role": "user", "content": turn["user"]})
            
            # Call agent
            res = agent.process_chat(history)
            
            print(f"    Agent Reply: {res['reply'][:120]}...")
            print(f"    Recommendations: {[r['name'] for r in res['recommendations']]}")
            print(f"    End of Conversation: {res['end_of_conversation']}")
            
            # Compare with expected
            expected_names = turn["expected_recommendations"]
            actual_names = [r["name"] for r in res["recommendations"]]
            
            print(f"    Expected Recs: {expected_names}")
            print(f"    Expected EOC: {turn['expected_end_of_conversation']}")
            
            # Append agent response to history for next turn
            history.append({"role": "assistant", "content": res["reply"]})
            print("-" * 40)

if __name__ == "__main__":
    run_tests()
