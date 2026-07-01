import os
import urllib.request
import json
import logging

logger = logging.getLogger(__name__)

def call_llm(system_prompt: str, messages: list[dict]) -> dict:
    """
    Calls Gemini, Groq, or OpenAI depending on what API key is found.
    Returns a dict with key 'content'.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    
    # Also check if any key was passed in config or is in home directory .env files for local testing
    if not (gemini_key or groq_key or openai_key):
        # Search parent directory's .env files for developer convenience
        for p in [
            "/home/aryan/ai-quiz-builder/backend/.env",
            "/home/aryan/js-capstone-project/backend/.env",
            "/home/aryan/shl-assessment/.env"
        ]:
            if os.path.exists(p):
                with open(p, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        elif line.startswith("GROQ_API_KEY="):
                            groq_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        elif line.startswith("OPENAI_API_KEY="):
                            openai_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    if gemini_key:
        return _call_gemini(system_prompt, messages, gemini_key)
    elif groq_key:
        return _call_groq(system_prompt, messages, groq_key)
    elif openai_key:
        return _call_openai(system_prompt, messages, openai_key)
    else:
        raise ValueError("No API Key found. Please set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY environment variable.")

def _call_gemini(system_prompt: str, messages: list[dict], api_key: str) -> dict:
    # Convert conversation messages to Gemini's format
    # Gemini API expects: contents: [ { role: "user"|"model", parts: [ { text: "..." } ] } ]
    # Note: system instruction can be passed separately.
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    data = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
    
    with urllib.request.urlopen(req, timeout=25) as response:
        res = json.loads(response.read().decode())
        try:
            text = res["candidates"][0]["content"]["parts"][0]["text"]
            return {"content": text}
        except (KeyError, IndexError) as e:
            logger.error(f"Gemini unexpected response: {res}")
            raise ValueError(f"Gemini API error: {e}")

def _call_groq(system_prompt: str, messages: list[dict], api_key: str) -> dict:
    # Format messages
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": api_messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=25) as response:
        res = json.loads(response.read().decode())
        text = res["choices"][0]["message"]["content"]
        return {"content": text}

def _call_openai(system_prompt: str, messages: list[dict], api_key: str) -> dict:
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": api_messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=25) as response:
        res = json.loads(response.read().decode())
        text = res["choices"][0]["message"]["content"]
        return {"content": text}
