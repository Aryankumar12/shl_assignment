from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from agent import SHLAgent
import uvicorn
import os

app = FastAPI(title="SHL Assessment Recommender API")

# Initialize SHLAgent
CATALOG_PATH = "/home/aryan/shl-assessment/shl_product_catalog.json"
agent = SHLAgent(CATALOG_PATH)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Convert Pydantic request messages to raw list of dicts for the agent
    history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if not history:
        raise HTTPException(status_code=400, detail="Conversation history cannot be empty")
        
    try:
        response = agent.process_chat(history)
        return ChatResponse(
            reply=response["reply"],
            recommendations=[
                Recommendation(
                    name=rec["name"],
                    url=rec["url"],
                    test_type=rec["test_type"]
                ) for rec in response["recommendations"]
            ],
            end_of_conversation=response["end_of_conversation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
