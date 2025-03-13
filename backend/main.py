from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from llm_control import parse_result, reset  # Import the functions

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    print("Request received:")
    print(request.messages)

    # Extract the last message from the chat history
    last_message = request.messages[-1].content
    
    # Call parse_result with the last message
    response_content = parse_result(last_message)
    
    # Create response message
    assistant_message = Message(
        role="assistant",
        content=str(response_content)  # Convert to string in case response is not a string
    )
    
    return {"message": assistant_message}

@app.post("/reset")
async def reset_chat():
    print("Reset request received")
    response = reset()  # Call the reset function
    return {"status": "success", "message": response} 