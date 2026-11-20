
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain


app = FastAPI(
    title="Conversational RAG API",
    description="Conversational RAG Chatbot using LangChain and FastAPI",
    version="1.0.0"
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Conversation History Storage
store: Dict[str, List] = {}


# Request Schema
class ChatRequest(BaseModel):
    session_id: str
    question: str


# Home Route
@app.get("/")
def home():
    return {
        "message": "Welcome to Conversational RAG API"
    }


# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Chat API
@app.post("/chat")
def chat(request: ChatRequest):

    try:

        session_id = request.session_id
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Get Session History
        if session_id not in store:
            store[session_id] = []

        chat_history = store[session_id]

        # Invoke RAG Chain
        answer = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history[-10:]
        })

        # Store User Message
        chat_history.append(
            HumanMessage(content=question)
        )

        # Store AI Response
        chat_history.append(
            AIMessage(content=answer)
        )

        return {
            "session_id": session_id,
            "question": question,
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Error processing your question"
        )


# Clear Session History
@app.delete("/clear-history/{session_id}")
def clear_history(session_id: str):

    store.pop(session_id, None)

    return {
        "message": "Chat history cleared successfully"
    }