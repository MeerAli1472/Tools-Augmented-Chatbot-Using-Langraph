from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
from typing import Optional
import uuid
import subprocess
import socket
import time

# Import YOUR existing backend
from langgraph_backend import (
    app_graph,
    ingest_pdf,
    retreive_all_threads,
    thread_document_metadata,
)

app = FastAPI(title="LangGraph RAG & Tools API")


# ==============================
# Auto-launch Streamlit
# ==============================

def is_port_open(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@app.get("/chat")
def launch_streamlit():
    STREAMLIT_PORT = 8501

    if not is_port_open(STREAMLIT_PORT):
        subprocess.Popen(["streamlit", "run", "streamlit_frontend.py"])

        # Wait until Streamlit starts (max 10 sec)
        for _ in range(20):
            if is_port_open(STREAMLIT_PORT):
                break
            time.sleep(0.5)

    return RedirectResponse("http://localhost:8501")


# ==============================
# Request Model
# ==============================

class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


# ==============================
# Utilities
# ==============================

def build_config(thread_id: str):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }


def serialize_message(msg):
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    elif isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_name": msg.name,
        }
    return {"role": "unknown", "content": str(msg)}


# ==============================
# Chat Endpoint
# ==============================

@app.post("/chat")
def chat(request: ChatRequest):

    thread_id = request.thread_id or str(uuid.uuid4())
    config = build_config(thread_id)

    try:
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None

    return {
        "thread_id": thread_id,
        "response": last_message.content if last_message else "",
        "document_metadata": thread_document_metadata(thread_id),
    }


# ==============================
# Streaming Chat
# ==============================

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    thread_id = request.thread_id or str(uuid.uuid4())
    config = build_config(thread_id)

    def event_stream():
        for message_chunk, _ in app_graph.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content

    return StreamingResponse(event_stream(), media_type="text/plain")


# ==============================
# PDF Upload
# ==============================

@app.post("/upload-pdf")
async def upload_pdf(thread_id: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()

        summary = ingest_pdf(
            file_bytes=content,
            thread_id=str(thread_id),
            filename=file.filename,
        )

        return {
            "thread_id": thread_id,
            "status": "indexed",
            "summary": summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Threads
# ==============================

@app.get("/threads")
def get_threads():
    return {"threads": retreive_all_threads()}


@app.get("/thread/{thread_id}")
def get_thread(thread_id: str):

    state = app_graph.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    return {
        "thread_id": thread_id,
        "messages": [serialize_message(m) for m in messages],
        "document_metadata": thread_document_metadata(thread_id),
    }