from __future__ import annotations
import os
import tempfile
from dotenv import load_dotenv
from typing import Annotated, Any, Dict, Optional, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core import tools
import sqlite3
import requests

load_dotenv()

# ------------------------
# LLM (Groq)
# ------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None



def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


#=================Tools===========================

search_tool = DuckDuckGoSearchRun()


@tool
def get_crypto_price(symbol: str) -> str:
    """
    Fetch latest crypto price for a given symbol (e.g. 'ETH', 'BTC')
    using Binance Futures API (fapi).
    Returns a natural language string.
    """
    # Convert symbol to Binance format
    binance_symbol = f"{symbol.upper()}USDT"

    # Fetch 1-minute kline
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": binance_symbol,
        "interval": "1m",
        "limit": 1  # Only latest candle
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()
    if not data:
        return f"Sorry, I could not fetch the price for {symbol.upper()}."

    latest_candle = data[0]
    close_price = float(latest_candle[4])

    # Format as natural sentence
    return f"The current price of {symbol.upper()} is ${close_price:,.2f}."

#================RAG Tool=============================

@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_crypto_price,rag_tool]
llm_with_tools = llm.bind_tools(tools)


# Define state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


#===================tools node=======================

tool_node = ToolNode(tools)

#chat function for node graph
def chat_node(state: AgentState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")


    system_message = SystemMessage(
    content=(
        "You are a helpful assistant.\n\n"
        "IMPORTANT TOOL RULES:\n"
        "- When calling rag_tool, ALWAYS pass both:\n"
        "  1. query (string)\n"
        "  2. thread_id (string)\n\n"
        f"The thread_id for this conversation is: {thread_id}\n\n"
        "If no document exists, ask the user to upload a PDF."
    )
)

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}



#======== Create database =========

conn = sqlite3.connect(database = "chatbot_db", check_same_thread=False)


# ------------------------
# Build Graph
# ------------------------
checkpointer = SqliteSaver(conn = conn)
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chat_node", chat_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", tools_condition)
workflow.add_edge("tools", "chat_node")

# Compile
app_graph = workflow.compile(checkpointer=checkpointer)

def retreive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]['thread_id'])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})