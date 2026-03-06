import os
from openai import OpenAI
from src.config import Config
import chromadb
from sentence_transformers import CrossEncoder


open_ai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
ollama_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
chroma_client = chromadb.PersistentClient(path=Config.VECTOR_STORE_PATH)
rerank_model =  CrossEncoder(Config.RERANK_MODEL)


