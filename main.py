from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
import openai
import os

# === КОНФИГ ===
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "psychologist_history")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI()

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai.api_key = OPENAI_API_KEY

class MessageIn(BaseModel):
    user_id: str
    role: str
    text: str
    created_at: str = None  # ISO format
    task_id: int = None

def get_embedding(text: str):
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    embedding = get_embedding(message.text)
    payload = {
        "user_id": message.user_id,
        "role": message.role,
        "text": message.text,
        "created_at": message.created_at or datetime.utcnow().isoformat(),
        "task_id": message.task_id
    }
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=None,
                vector=embedding,
                payload=payload
            )
        ]
    )
    return {"status": "ok", "msg": "Message added to Qdrant"}

@app.get("/")
def root():
    return {"status": "ok"}

