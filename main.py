from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
import openai
import os
from uuid import uuid4

app = FastAPI()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "psychologist_history")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai.api_key = OPENAI_API_KEY

class MessageIn(BaseModel):
    user_id: str
    role: str
    question: str = None
    answer: str = None
    created_at: str = None  # ISO format
    task_id: int = None

def get_embedding(text: str):
    # Для embedding можно брать question+answer, или только question/answer (на твой выбор)
    source = (text or "")  # чтобы не было ошибки, если оба None
    return openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=source
    ).data[0].embedding

@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    # Формируем текст для embedding (например, question+answer для максимального контекста)
    embedding_text = (message.question or "") + " " + (message.answer or "")
    embedding = get_embedding(embedding_text)

    # Обработка времени: created_at_ts всегда float, даже если передали свою дату
    if message.created_at:
        try:
            dt = datetime.fromisoformat(message.created_at)
        except Exception:
            dt = datetime.utcnow()
    else:
        dt = datetime.utcnow()
    created_at_str = dt.isoformat()
    created_at_ts = dt.timestamp()

    payload = {
        "user_id": str(message.user_id),
        "role": message.role,
        "question": message.question,
        "answer": message.answer,
        "created_at": created_at_str,
        "created_at_ts": created_at_ts,
        "task_id": message.task_id
    }
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload
            )
        ]
    )
    return {"status": "ok", "msg": "Structured message added to Qdrant"}

@app.get("/")
def root():
    return {"status": "ok"}
