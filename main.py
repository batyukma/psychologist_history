from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime, timezone
import openai
import os
from uuid import uuid4

app = FastAPI()

# === Конфигурация ===
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "psychologist_history")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai.api_key = OPENAI_API_KEY

# === Модели ===
class MessageIn(BaseModel):
    user_id: str
    role: str
    question: str | None = None
    answer: str | None = None
    created_at: str | None = None
    task_id: int | None = None

class ReminderLog(BaseModel):
    user_id: str
    task_id: int
    reminder_text: str
    bot_message_id: str | None = None
    created_at: str | None = None

# === Утилиты ===
def _parse_iso(dt_str: str | None) -> datetime:
    """Парсинг ISO8601 с поддержкой 'Z' и без таймзоны."""
    if not dt_str:
        return datetime.now(timezone.utc)
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)

def get_embedding(text: str):
    """Создание эмбеддинга для текста."""
    return openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=(text or " ")
    ).data[0].embedding

def _upsert_to_qdrant(vector: list[float], payload: dict):
    """Упрощённая вставка в Qdrant."""
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=str(uuid4()), vector=vector, payload=payload)]
    )

# === Эндпоинты ===
@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    dt = _parse_iso(message.created_at)
    payload = {
        "user_id": str(message.user_id),
        "role": message.role,
        "question": message.question,
        "answer": message.answer,
        "created_at": dt.isoformat(),
        "created_at_ts": dt.timestamp(),
        "task_id": message.task_id
    }
    embedding_text = (message.question or "") + " " + (message.answer or "")
    _upsert_to_qdrant(get_embedding(embedding_text), payload)
    return {"status": "ok", "msg": "Structured message added to Qdrant"}

@app.post("/log_reminder")
async def log_reminder(body: ReminderLog):
    dt = _parse_iso(body.created_at)
    payload = {
        "user_id": str(body.user_id),
        "role": "assistant",
        "question": None,
        "answer": body.reminder_text,
        "created_at": dt.isoformat(),
        "created_at_ts": dt.timestamp(),
        "task_id": body.task_id,
        "event": "reminder",
        "reminder_type": "24h",
        "bot_message_id": body.bot_message_id,
        "is_reminder": True
    }
    _upsert_to_qdrant(get_embedding(body.reminder_text), payload)
    return {"status": "ok", "msg": "Reminder logged to Qdrant"}

@app.get("/")
def root():
    return {"status": "ok"}
