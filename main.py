from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct  # оставляем как есть
from datetime import datetime, timezone
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
    question: str | None = None
    answer: str | None = None
    created_at: str | None = None  # ISO format
    task_id: int | None = None

# === [ДОБАВЛЕНО] Модель для логирования напоминаний ===
class ReminderLog(BaseModel):
    user_id: str
    task_id: int
    reminder_text: str
    bot_message_id: str | None = None
    created_at: str | None = None  # ISO8601; если нет — проставим сами

def get_embedding(text: str):
    source = (text or " ")  # защита от пустых строк
    return openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=source
    ).data[0].embedding

# === [ДОБАВЛЕНО] Вспомогательная функция парсинга ISO8601 ===
def _parse_iso(dt_str: str | None) -> datetime:
    if not dt_str:
        return datetime.now(timezone.utc)
    try:
        # поддержка 'Z'
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        # делаем tz-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)

@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    embedding_text = (message.question or "") + " " + (message.answer or "")
    embedding = get_embedding(embedding_text)

    dt = _parse_iso(message.created_at)
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
                vector=embedding,  # НЕ меняем твою схему (без именованного вектора)
                payload=payload
            )
        ]
    )
    return {"status": "ok", "msg": "Structured message added to Qdrant"}

# === [ДОБАВЛЕНО] Новый эндпоинт для записи 24h-напоминания ===
@app.post("/log_reminder")
async def log_reminder(body: ReminderLog):
    dt = _parse_iso(body.created_at)
    created_at_str = dt.isoformat()
    created_at_ts = dt.timestamp()

    payload = {
        "user_id": str(body.user_id),
        "role": "assistant",
        "question": None,
        "answer": body.reminder_text,
        "created_at": created_at_str,
        "created_at_ts": created_at_ts,
        "task_id": body.task_id,
        "event": "reminder",
        "reminder_type": "24h",
        "bot_message_id": body.bot_message_id,
        "is_reminder": True
    }

    # Используем тот же формат вектора, что и в основном эндпоинте
    embedding = get_embedding(body.reminder_text)
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
    return {"status": "ok", "msg": "Reminder logged to Qdrant"}

@app.get("/")
def root():
    return {"status": "ok"}
