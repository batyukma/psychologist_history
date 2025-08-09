from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
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
    role: str                           # "user" | "assistant" | др.
    question: Optional[str] = None
    answer: Optional[str] = None
    created_at: Optional[str] = None
    task_id: Optional[int] = None
    event: Optional[str] = None         # "task_update" | "task_update_reply" | "button_click" | "message" ...
    meta: Optional[Dict[str, Any]] = None
    interaction: Optional[Dict[str, Any]] = None

class ReminderLog(BaseModel):
    user_id: str
    task_id: int
    reminder_text: str
    bot_message_id: Optional[str] = None
    created_at: Optional[str] = None

# === Утилиты ===
def _parse_iso(dt_str: Optional[str]) -> datetime:
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

def get_embedding(text: str) -> List[float]:
    """Создание эмбеддинга для текста."""
    return openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=(text or " ")
    ).data[0].embedding

def _upsert_to_qdrant(vector: List[float], payload: dict):
    """Упрощённая вставка в Qdrant."""
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(id=str(uuid4()), vector=vector, payload=payload)]
    )

# === Эндпоинты ===
@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    dt = _parse_iso(message.created_at)

    # Текст, по которому строим эмбеддинг (и сохраняем в payload для отладки):
    embedding_text = ((message.question or "") + " " + (message.answer or "")).strip() or " "

    # Умный дефолт события — сохраняет обратную совместимость:
    inferred_event = (
        message.event
        or ("task_update" if (message.task_id is not None and message.role == "assistant" and message.question is None) else "message")
    )

    payload = {
        "user_id": str(message.user_id),
        "role": message.role,
        "question": message.question,
        "answer": message.answer,
        "text": embedding_text,
        "created_at": dt.isoformat(),
        "created_at_ts": dt.timestamp(),
        "task_id": message.task_id,
        "event": inferred_event,
        "meta": message.meta,
        "interaction": message.interaction,
    }

    _upsert_to_qdrant(get_embedding(embedding_text), payload)
    return {"status": "ok", "payload": payload}

@app.post("/log_reminder")
async def log_reminder(body: Union[ReminderLog, MessageIn] = Body(...)):
    """
    Принимает как ReminderLog (чистый формат), так и MessageIn-подобный JSON
    из текущих нод n8n. Автоматически приводит данные к нужному виду.
    """
    # Определяем формат
    if isinstance(body, ReminderLog):
        dt = _parse_iso(body.created_at)
        reminder_text = body.reminder_text or " "
        task_id = body.task_id
        bot_message_id = body.bot_message_id
    else:
        # Пришёл MessageIn
        msg: MessageIn = body  # type: ignore
        dt = _parse_iso(msg.created_at)
        # reminder_text: answer > meta.task_text > question
        reminder_text = (
            (msg.answer or "").strip()
            or str((msg.meta or {}).get("task_text") or "").strip()
            or (msg.question or "").strip()
            or " "
        )
        # task_id может быть не int
        try:
            task_id = int(msg.task_id) if msg.task_id is not None else None
        except Exception:
            task_id = None
        bot_message_id = None
        if msg.meta and "bot_message_id" in msg.meta:
            bot_message_id = str(msg.meta["bot_message_id"])

    # Формируем payload
    payload = {
        "user_id": str(body.user_id),
        "role": "assistant",
        "question": None,
        "answer": reminder_text,
        "text": reminder_text,
        "created_at": dt.isoformat(),
        "created_at_ts": dt.timestamp(),
        "task_id": task_id,
        "event": "reminder",
        "reminder_type": "24h",
        "bot_message_id": bot_message_id,
        "is_reminder": True
    }

    _upsert_to_qdrant(get_embedding(reminder_text), payload)
    return {"status": "ok", "payload": payload}

@app.get("/")
def root():
    return {"status": "ok"}
