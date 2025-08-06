from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI()

# === КОНФИГ ===
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "psychologist_history")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print("QDRANT_URL:", QDRANT_URL)
print("QDRANT_API_KEY:", "SET" if QDRANT_API_KEY else "MISSING")
print("OPENAI_API_KEY:", "SET" if OPENAI_API_KEY else "MISSING")

# === КЛИЕНТЫ ===
client = None
openai = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    if QDRANT_URL and QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    print(f"Qdrant client init error: {e}")

try:
    import openai as openai_module
    if OPENAI_API_KEY:
        openai = openai_module
        openai.api_key = OPENAI_API_KEY
except Exception as e:
    print(f"OpenAI client init error: {e}")

class MessageIn(BaseModel):
    user_id: str
    role: str
    text: str
    created_at: str = None  # ISO format
    task_id: int = None

def get_embedding(text: str):
    if not openai:
        raise RuntimeError("OpenAI не инициализирован!")
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

@app.post("/add_to_qdrant")
async def add_to_qdrant(message: MessageIn):
    if not client:
        return {"status": "error", "msg": "Qdrant не инициализирован!"}
    if not openai:
        return {"status": "error", "msg": "OpenAI не инициализирован!"}
    try:
        embedding = get_embedding(message.text)
    except Exception as e:
        return {"status": "error", "msg": f"OpenAI error: {str(e)}"}
    payload = {
        "user_id": message.user_id,
        "role": message.role,
        "text": message.text,
        "created_at": message.created_at or datetime.utcnow().isoformat(),
        "task_id": message.task_id
    }
    try:
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
    except Exception as e:
        return {"status": "error", "msg": f"Qdrant error: {str(e)}"}
    return {"status": "ok", "msg": "Message added to Qdrant"}

@app.get("/")
def root():
    return {"status": "ok"}
