import redis
from typing import Optional
from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from src.graph.models.llm import get_llm


REDIS_HOST = "localhost"
REDIS_PORT = 6379

# get local redis client
def get_local_redis():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=None,
        ssl=False,
        decode_responses=True,
    )

# generate redis key for conversation memory
def get_memory_key(customer_id_number: str, session_id: str) -> str:
    return f"conversation:{customer_id_number}:{session_id}"


# create conversation summary buffer memory
def create_memory(
    customer_id_number: str,
    session_id: str,
    llm: Optional[any] = None,
    max_token_limit: int = 2000) -> ConversationSummaryBufferMemory:
    if llm is None:
        llm = get_llm(
            role="generic",
            use_langchain=True,
        )

    # Create Redis-backed chat message history
    redis_key = get_memory_key(customer_id_number, session_id)
    # RedisChatMessageHistory accepts url parameter
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    message_history = RedisChatMessageHistory(
        url=redis_url,
        session_id=redis_key,
    )

    # Create memory with summary buffer
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=max_token_limit,
        return_messages=True,
    )

    return memory

# clear conversation memory for a specific session
def clear_memory(customer_id_number: str, session_id: str):
    redis_key = get_memory_key(customer_id_number, session_id)
    r = get_local_redis()
    r.delete(redis_key)

