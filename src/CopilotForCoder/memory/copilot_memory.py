import asyncio

import aiosqlite
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.postgres import PostgresStore
from psycopg import Connection


async def query_memory(thread_id):
    memory_saver = AsyncSqliteSaver(conn=aiosqlite.connect("/Users/luxun/workspace/ai/mine/project/open/CopilotForCoder/temp/sqllite/copilotForCode.db"))
    c = await memory_saver.aget(config={"configurable": {"thread_id": thread_id}})
    print(c)
    return c


async def main():
    r = await query_memory("1742045653")
    print(r)


def test_store():
    conn = Connection.connect("postgresql://postgres@localhost:5432/copilotforcoder")
    conn.autocommit = True
    store = PostgresStore(
        conn=conn,
        index={
            "dims": 1024,
            "embed": init_embeddings("quentinz/bge-large-zh-v1.5", provider="ollama"),
            # specify which fields to embed. Default is the whole serialized value
            "fields": ["text"],
        },
    )
    store.setup()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
    print(item)


if __name__ == "__main__":
    asyncio.run(main())
    test_store()
