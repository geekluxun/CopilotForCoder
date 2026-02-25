import os

from langchain_ollama import ChatOllama

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = '123'

if "OLLAMA_DEBUG" not in os.environ:
    os.environ["OLLAMA_DEBUG"] = "1"


def getLLM():
    llm = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0,
        # other params...
    )
    return llm


def getEmbedding():
    from langchain_ollama import OllamaEmbeddings
    # 向量的维数如何设置？
    embeddings_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")
    return embeddings_model
