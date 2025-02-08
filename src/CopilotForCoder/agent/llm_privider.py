import os

from langchain_ollama import ChatOllama

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = '123'

if "OLLAMA_DEBUG" not in os.environ:
    os.environ["OLLAMA_DEBUG"] = "1"


def getLLM():
    # llm = ChatOpenAI(model_name="gpt-4-turbo",
    #                  base_url="http://127.0.0.1:8080/")
    llm = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0,
        # other params...
    )
    return llm
