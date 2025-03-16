from fastapi import APIRouter
from openai import OpenAI
from pydantic import BaseModel

from CopilotForCoder.backend.agent.copilot import ChiefAgent

chat_router = APIRouter()

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)


# 定义请求体数据模型
class ChatRequest(BaseModel):
    messages: list


@chat_router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    接受从前端发送过来的会话上下文(messages)，
    调用OpenAI或者其他大模型，
    返回生成的回复。
    """
    chief = ChiefAgent({"prompt": request.messages[-1]['content']}, user_id="luxun")
    result = await chief.run_code_copilot()
    print("任务结束，返回代码->", result.get('code'))
    return {"reply": result.get('code')}


@chat_router.post("/chat2")
async def chat2_endpoint(request: ChatRequest):
    try:
        # 这里以 ChatCompletion 方式为例
        response = client.chat.completions.create(
            model="qwen2.5-coder:7b",
            messages=request.messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}
