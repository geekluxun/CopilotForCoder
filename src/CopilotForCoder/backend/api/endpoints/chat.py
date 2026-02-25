import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel, Field

from CopilotForCoder.backend.agent.copilot import ChiefAgent
from CopilotForCoder.backend.util.task import generate_task_id

chat_router = APIRouter()

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)


# 定义请求体数据模型
class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        # alias="sessionId",  # 支持前端使用camelCase
        description="会话ID（首次请求不需要传）"
    )
    messages: list = Field(..., min_length=1, description="用户输入的消息内容")


sessions: Dict[str, ChiefAgent] = {}


@chat_router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 获取或创建会话
        if not request.session_id or request.session_id not in sessions:
            # 新会话初始化
            session_id = str(uuid.uuid4())
            chief = ChiefAgent(
                {"prompt": request.messages[-1]['content']},
                user_id="luxun",
            )
            sessions[session_id] = chief
            # 执行任务
            await chief.run_code_copilot(task_id=generate_task_id())

        else:
            human_review_message = request.messages[-1]['content']
            # 恢复已有会话
            session_id = request.session_id
            chief = sessions[session_id]
            config = chief.graph.config
            await chief.graph.ainvoke(
                Command(update={"human_review_message": human_review_message}, resume=human_review_message),
                config=config
            )

        # 所有的任务结果状态统一从state获取
        agent_state = await chief.graph.aget_state(config=chief.graph.config)
        if agent_state.tasks:
            reply = agent_state.tasks[0].interrupts[0].value["question"]
        else:
            reply = agent_state.values["full_source_code"]

        return {
            "session_id": session_id,
            "reply": reply,
        }

    except KeyError as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"无效的会话ID: {request.session_id}")
    except Exception as e:
        sessions.pop(session_id, None)  # 清理异常会话
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


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
