from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from CopilotForCoder.memory.copilot_state import CopilotState


class Code(BaseModel):
    """生成的代码，包括import语句，源代码块，单元测试代码块"""
    code: str = Field(description="代码")


class CodeGenAgent:
    def __init__(self, llm):
        self.code_gen = llm.with_structured_output(Code)

    async def run(self, state: CopilotState, config: RunnableConfig):
        print("In CodeGenAgent config is: ", config["configurable"]["user_id"])

        print("CodeGenAgent run", state["task"])

        messages = [
            SystemMessage(content="你是一个代码助手。根据用户的需求写一个代码(需要包含必要的import语句)，同时写出单元测试代码"),
            HumanMessage(content=state["task"]["prompt"])
        ]
        result = self.code_gen.invoke(messages)

        return {"code": result.code, "messages": [HumanMessage(
            content=result.code, name="code_gen_node"
        )]}
