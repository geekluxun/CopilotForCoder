from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from CopilotForCoder.backend.memory.copilot_state import CopilotState, AgentExecResult


class Code(BaseModel):
    """生成的代码，包括import语句，源代码块，单元测试代码块"""
    code: str = Field(description="代码")


class CodeGenAgent:
    def __init__(self, llm):
        self.code_gen = llm.with_structured_output(Code)

    async def run(self, state: CopilotState, config: RunnableConfig):
        print("In CodeGenAgent config is: ", config["configurable"]["user_id"])
        print("CodeGenAgent run", state["task"])

        try:
            messages = [
                SystemMessage(
                    content="你是一个代码助手。根据用户的需求写一个代码,要求1.包含完整的import语句, 2.包含单元测试代码, 3.如果有需要安装的pip包在第一行以“#REQUIRES pacakgeName”形式返回"),
                HumanMessage(content=state["task"]["prompt"])
            ]
            # 增加到代码生成上下文中
            if state["messages"]:
                messages.extend(state["messages"])
            result = self.code_gen.invoke(messages)
            return {"full_source_code": result.code, "code_gen_result": AgentExecResult.SUCCESS}
        except BaseException as e:
            print(e)
            return {"code_gen_result": AgentExecResult.FAILURE, "task": {"task_response_msg": str(e)}}
