import traceback

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from CopilotForCoder.backend.memory.copilot_state import CopilotState, AgentExecResult


class Code(BaseModel):
    """生成的代码，包括import语句，源代码块，单元测试代码块"""
    code: str = Field(description="代码")


class CodeRefactorAgent:
    def __init__(self, llm):
        self.code_refactor = llm.with_structured_output(Code)

    async def run(self, state: CopilotState, config: RunnableConfig):
        print("In CodeRefactorAgent config is: ", config["configurable"]["user_id"])
        print("CodeRefactorAgent run", state["task"])

        try:
            messages = [
                SystemMessage(
                    content=f"""你是一个代码优化专家.根据用户的要求对以下代码完成优化，代码为:{state["full_source_code"]}
            要求1.包含完整的import语句
            2.包含单元测试代码
            3.如果有需要安装的pip包在第一行以“  # REQUIRES pacakgeName”形式返回"""
                ),
                HumanMessage(content=state["human_review_message"])
            ]
            result = self.code_refactor.invoke(messages)
            return {
                "full_source_code": result.code,
                "code_refactor_result": AgentExecResult.SUCCESS
            }
        except Exception:
            traceback_info = traceback.format_exc()
            print(traceback_info)
            return {
                "code_refactor_result": AgentExecResult.FAILURE,
                "task": {"task_response_msg": traceback_info}
            }
