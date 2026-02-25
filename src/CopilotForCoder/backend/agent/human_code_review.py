from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from CopilotForCoder.backend.memory.copilot_state import CopilotState, ReviewResult


class Result(BaseModel):
    """代码Review结果"""
    reviewResult: str = Field(description="代码Review结果是否通过")


class HumanCodeReview:
    def __init__(self, llm):
        self.human_code_review = llm

    async def run(self, state: CopilotState):
        print("---HumanCodeReview start---")
        question = f"""请review生成的代码，如果有问题，请反馈给我，代码为:\n{state["full_source_code"]}"""
        human_review_message = interrupt({"question": question})
        print("用户的review结果为:", human_review_message)

        try:
            messages = [
                SystemMessage(
                    content="你是一个情感判断专家，请根据上下文和用户对你的代码Review的结果判断，如果Review通过，请返回'通过',如果用户对你写的代码不满意或者需要重构或有疑问的，请返回'不通过'"),
                HumanMessage(content="我对你的代码Review结果为:" + human_review_message)
            ]



            if state["messages"]:
                state["messages"].extend(messages)

            result = await self.human_code_review.ainvoke(state["messages"])
            state["human_review_results"] = ReviewResult.PASS if result.content == "通过" else ReviewResult.NO_PASS
            state["human_review_message"] = human_review_message
            return state
        except Exception as e:
            print(e)
            return {"human_review_results": ReviewResult.NO_PASS, "human_review_message": human_review_message, "task": {"task_response_msg": str(e)}}
