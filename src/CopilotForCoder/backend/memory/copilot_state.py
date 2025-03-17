import operator
from enum import Enum
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel


class TaskExecResultCode(Enum):
    SUCCESS = "成功"
    FAILURE = "失败"


class AgentExecResult(Enum):
    SUCCESS = "成功"
    FAILURE = "失败"


class ReviewResult(Enum):
    PASS = "通过"
    NO_PASS = "未通过"


class Task(BaseModel):
    task_id: str
    task_exec_code: TaskExecResultCode
    task_exec_msg: str


class CopilotState(MessagesState):
    # 每次更新只增加messages
    messages: Annotated[list[AnyMessage], add_messages]
    task: Task | None
    full_source_code: str | None
    code_gen_result: AgentExecResult | None
    unit_test_result: AgentExecResult | None
    code_refactor_result: AgentExecResult | None
    human_review_results: Annotated[list[ReviewResult] | None, operator]
    human_review_message: Annotated[list[str] | None, operator]
