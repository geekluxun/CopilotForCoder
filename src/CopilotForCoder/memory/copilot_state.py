from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages


class CopilotState(MessagesState):
    # 每次更新只增加messages
    messages: Annotated[list[AnyMessage], add_messages]
    task: dict
    code: str
    exec_result: bool
