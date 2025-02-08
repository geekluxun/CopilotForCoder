from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from CopilotForCoder.agent.llm_privider import getLLM
from CopilotForCoder.agent.tools.python_repl_tool import python_repl_tool


class Code(BaseModel):
    """生成的代码，包括import语句，代码块"""
    imports: str = Field(description="import语句")
    code: str = Field(description="代码块")


class InputState(MessagesState):
    pass


class OutputState(MessagesState):
    pass


class OverallState(InputState, OutputState):
    exec_result: bool


def test_code_gen_node(state: OverallState) -> Command[Literal["test_code_exec_node"]]:
    result = test_code_gen_agent.invoke(state, debug=True)
    content = result["messages"][-1].content
    return Command(
        update={
            "messages": [HumanMessage(
                content=content, name="test_code_gen_node"
            )]
        },
        goto="test_code_exec_node",
    )


def test_code_exec_node(state: OverallState) -> Command[Literal["supervisor_node", "code_gen_node"]]:
    result = code_test_exec_agent.invoke(state, debug=True)
    content = result["messages"][-1].content
    if "执行成功" in content:
        goto = "supervisor_node"
        exec_result = True
    else:
        goto = "code_gen_node"
        exec_result = False

    return Command(
        update={
            "messages": [HumanMessage(
                content=content, name="test_code_exec_node"
            )],
            "exec_result": exec_result
        },
        goto=goto,
    )


def code_gen_node(state: OverallState) -> Command[Literal["test_code_gen_node"]]:
    result = code_gen_agent.invoke(state['messages'])
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": [HumanMessage(
                content=result.code, name="code_gen_node"
            )],
        },
        goto="test_code_gen_node",
    )


def supervisor_node(state: OverallState) -> Command[Literal["code_gen_node", "__end__"]]:
    exec_result = state.get("exec_result")
    if exec_result:
        goto = END
    else:
        goto = "code_gen_node"
    return Command(goto=goto)


def buildCopilot():
    builder = StateGraph(OverallState, input=InputState, output=OutputState)
    builder.add_edge(START, "supervisor_node")
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("code_gen_node", code_gen_node)
    builder.add_node("test_code_gen_node", test_code_gen_node)
    builder.add_node("test_code_exec_node", test_code_exec_node)
    return builder.compile(checkpointer=memory)


def displayGraph(graph):
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )


if __name__ == "__main__":
    llm = getLLM()
    # 测试代码生成
    test_code_gen_agent = create_react_agent(llm, tools=[],
                                             prompt="你是一个代码测试工程师，根据用户的代码生成单元测试用例，要求返回可执行的结构化的完整代码（不要返回任何无关的提示信息），包括用户代码，单元测试代码，单元测试调用代码”")

    # 测试代码执行
    code_test_exec_agent = create_react_agent(llm, tools=[python_repl_tool],
                                              prompt="你是一个代码测试工程师，调用相应的工具执行完整的单元测试用例，如果执行工具失败，请返回“执行失败，原因:”，如果执行工具成功，请返回“执行成功”")
    # 代码生成
    code_gen_agent = llm.with_structured_output(Code)

    memory = MemorySaver()
    graph = buildCopilot()
    config = {"configurable": {"thread_id": "1"}}
    events = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    "帮我用python写一个函数，实现两个数相乘. 包括必要的import，不需要无关的测试代码"
                )
            ],
        },
        config=config
    )
    messages = graph.get_state(config).values["messages"]
    messages = [m.content for m in messages]
    print(messages)
