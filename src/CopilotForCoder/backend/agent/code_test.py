from langgraph.prebuilt import create_react_agent

from CopilotForCoder.backend.memory.copilot_state import CopilotState
from CopilotForCoder.backend.tools.python_repl_tool import python_repl_tool, DockerPythonTool


class CodeTestAgent:
    def __init__(self, llm):
        self.code_test_exec = create_react_agent(llm, tools=[python_repl_tool, DockerPythonTool()],
                                                 prompt="你是一个代码测试工程师，调用合适的工具执行完整的单元测试用例，工具选择上要选择容器环境执行环境执行单元测试用例，如果执行工具失败，请返回“执行失败，原因:”，如果执行工具成功，请返回“执行成功”")

    async def run(self, code_test_state: CopilotState):
        print("CodeTestAgent run", code_test_state["code"])
        result = await self.code_test_exec.ainvoke(code_test_state)
        response = result["messages"][-1].content
        if "执行成功" in response:
            exec_result = True
        else:
            exec_result = False
        return {"messages": result["messages"], "exec_result": exec_result}
