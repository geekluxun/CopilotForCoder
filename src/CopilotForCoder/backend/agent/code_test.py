from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from CopilotForCoder.backend.memory.copilot_state import CopilotState, AgentExecResult
from CopilotForCoder.backend.tools.python_repl_tool import python_repl_tool, DockerPythonTool


class CodeTestAgent:
    def __init__(self, llm):
        self.code_test_exec = create_react_agent(llm, tools=[python_repl_tool, DockerPythonTool()])

    async def run(self, state: CopilotState):
        print("CodeTestAgent start...")
        try:
            request = {
                "messages": [
                    SystemMessage(
                        content="你是一个代码测试工程师，调用合适的工具执行完整的单元测试用例，工具选择上要选择容器环境执行环境执行单元测试用例，如果执行工具失败，请返回“执行失败，原因:”，如果执行工具成功，请返回“执行成功”"),
                    HumanMessage(content=state["full_source_code"])
                ]
            }
            result = await self.code_test_exec.ainvoke(request)
            response = result["messages"][-1].content
            if "执行成功" in response:
                unit_test_result = AgentExecResult.SUCCESS
            else:
                unit_test_result = AgentExecResult.FAILURE
            return {"messages": result["messages"], "unit_test_result": unit_test_result}
        except BaseException as e:
            print(e)
            return {"unit_test_result": AgentExecResult.FAILURE, "task": {"task_response_msg": str(e)}}
