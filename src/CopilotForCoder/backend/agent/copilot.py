import asyncio
import time

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from CopilotForCoder.backend.agent.code_gen import CodeGenAgent
from CopilotForCoder.backend.agent.code_test import CodeTestAgent
from CopilotForCoder.backend.memory.copilot_state import CopilotState
from CopilotForCoder.backend.util.display_graph import display_graph
from CopilotForCoder.backend.util.llm_privider import getLLM


class ChiefAgent:
    def __init__(self, task: dict, user_id: str):
        self.task = task
        self.task_id = self._generate_task_id()
        self.llm = getLLM()
        self.user_id = user_id
        memory = AsyncSqliteSaver(conn=aiosqlite.connect("/Users/luxun/workspace/ai/mine/project/open/CopilotForCoder/temp/sqllite/copilotForCode.db"))
        research_team = self.init_research_team()
        self.graph = research_team.compile(checkpointer=memory)
        display_graph(self.graph)

    def init_research_team(self):
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def run_code_copilot(self):
        self.config = {
            "configurable": {
                "thread_id": self.task_id,
                "user_id": self.user_id
            }
        }

        result = await self.graph.ainvoke({"task": self.task}, config=self.config)
        return result

    def _generate_task_id(self):
        return int(time.time())

    def _initialize_agents(self):
        return {
            "codeGen": CodeGenAgent(llm=self.llm),
            "codeTest": CodeTestAgent(llm=self.llm),
        }

    def _create_workflow(self, agents):
        workflow = StateGraph(CopilotState)

        workflow.add_node("codeGen", agents["codeGen"].run)
        workflow.add_node("codeTest", agents["codeTest"].run)

        self._add_workflow_edges(workflow)

        return workflow

    def _add_workflow_edges(self, workflow):
        workflow.set_entry_point("codeGen")
        workflow.add_edge('codeGen', 'codeTest')
        workflow.add_edge('codeTest', END)


async def main():
    chief = ChiefAgent({"prompt": "帮我用python写一个函数，实现两个数相乘。"}, user_id="luxun")
    result = await chief.run_code_copilot()
    print(result)
    # c = chief.graph.checkpointer.get(config=chief.config)
    # print(c)
    states = await chief.graph.aget_state(config=chief.config)
    print(states)
    history = chief.graph.get_state_history(config=chief.config)
    print(history)


if __name__ == "__main__":
    asyncio.run(main())
