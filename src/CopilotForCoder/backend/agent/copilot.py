import asyncio

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from CopilotForCoder.backend.agent.code_gen import CodeGenAgent
from CopilotForCoder.backend.agent.code_refactor import CodeRefactorAgent
from CopilotForCoder.backend.agent.code_test import CodeTestAgent
from CopilotForCoder.backend.agent.human_code_review import HumanCodeReview
from CopilotForCoder.backend.memory.copilot_state import CopilotState, ReviewResult
from CopilotForCoder.backend.util.display_graph import display_graph
from CopilotForCoder.backend.util.llm_privider import getLLM
from CopilotForCoder.backend.util.task import generate_task_id


class ChiefAgent:
    def __init__(self, task: dict, user_id: str):
        self.task = task
        self.llm = getLLM()
        self.user_id = user_id
        memory = AsyncSqliteSaver(conn=aiosqlite.connect("/Users/luxun/workspace/ai/mine/project/open/CopilotForCoder/temp/sqllite/copilotForCode.db"))
        research_team = self._init_research_team()
        self.graph = research_team.compile(checkpointer=memory)
        display_graph(self.graph)

    def _init_research_team(self):
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def run_code_copilot(self, task_id: str):
        self.task_id = task_id
        config = {
            "configurable": {
                "thread_id": self.task_id,
                "user_id": self.user_id
            }
        }
        self.graph.config = config

        result = await self.graph.ainvoke({"task": self.task}, config=config)
        return result

    def _initialize_agents(self):
        return {
            "codeGen": CodeGenAgent(llm=self.llm),
            "codeTest": CodeTestAgent(llm=self.llm),
            "humanCodeReview": HumanCodeReview(llm=self.llm),
            "codeRefactor": CodeRefactorAgent(llm=self.llm)
        }

    def _create_workflow(self, agents):
        workflow = StateGraph(CopilotState)

        workflow.add_node("codeGen", agents["codeGen"].run)
        workflow.add_node("codeTest", agents["codeTest"].run)
        workflow.add_node("humanCodeReview", agents["humanCodeReview"].run)
        workflow.add_node("codeRefactor", agents["codeRefactor"].run)
        self._add_workflow_edges(workflow)
        return workflow

    def _add_workflow_edges(self, workflow):
        def decide(state: CopilotState):
            if state["human_review_results"] == ReviewResult.PASS:
                return "end"
            else:
                return "codeRefactor"

        workflow.set_entry_point("codeGen")
        workflow.add_edge('codeGen', 'codeTest')
        workflow.add_edge("codeTest", "humanCodeReview")
        workflow.add_conditional_edges('humanCodeReview', decide, {
            "end": END,
            "codeRefactor": "codeRefactor"
        })
        workflow.add_edge("codeRefactor", "codeTest")


async def main():
    chief = ChiefAgent({"prompt": "帮我用python写一个函数，实现两个数相乘。"}, user_id="luxun")
    result = await chief.run_code_copilot(task_id=generate_task_id())
    print(result)
    # c = chief.graph.checkpointer.get(config=chief.config)
    # print(c)
    states = await chief.graph.aget_state(config=chief.graph.config)
    print(states)
    history = chief.graph.get_state_history(config=chief.graph.config)
    print(history)


if __name__ == "__main__":
    asyncio.run(main())
