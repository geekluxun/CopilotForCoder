from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph


def display_graph(graph: StateGraph, jupyter=False):
    if jupyter:
        display(
            Image(
                graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )
    else:
        img_path = "graph.png"
        graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API, output_file_path=img_path)


if __name__ == "__main__":
    display_graph()
