from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()


@tool
def python_repl_tool(
        fullCode: Annotated[str, "执行完整的单元测试."],
):
    """执行完整的单元测试，代码的组织顺序为用户代码，单元测试代码，执行单元测试代码"""
    try:
        result = repl.run(fullCode)
        if result is None or result == "":
            result_str = f"Successfully executed:\n```python\n{fullCode}\n```\nStdout: {result}"
        else:
            result_str = f"Failed to execute. Error:\n```python\n{fullCode}\n```\nStderr: {result}"
    except BaseException as e:
        result_str = f"Failed to execute. Error: {repr(e)}"
    return result_str
