import re
from typing import Annotated, List

import docker
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from CopilotForCoder.backend.util.llm_privider import getLLM

repl = PythonREPL()
# 允许在docker中安装的
white_package = ["numpy", "pandas", "scikit-learn"]


@tool
def python_repl_tool(
        fullCode: Annotated[str, "执行完整的单元测试."],
):
    """执行完整的单元测试，代码的组织顺序为用户代码，单元测试代码，执行单元测试代码"""
    try:
        result = repl.run(fullCode)
        if result is None or result == "":
            # result_str = f"Successfully executed:\n```python\n{fullCode}\n```\nStdout: {result}"
            result_str = f"Successfully executed"
        else:
            result_str = f"Failed to execute. Error:\n```python\n{fullCode}\n```\nStderr: {result}"
    except BaseException as e:
        result_str = f"Failed to execute. Error: {repr(e)}"
    return result_str


class DockerPythonTool(BaseTool):
    name: str = "docker_python_tool"
    description: str = "在Docker容器中执行完整的单元测试"

    def _parse_requirements(self, code: str) -> List[str]:
        """从代码中解析#REQUIRES声明的依赖"""
        pattern = r"^\s*#REQUIRES\s+(.+?)\s*$"
        matches = re.findall(pattern, code, flags=re.MULTILINE)
        if not matches:
            return []
        return [pkg.strip() for pkg in matches[0].split() if pkg.strip()]

    def _validate_packages(self, packages: List[str]) -> List[str]:
        """验证包是否在白名单中"""
        valid_pkgs = []
        for pkg in packages:
            base_pkg = pkg.split("==")[0].split("[")[0]  # 处理带版本或扩展的包名
            if base_pkg in white_package:
                valid_pkgs.append(pkg)
            else:
                raise ValueError(f"包 {base_pkg} 不在白名单中")
        return valid_pkgs

    def _run(self, fullCode: Annotated[str, "执行完整的单元测试."]) -> str:
        try:
            client = docker.from_env()
            # 1. 解析并验证依赖
            required_packages = self._parse_requirements(fullCode)
            valid_packages = self._validate_packages(required_packages)

            # 2. 转义代码中的特殊字符
            escaped_code = fullCode.replace('"', '\\"').replace('$', '\\$')

            # 3. 构建动态安装命令
            install_cmd = ""
            if valid_packages:
                install_cmd = f"pip install {' '.join(valid_packages)} && "

            # 4. 执行完整命令（安装依赖 + 运行代码）
            full_command = [
                "/bin/sh",
                "-c",
                f"{install_cmd} python -c \"{escaped_code}\""
            ]

            # 5. 启动容器（临时开启网络）
            container = client.containers.run(
                image="python:3.9-slim",
                command=full_command,
                mem_limit="512m",
                network_mode="host" if valid_packages else "none",  # 按需开网络
                detach=True,
            )
            result = container.wait(timeout=30)
            logs = container.logs().decode()
        except BaseException as e:
            print("docker执行失败", e)
            logs = "docker执行失败" + e
        finally:
            container.remove(force=True)
        return logs


if __name__ == "__main__":
    llm = getLLM()
    tools = [DockerPythonTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    result = agent.run("用Python计算1到10的和，输出结果。")
    print(result)  # 输出：55
