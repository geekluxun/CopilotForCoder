import time


def generate_task_id():
    return int(time.time())


if __name__ == "__main__":
    import textwrap

    text = """
        这是第一行示例。
            这是第二行示例。
        这是第三行示例。
    """

    # 去除文本的公共缩进
    dedented_text = textwrap.dedent(text)
    print(dedented_text)
