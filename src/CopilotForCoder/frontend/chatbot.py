import gradio as gr
import requests

# FastAPI 接口地址
API_URL = "http://127.0.0.1:8000/chat"


# 用于保存多轮对话的历史
def chatbot(user_input, history):
    """
    将用户输入和历史消息打包成OpenAI所需的messages格式，
    发给后端的FastAPI接口，然后拿到回复。
    """
    # Gradio里，history 通常是 [[user1, bot1], [user2, bot2], ...]
    # 我们需要把它转成 OpenAI ChatCompletion 所需的格式:
    # [
    #   {"role": "user", "content": "你好"},
    #   {"role": "assistant", "content": "你好!有什么可以帮你的吗?"},
    #   ...
    # ]
    messages = []
    for pair in history:
        # pair[0] 是用户的内容，pair[1] 是机器人的回复
        messages.append({"role": "user", "content": pair[0]})
        messages.append({"role": "assistant", "content": pair[1]})

    # 现在，把最新一轮的 user_input 加进 messages
    messages.append({"role": "user", "content": user_input})

    # 向 FastAPI 后端发送请求
    try:
        response = requests.post(API_URL, json={"messages": messages})
        result = response.json()
        if "reply" in result:
            reply = result["reply"]
        else:
            reply = "出现错误: " + str(result.get("error", "Unknown Error"))
    except Exception as e:
        reply = f"请求后端出现异常: {str(e)}"

    # 返回给 Gradio
    # Gradio 需要一个 (reply, history) 的元组
    # 其中 reply 是机器人新回复，history 需要把最新对话append进去
    history.append((user_input, reply))
    return history, history


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# ChatGPT-like Chatbot with FastAPI & Gradio")

        # 设置对话展示
        chatbot_ui = gr.Chatbot(label="ChatGPT-like Bot")

        # 输入框
        message = gr.Textbox(label="请输入内容")

        # 隐藏态，存储对话历史
        state = gr.State([])  # 这里保存的是history

        # 点击“发送”后执行chatbot函数
        send_btn = gr.Button("发送")
        send_btn.click(chatbot, inputs=[message, state], outputs=[chatbot_ui, state])

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
