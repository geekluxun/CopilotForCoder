from typing import Tuple

import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chat"


def chatbot(user_input: str, history: list, session_id: str) -> Tuple[list, list, str]:
    """
    处理聊天逻辑，维护session_id
    """
    # 构造请求体
    payload = {
        "messages": [{"role": "user", "content": user_input}]
    }

    # 如果存在session_id则添加
    if session_id:
        payload["session_id"] = session_id

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        # 处理正常响应
        if response.status_code == 200:
            new_session_id = result.get("session_id", "")
            reply = result.get("reply", "收到空回复")

            # 更新session_id（如果是新会话）
            if not session_id and new_session_id:
                session_id = new_session_id

            # 追加到历史记录
            history.append((user_input, reply))

            return history, history, session_id

        # 处理错误响应
        error_msg = result.get("detail", "未知错误")
        reply = f"请求失败: {error_msg}"
        history.append((user_input, reply))
        return history, history, session_id

    except Exception as e:
        error_reply = f"请求异常: {str(e)}"
        history.append((user_input, error_reply))
        return history, history, session_id


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 代码生成助手")

        # 隐藏状态存储
        session_state = gr.State("")  # 保存session_id
        history_state = gr.State([])  # 保存对话历史

        # 聊天界面
        chatbot_ui = gr.Chatbot(label="对话记录", height=500)
        msg_input = gr.Textbox(label="输入消息", placeholder="请输入需求或反馈...")

        # 控制按钮
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            new_chat_btn = gr.Button("新对话")

        # 事件处理
        def new_chat():
            """重置会话"""
            return [], [], ""

        send_btn.click(
            chatbot,
            inputs=[msg_input, history_state, session_state],
            outputs=[chatbot_ui, history_state, session_state]
        )

        new_chat_btn.click(
            new_chat,
            outputs=[chatbot_ui, history_state, session_state]
        )

        msg_input.submit(
            chatbot,
            inputs=[msg_input, history_state, session_state],
            outputs=[chatbot_ui, history_state, session_state]
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
