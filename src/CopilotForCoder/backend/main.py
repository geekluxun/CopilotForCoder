import os

import uvicorn
from fastapi import FastAPI
from langchain.globals import set_debug, set_verbose

from CopilotForCoder.backend.api.endpoints.chat import chat_router

set_debug(True)
set_verbose(True)
app = FastAPI()
app.include_router(chat_router, tags=["chat"])

# 通过mitmproxy代理拦截流量
os.environ["HTTP_PROXY"] = "http://localhost:8080"
# os.environ["HTTPS_PROXY"] = "http://localhost:8080"

if __name__ == "__main__":
    # 启动服务，默认监听 http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
