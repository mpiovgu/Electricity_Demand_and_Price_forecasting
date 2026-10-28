"""方式一（NL 反问）智能体 —— 最简服务端。

复用 risk_model.agent.build_agent，用 InMemorySaver 按 thread_id 做多轮记忆。
生产版见 risk_model/app_langfuse.py（含 JWT / Langfuse / Prometheus）。

运行：
    pip install fastapi uvicorn langgraph langchain langchain-openai requests
    export OPENAI_API_KEY=sk-...
    uvicorn agent_server:app --host 0.0.0.0 --port 8000
"""
import os
from typing import List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from risk_model.agent import build_agent

API_KEY = os.getenv("RISK_API_KEY", "risk-demo-secret-key")

app = FastAPI(title="电力价格风控 Agent（方式一 NL）")

# 进程级共享 Agent + 共享内存检查点（多轮记忆都在这里）
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        try:
            _agent = build_agent(checkpointer=InMemorySaver())
        except Exception as e:  # 缺 OPENAI_API_KEY / 模型未训练
            raise HTTPException(status_code=503, detail=f"Agent 未就绪：{e}")
    return _agent


class ChatIn(BaseModel):
    message: str                       # 只传「这一句」新消息，历史由 checkpointer 接
    thread_id: str = "default"         # 同一会话用同一 thread_id


class Msg(BaseModel):
    role: str
    content: str


class ChatOut(BaseModel):
    thread_id: str
    reply: str                         # 模型本轮的自然语言回复（可能是在反问你）
    messages: List[Msg]                # 本轮完整轨迹（含工具调用）


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent/chat", response_model=ChatOut)
def agent_chat(inp: ChatIn, x_api_key: str = Header(...)):
    # 简单服务间鉴权；生产换成 app_langfuse.py 的 JWT 身份
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效 API Key")

    # 一次 HTTP 请求 = 一次 agent.invoke；thread_id 让图恢复历史后继续
    result = get_agent().invoke(
        {"messages": [("user", inp.message)]},
        config={"configurable": {"thread_id": inp.thread_id}},
    )
    msgs = result["messages"]
    return ChatOut(
        thread_id=inp.thread_id,
        reply=msgs[-1].content or "",
        messages=[Msg(role=m.type,content=(m.content), for m in msgs],
    )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
