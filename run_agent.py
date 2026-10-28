"""电力价格风控 ReAct Agent 交互式客户端。

用法：
    export OPENAI_API_KEY=sk-...
    python run_agent.py
退出：输入 exit / quit / q
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve()))

from langgraph.checkpoint.memory import InMemorySaver
from risk_model.agent import build_agent


def main():
    print("正在初始化 ReAct Agent（首次调用会加载风控模型）...")
    agent = build_agent(checkpointer=InMemorySaver())
    # 固定 thread_id 以在 InMemorySaver 中保持会话记忆
    cfg = {"configurable": {"thread_id": "cli-session"}}

    print("电力价格风控助手已就绪。输入问题开始对话（exit 退出）。\n")
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            print("再见。")
            break

        result = agent.invoke({"messages": [("user", user)]}, config=cfg)
        reply = result["messages"][-1].content
        print(f"\n助手> {reply}\n")


if __name__ == "__main__":
    main()
