"""方式一（NL 反问）智能体 —— HTTP 客户端。

交互式：每行用户输入 = 一次 POST /agent/chat（携带同一 thread_id）。
服务端用 checkpointer 记住历史，所以「模型反问 → 用户补信息 → 模型继续」
完全由多轮消息自然驱动，客户端无需拼全量历史。

运行：
    pip install requests
    python agent_client.py
退出：输入 exit / quit / q
"""
import requests

BASE = "http://localhost:8000"
API_KEY = "risk-demo-secret-key"
THREAD = "demo-1"          # 同一会话固定此 ID，记忆就在服务端按它串联


def main():
    print("电力价格风控助手（方式一 NL）。缺信息时助手会反问，直接补上即可。\n")
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。"); break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            print("再见。"); break

        resp = requests.post(
            f"{BASE}/agent/chat",
            headers={"X-API-Key": API_KEY},
            json={"message": user, "thread_id": THREAD},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"\n助手> {data['reply']}\n")


if __name__ == "__main__":
    main()
