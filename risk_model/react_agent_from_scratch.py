"""手写 create_react_agent：看清 LangGraph 预构建 ReAct Agent 的内部实现。

两种等价实现：
  A) build_react_agent()      —— 用 StateGraph 从零拼出和 langgraph.prebuilt
                                  .create_react_agent 一模一样的图（节点+边+条件路由）。
  B) run_react_loop()         —— 不依赖 LangGraph，纯 Python 把 ReAct 循环写出来，
                                  直接展示算法本质。

为方便离线运行（无需 OPENAI_API_KEY），附带 FakeModel：第 1 轮假装要调工具，
第 2 轮给出最终回答，从而把「agent→tools→agent→END」的全过程跑出来。

运行：
    python -m risk_model.react_agent_from_scratch
"""
from typing import Optional, Sequence
from pydantic import PrivateAttr

from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from langgraph.graph import StateGraph, START, END, MessagesState


# ======================================================================
# 1) 手写版的 create_react_agent（基于 StateGraph，等价于预构建实现）
# ======================================================================
def build_react_agent(model, tools, prompt: Optional[str] = None,
                      checkpointer=None):
    """从零拼装 ReAct 图：

        START ─▶ agent ──(有 tool_calls)──▶ tools ─▶ agent ─▶ ...
                        └────(无 tool_calls)──────▶ END
    """
    # ① 把工具「绑定」到模型上（让模型知道有哪些工具可调）
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    # ② agent 节点：把消息喂给绑定了工具的 LLM，返回它的决策
    def agent_node(state: MessagesState):
        msgs: Sequence[BaseMessage] = state["messages"]
        if prompt:                                  # 系统提示词前置
            msgs = [SystemMessage(content=prompt)] + list(msgs)
        ai_msg = model_with_tools.invoke(msgs)      # 可能带 tool_calls
        return {"messages": [ai_msg]}

    # ③ tools 节点：执行上一条 AIMessage 里的每个工具调用，回写 ToolMessage
    def tools_node(state: MessagesState):
        outputs = []
        last_ai = state["messages"][-1]
        for call in last_ai.tool_calls:             # [{name, args, id}, ...]
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            outputs.append(ToolMessage(
                content=str(result),
                tool_call_id=call["id"],            # 必须回指，模型才能对应
                name=call["name"]))
        return {"messages": outputs}

    # ④ 条件路由：最后一条消息是否还要调工具？
    def should_continue(state: MessagesState):
        last_ai = state["messages"][-1]
        return "tools" if getattr(last_ai, "tool_calls", None) else END

    # ⑤ 组装图
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue,
                            {"tools": "tools", END: END})
    g.add_edge("tools", "agent")                    # 工具结果回灌 agent
    return g.compile(checkpointer=checkpointer)


# ======================================================================
# 2) 纯 Python 版 ReAct 循环（不依赖 LangGraph，看清算法本质）
# ======================================================================
def run_react_loop(model, tools, user_input: str,
                   prompt: Optional[str] = None, max_steps: int = 8):
    """和上面构图逻辑完全一致，只是用 while 循环手动驱动。"""
    tools_by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    if prompt:
        messages.insert(0, SystemMessage(content=prompt))

    for _ in range(max_steps):
        ai_msg = model.bind_tools(tools).invoke(messages)
        messages.append(ai_msg)
        if not getattr(ai_msg, "tool_calls", None):
            return ai_msg.content                 # 不再调工具 → 结束

        for call in ai_msg.tool_calls:             # 逐个执行并回写
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(
                content=str(result), tool_call_id=call["id"], name=call["name"]))
        # 回到循环顶部，把工具结果一起喂给模型继续推理
    return "（达到最大步数，强制结束）"


# ======================================================================
# 3) 离线演示：假模型 + 假工具，无需 OpenAI 也能看到完整过程
# ======================================================================
@tool
def get_current_risk() -> dict:
    """查询当前电价尖峰风险（演示用假工具）。"""
    return {"risk_score": 0.73, "is_risk": True, "threshold": 0.5}


class FakeModel(BaseChatModel):
    """脚本化假模型：第 1 轮决定调工具，第 2 轮给出最终回答。"""
    tools: list = []
    _turn: int = PrivateAttr(default=0)

    def bind_tools(self, tools):
        self.tools = list(tools)
        return self

    def _llm_type(self) -> str:
        return "fake-reasoning-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._turn += 1
        if self._turn == 1:                        # 第 1 轮：调用工具
            call = {"name": self.tools[0].name,
                    "args": {}, "id": "call_demo_1"}
            ai = AIMessage(content="", tool_calls=[call])
        else:                                      # 第 2 轮：总结回复
            ai = AIMessage(content="根据工具返回，当前电价尖峰风险得分为 0.73，"
                                   "超过阈值 0.5，属于尖峰风险，建议预警。")
        return ChatResult(generations=[ChatGeneration(message=ai)])


# ======================================================================
# 4) 零依赖版：用原生 dict 演示 ReAct 算法（无需安装任何库即可运行）
# ======================================================================
def run_react_loop_naive(user_input: str, max_steps: int = 8):
    """完全不依赖 langchain/langgraph，用普通 dict 表示消息与工具调用。

    消息 dict: {"role": "user"|"assistant"|"tool", "content": ..., "tool_calls": [...]}
    工具调用: {"name": ..., "args": ..., "id": ...}
    """
    # 假「模型」：根据是否见过工具结果决定下一步
    seen_tool = {"v": False}

    def fake_llm(messages: list) -> dict:
        if not seen_tool["v"]:                  # 第 1 步：决定调工具
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"name": "get_current_risk",
                                    "args": {}, "id": "c1"}]}
        return {"role": "assistant",            # 第 2 步：总结
                "content": "电价尖峰风险 0.73 > 阈值 0.5，建议预警。",
                "tool_calls": []}

    tools = {"get_current_risk": lambda args: {"risk_score": 0.73, "is_risk": True}}

    messages = [{"role": "user", "content": user_input, "tool_calls": []}]
    for step in range(max_steps):
        ai = fake_llm(messages)
        messages.append(ai)
        print(f"  step {step}: assistant 决策 -> "
              f"{('调用工具 ' + ai['tool_calls'][0]['name']) if ai['tool_calls'] else '直接回答'}")
        if not ai["tool_calls"]:
            return ai["content"]

        for call in ai["tool_calls"]:           # 执行工具并回写
            result = tools[call["name"]](call["args"])
            seen_tool["v"] = True
            messages.append({"role": "tool",
                             "content": str(result),
                             "tool_call_id": call["id"]})
            print(f"         tool {call['name']} -> {result}")
    return "（达到最大步数）"


# ======================================================================
# 5) 手写版 + Human-in-the-loop：执行敏感工具前先问人类确认
# ======================================================================
# 标记为「需人工确认」的敏感工具名
SENSITIVE_TOOLS = {"get_current_risk"}


def build_react_agent_with_hitl(model, tools, prompt: Optional[str] = None,
                                checkpointer=None):
    """在 agent 与 tools 之间插入 guard 节点：

        START -> agent -> guard --(有 tool_calls)--> tools -> agent -> ...
                            └────────(无 tool_calls)────────> END

    guard 命中敏感工具时调用 interrupt() 暂停，把待执行工具交给人类；
    人类 resume 批准则放行、拒绝则回写「被拒」ToolMessage，由模型收尾。
    """
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def agent_node(state: MessagesState):
        msgs = list(state["messages"])
        if prompt and not (msgs and isinstance(msgs[0], SystemMessage)):
            msgs = [SystemMessage(content=prompt)] + msgs
        return {"messages": [model_with_tools.invoke(msgs)]}

    def tools_node(state: MessagesState):
        out = []
        for call in state["messages"][-1].tool_calls:
            out.append(ToolMessage(
                content=str(tools_by_name[call["name"]].invoke(call["args"])),
                tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    def guard_node(state: MessagesState):
        """敏感工具拦截：暂停等人类批准。"""
        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        sensitive = [c for c in calls if c["name"] in SENSITIVE_TOOLS]
        if not sensitive:
            return {}                                  # 无敏感工具，直接放行
        decision = interrupt({                         # ★ 暂停点
            "question": "即将执行敏感工具，是否允许？",
            "tools": [c["name"] for c in sensitive],
        })
        if not decision.get("approve", False):         # 人类拒绝
            return {"messages": [ToolMessage(
                content="用户拒绝了该工具调用。",
                tool_call_id=c["id"], name=c["name"]) for c in sensitive]}
        return {}                                      # 人类批准，原样放行

    def route_after_guard(state: MessagesState):
        # 看【最后一条】消息决定去向：
        #  - 仍是带 tool_calls 的 AI 消息 → 工具还没执行（批准/无敏感）→ 去 tools
        #  - ToolMessage（拒绝已回写/工具已执行）→ 回 agent 让模型收尾
        #  - 普通 AI 消息（无 tool_calls，已是最终答复）→ END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        if isinstance(last, ToolMessage):
            return "agent"
        return END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("guard", guard_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"tools": "tools", "agent": "agent", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def main_hitl():
    agent = build_react_agent_with_hitl(
        FakeModel(), [get_current_risk],
        prompt="你是电力价格风控助手，请用工具回答。",
        checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "hitl"}}

    print("--- 第 1 次 invoke：模型决定调 get_current_risk，触发中断 ---")
    r1 = agent.invoke({"messages": [("user", "帮我查当前电价尖峰风险")]}, config=cfg)
    intr = r1.get("__interrupts__")
    print("中断返回:", intr[0].value if intr else None)

    print("--- 人类批准（Command(resume={'approve': True})）继续 ---")
    r2 = agent.invoke(Command(resume={"approve": True}), config=cfg)
    for i, m in enumerate(r2["messages"]):
        print(f"[{i}] {m.type}: {m.content or ''}"
              + (f"  tool_calls={m.tool_calls}" if getattr(m, 'tool_calls', None) else ""))

    # 演示「拒绝」分支：重开一个线程
    print("\n--- 另起线程演示「拒绝」：人类不批准 ---")
    cfg2 = {"configurable": {"thread_id": "hitl-deny"}}
    agent.invoke({"messages": [("user", "查风险")]}, config=cfg2)
    r3 = agent.invoke(Command(resume={"approve": False}), config=cfg2)
    print("最终回答:", r3["messages"][-1].content)


def main():
    print("===== A) StateGraph 手写版（与 langgraph.prebuilt 等价）=====")
    agent = build_react_agent(
        FakeModel(), [get_current_risk],
        prompt="你是电力价格风控助手，请用工具回答。")
    result = agent.invoke({"messages": [("user", "现在电价尖峰风险高吗？")]})
    for i, m in enumerate(result["messages"]):
        print(f"[{i}] {m.type}: {m.content or ''}"
              + (f"  tool_calls={m.tool_calls}" if getattr(m, 'tool_calls', None) else ""))

    print("\n===== B) 纯 Python 循环版 =====")
    answer = run_react_loop(
        FakeModel(), [get_current_risk],
        "现在电价尖峰风险高吗？",
        prompt="你是电力价格风控助手，请用工具回答。")
    print("最终回答:", answer)

    print("\n===== C) 零依赖 dict 版（无需安装任何库）=====")
    print("最终回答:", run_react_loop_naive("现在电价尖峰风险高吗？"))

    print("\n===== D) 手写版 + Human-in-the-loop（需 langgraph）=====")
    main_hitl()


if __name__ == "__main__":
    main()
