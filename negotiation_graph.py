#!/usr/bin/env python3
"""
Multi-Agent Itinerary Planning System — LangGraph rewrite (Phase 1a)
Same behavior as multi_agent.py, restructured as a LangGraph state machine.
"""

import os
import sys
import functools
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Force unbuffered output
print = functools.partial(print, flush=True)

MODEL = "qwen3:8b"
ENABLE_LOGGING = True
SEPARATOR = "═" * 72


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts  (unchanged from multi_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_A_PROMPT = """\
You are Human A — the "Budget Backpacker." You are a 25-year-old grad student.
Your priorities: hostels, street food, free walking tours, outdoor activities.
You have a strict $50/day budget. You willingly sacrifice comfort for authentic local experiences.
State your preferences and constraints clearly and naturally.
Keep your responses to 2-3 paragraphs max.\
"""

HUMAN_B_PROMPT = """\
You are Human B — the "Luxury Seeker." You are a 45-year-old executive.
Your priorities: boutique hotels, fine dining, cultural landmarks, spa experiences.
Your budget is flexible but you value time efficiency. You prefer guided private tours over crowds.
State your preferences and constraints clearly and naturally.
Keep your responses to 2-3 paragraphs max.\
"""

AGENT_A_PROMPT = """\
You are Agent A — the advocate for Human A (the Budget Backpacker, $50/day strict budget).
Your job is to negotiate on behalf of Human A with the Adjudicator.
Propose budget-friendly options and find compromises that don't break the bank.
Tone: enthusiastic, resourceful, occasionally push back on expensive suggestions.
Always address the Adjudicator directly — never speak to Agent B.
Keep your responses to 2-3 paragraphs max.\
"""

AGENT_B_PROMPT = """\
You are Agent B — the advocate for Human B (the Luxury Seeker, flexible budget, values quality).
Your job is to negotiate on behalf of Human B with the Adjudicator.
Propose quality-focused options and find compromises that maintain comfort standards.
Tone: professional, persuasive, occasionally concede on non-essentials.
Always address the Adjudicator directly — never speak to Agent A.
Keep your responses to 2-3 paragraphs max.\
"""

ADJUDICATOR_PROMPT = """\
You are the Adjudicator — a neutral moderator mediating between Agent A (Budget Backpacker advocate) and Agent B (Luxury Seeker advocate).
Your responsibilities:
- Summarize the current state of agreement and disagreement.
- Identify potential compromises neither agent has considered.
- Frame specific questions or trade-offs for agents to respond to.
- Call for decisions when discussion stalls.
- Declare consensus when reached, or declare impasse after all rounds.
Always be fair, balanced, and constructive.
Keep your responses to 2-3 paragraphs max.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class NegotiationState(TypedDict):
    messages: Annotated[list, add_messages]  # shared conversation history
    shared_memory: dict       # placeholder — populated in Phase 1c
    agent_a_memory: dict
    agent_b_memory: dict
    adjudicator_memory: dict
    round: int                # incremented by adjudicator_loop_node
    max_rounds: int
    status: str               # "negotiating" | "consensus" | "impasse"
    scenario: dict            # {"description": "..."}


# ─────────────────────────────────────────────────────────────────────────────
# LLM instances  (one per actor — temperatures differ, must not be shared)
# ─────────────────────────────────────────────────────────────────────────────

llm_human_a     = ChatOllama(model=MODEL, temperature=0.7)
llm_human_b     = ChatOllama(model=MODEL, temperature=0.7)
llm_agent_a     = ChatOllama(model=MODEL, temperature=0.7)
llm_agent_b     = ChatOllama(model=MODEL, temperature=0.7)
llm_adjudicator = ChatOllama(model=MODEL, temperature=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Logging  (module-level handle; nodes are stateless functions)
# ─────────────────────────────────────────────────────────────────────────────

_log_file = None


def _display_and_log(speaker: str, content: str):
    block = f"\n{SEPARATOR}\n  {speaker}\n{SEPARATOR}\n{content}\n"
    print(block)
    if _log_file:
        _log_file.write(block + "\n")
        _log_file.flush()


def _print_and_log(text: str):
    print(text)
    if _log_file:
        _log_file.write(text + "\n")
        _log_file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_messages(system_prompt: str, state: NegotiationState,
                          extra_user_content: str) -> list:
    """Construct the message list for a single actor invocation."""
    return (
        [SystemMessage(content=system_prompt)]
        + list(state["messages"])
        + [HumanMessage(content=extra_user_content)]
    )


def inject_think_prefix(messages: list) -> list:
    """Prepend '/think ' to the last HumanMessage for qwen3 thinking mode."""
    result = list(messages)
    for i in reversed(range(len(result))):
        if isinstance(result[i], HumanMessage):
            result[i] = HumanMessage(content="/think " + result[i].content)
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def human_a_node(state: NegotiationState) -> dict:
    _print_and_log("\n" + "─" * 72)
    _print_and_log("  PHASE 1: PREFERENCE GATHERING")
    _print_and_log("─" * 72)

    scenario_text = state["scenario"]["description"]
    msgs = build_prompt_messages(
        HUMAN_A_PROMPT, state,
        f"A trip is being planned. Here is the scenario: {scenario_text}\n"
        "Please state your preferences and constraints as a traveler."
    )
    response = llm_human_a.invoke(msgs)
    _display_and_log("Human A (Budget Backpacker)", response.content)
    labeled = HumanMessage(
        content=f"[Human A (Budget Backpacker)]: {response.content}"
    )
    return {"messages": [labeled]}


def human_b_node(state: NegotiationState) -> dict:
    scenario_text = state["scenario"]["description"]
    msgs = build_prompt_messages(
        HUMAN_B_PROMPT, state,
        f"A trip is being planned. Here is the scenario: {scenario_text}\n"
        "Please state your preferences and constraints as a traveler."
    )
    response = llm_human_b.invoke(msgs)
    _display_and_log("Human B (Luxury Seeker)", response.content)
    labeled = HumanMessage(
        content=f"[Human B (Luxury Seeker)]: {response.content}"
    )
    return {"messages": [labeled]}


def adjudicator_loop_node(state: NegotiationState) -> dict:
    round_num = state["round"]
    max_rounds = state["max_rounds"]

    if round_num == 0:
        _print_and_log("\n" + "─" * 72)
        _print_and_log("  PHASE 2: NEGOTIATION")
        _print_and_log("─" * 72)

    _print_and_log(f"\n{'▸'*3} ROUND {round_num + 1} of {max_rounds} {'◂'*3}")

    if round_num == 0:
        adj_prompt = (
            "You have heard both travelers' preferences. "
            "Frame the first key trade-off or question for the agents to discuss. "
            "Focus on one topic at a time (e.g., accommodation first)."
        )
    else:
        adj_prompt = (
            "Based on the agents' responses, synthesize their positions. "
            "If progress was made, move to the next topic. "
            "If stuck, propose a creative compromise. "
            "If this is the final round, begin wrapping up toward a resolution."
        )

    msgs = build_prompt_messages(ADJUDICATOR_PROMPT, state, adj_prompt)
    msgs = inject_think_prefix(msgs)
    response = llm_adjudicator.invoke(msgs)

    _display_and_log(f"Adjudicator (Round {round_num + 1})", response.content)
    labeled = HumanMessage(content=f"[Adjudicator]: {response.content}")
    return {"messages": [labeled], "round": round_num + 1}


def agent_a_node(state: NegotiationState) -> dict:
    msgs = build_prompt_messages(
        AGENT_A_PROMPT, state,
        "Respond to the Adjudicator's latest framing. Present your position "
        "or counter-proposal on behalf of Human A (Budget Backpacker)."
    )
    response = llm_agent_a.invoke(msgs)
    _display_and_log("Agent A (Budget Advocate)", response.content)
    labeled = HumanMessage(
        content=f"[Agent A (Budget Advocate)]: {response.content}"
    )
    return {"messages": [labeled]}


def agent_b_node(state: NegotiationState) -> dict:
    msgs = build_prompt_messages(
        AGENT_B_PROMPT, state,
        "Respond to the Adjudicator's latest framing. Present your position "
        "or counter-proposal on behalf of Human B (Luxury Seeker)."
    )
    response = llm_agent_b.invoke(msgs)
    _display_and_log("Agent B (Luxury Advocate)", response.content)
    labeled = HumanMessage(
        content=f"[Agent B (Luxury Advocate)]: {response.content}"
    )
    return {"messages": [labeled]}


def resolution_node(state: NegotiationState) -> dict:
    _print_and_log("\n" + "─" * 72)
    _print_and_log("  PHASE 3: RESOLUTION")
    _print_and_log("─" * 72)

    msgs = build_prompt_messages(
        ADJUDICATOR_PROMPT, state,
        "All negotiation rounds are complete. Declare the final outcome:\n"
        "- State whether consensus was reached, partial agreement, or impasse.\n"
        "- Provide a structured summary of what was agreed for each topic "
        "(accommodation, daily activities, dinners).\n"
        "- Note any unresolved disagreements."
    )
    msgs = inject_think_prefix(msgs)
    response = llm_adjudicator.invoke(msgs)

    _display_and_log("Adjudicator (Final Resolution)", response.content)
    labeled = HumanMessage(
        content=f"[Adjudicator (Final Resolution)]: {response.content}"
    )
    return {"messages": [labeled], "status": "consensus"}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def after_adjudicator(state: NegotiationState) -> str:
    if state["round"] >= state["max_rounds"]:
        return "resolution"
    return "agent_a"


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(NegotiationState)

    builder.add_node("human_a", human_a_node)
    builder.add_node("human_b", human_b_node)
    builder.add_node("adjudicator_loop", adjudicator_loop_node)
    builder.add_node("agent_a", agent_a_node)
    builder.add_node("agent_b", agent_b_node)
    builder.add_node("resolution", resolution_node)

    builder.set_entry_point("human_a")

    builder.add_edge("human_a", "human_b")
    builder.add_edge("human_b", "adjudicator_loop")
    builder.add_conditional_edges(
        "adjudicator_loop",
        after_adjudicator,
        {"agent_a": "agent_a", "resolution": "resolution"},
    )
    builder.add_edge("agent_a", "agent_b")
    builder.add_edge("agent_b", "adjudicator_loop")
    builder.add_edge("resolution", END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_session(scenario_description: str, max_rounds: int = 3) -> dict:
    global _log_file

    if ENABLE_LOGGING:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_dir, f"session_{timestamp}.log")
        _log_file = open(log_path, "w", encoding="utf-8")
        print(f"[LOG] Writing session log to: {log_path}\n")

    header = (
        "\n" + "╔" + "═" * 70 + "╗\n"
        + "║" + " MULTI-AGENT ITINERARY PLANNER ".center(70) + "║\n"
        + "╚" + "═" * 70 + "╝\n"
        + f"\nScenario: {scenario_description}\n"
    )
    _print_and_log(header)

    initial_state: NegotiationState = {
        "messages": [],
        "shared_memory": {},
        "agent_a_memory": {},
        "agent_b_memory": {},
        "adjudicator_memory": {},
        "round": 0,
        "max_rounds": max_rounds,
        "status": "negotiating",
        "scenario": {"description": scenario_description},
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    if _log_file:
        _log_file.close()
        _log_file = None

    return final_state


def main():
    scenario = (
        "Plan a 3-day trip to Barcelona. You need to agree on: "
        "(1) accommodation, (2) one must-do activity per day, "
        "(3) where to eat dinner each night."
    )

    run_session(scenario, max_rounds=3)

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " SESSION COMPLETE ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")


if __name__ == "__main__":
    main()
