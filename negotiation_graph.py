#!/usr/bin/env python3
"""
Multi-Agent Itinerary Planning System — LangGraph rewrite (Phase 1c)
Config-driven + per-actor memory: agents track private proposals across rounds;
adjudicator tracks position deltas; shared memory surfaces agreed items to all.
"""

import os
import json
import subprocess
import functools
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

import yaml
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

print = functools.partial(print, flush=True)

MODEL = "qwen3:8b"


def _get_git_info() -> tuple[str, str]:
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"
ENABLE_LOGGING = True
SEPARATOR = "═" * 72

ADJUDICATOR_PROMPT = """\
You are the Adjudicator — a neutral moderator mediating between the agents representing each traveler.
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

def merge_agent_memories(existing: dict, update: dict) -> dict:
    """Shallow-merge so parallel agent writes to different keys don't clobber each other."""
    return {**existing, **update}


class NegotiationState(TypedDict):
    messages: Annotated[list, add_messages]
    shared_memory: dict
    agent_memories: Annotated[dict, merge_agent_memories]  # {"agent_a": {...}, ...}
    adjudicator_memory: dict
    round: int
    max_rounds: int
    status: str               # "negotiating" | "consensus" | "impasse"
    scenario: dict            # full enriched config (actors, topics, description, ...)


# ─────────────────────────────────────────────────────────────────────────────
# LLM registry  (populated in run_session before graph invocation)
# ─────────────────────────────────────────────────────────────────────────────

_llms: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Config loading and prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _human_prompt(actor: dict) -> str:
    budget_line = (
        f"You have a strict ${actor['budget']}/day budget."
        if actor.get("budget")
        else "Your budget is flexible."
    )
    return (
        f"You are {actor['name']}.\n"
        f"{actor['personality']}.\n"
        f"{budget_line} "
        "State your preferences and constraints clearly and naturally. "
        "Keep your responses to 2-3 paragraphs max."
    )


def _agent_prompt(human: dict, agent_key: str, agent_display: str,
                  all_agent_meta: dict, scenario_desc: str, topics: list) -> str:
    budget_line = (
        f"${human['budget']}/day strict budget — do not exceed this"
        if human.get("budget")
        else "flexible budget — comfort and quality matter more than cost"
    )
    personality = human.get("personality", "")

    other_lines = []
    for key, meta in all_agent_meta.items():
        if key != agent_key:
            other_lines.append(f"  - {meta['display_name']}: advocates for {meta['human_name']}")
    other_agents_text = "\n".join(other_lines) if other_lines else "  (none)"

    return (
        f"=== NEGOTIATION SETUP ===\n"
        f"Scenario: {scenario_desc}\n"
        f"Topics under negotiation: {', '.join(topics)}\n"
        f"Participants: An Adjudicator (neutral moderator) and {len(all_agent_meta)} advocate agents.\n"
        f"\n"
        f"=== YOUR IDENTITY ===\n"
        f"You are: {agent_display}\n"
        f"You advocate for: {human['name']}\n"
        f"Your human's profile: {personality}\n"
        f"Your human's budget: {budget_line}\n"
        f"\n"
        f"=== OTHER AGENTS ===\n"
        f"{other_agents_text}\n"
        f"\n"
        f"=== YOUR INSTRUCTIONS ===\n"
        f"1. Respond ONLY to the Adjudicator's latest question or framing.\n"
        f"2. Argue firmly for options that fit {human['name']}'s profile and budget.\n"
        f"3. Push back on proposals that violate {human['name']}'s constraints.\n"
        f"4. Only concede on points genuinely acceptable to {human['name']}.\n"
        f"5. Address the Adjudicator directly — never address another agent.\n"
        f"6. Do NOT repeat or echo what other agents have said.\n"
        f"7. Keep responses to 2-3 paragraphs max.\n"
        f"\n"
        f"IMPORTANT: You are {agent_display}. Speak in your own voice based on "
        f"{human['name']}'s priorities. Do not copy or paraphrase other agents' positions."
    )


def build_scenario(config: dict) -> dict:
    """Enrich raw config with generated system_prompt and display_name for every actor."""
    actors = config["actors"]
    humans = {k: v for k, v in actors.items() if v["role"] == "human"}
    scenario_desc = config["scenario"]
    topics = config.get("topics", [])

    enriched: dict = {}
    agent_meta: dict = {}  # first pass: collect agent metadata for cross-references

    for i, (key, human) in enumerate(humans.items(), start=1):
        letter = chr(ord("a") + i - 1)
        agent_key = f"agent_{letter}"
        display_name = f"Agent {letter.upper()} ({human['name']} Advocate)"

        enriched[key] = {
            **human,
            "system_prompt": _human_prompt(human),
            "display_name": f"Human {letter.upper()} ({human['name']})",
        }
        agent_meta[agent_key] = {
            "display_name": display_name,
            "human_name": human["name"],
            "human": human,
        }

    # second pass: generate agent prompts with cross-references to other agents
    for agent_key, meta in agent_meta.items():
        enriched[agent_key] = {
            "role": "agent",
            "system_prompt": _agent_prompt(
                meta["human"], agent_key, meta["display_name"],
                agent_meta, scenario_desc, topics,
            ),
            "display_name": meta["display_name"],
        }

    enriched["adjudicator"] = {
        "role": "adjudicator",
        "system_prompt": ADJUDICATOR_PROMPT,
        "display_name": "Adjudicator",
    }

    return {**config, "description": scenario_desc, "actors": enriched}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
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


def _log_session_header(config_path: str, scenario: dict, started_at: datetime):
    """Write a structured preamble to the log file before the conversation begins."""
    if not _log_file:
        return

    actors = scenario["actors"]
    topics = ", ".join(scenario.get("topics", []))
    branch, commit = _get_git_info()

    lines = [
        "╔" + "═" * 70 + "╗",
        "║" + " SESSION METADATA ".center(70) + "║",
        "╚" + "═" * 70 + "╝",
        f"  Started   : {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Branch    : {branch}",
        f"  Commit    : {commit}",
        f"  Config    : {os.path.abspath(config_path)}",
        f"  Model     : {MODEL}",
        f"  Max rounds: {scenario.get('max_rounds', 3)}",
        f"  Topics    : {topics}",
        "",
        "  ACTORS",
        "  " + "─" * 40,
    ]

    for key, actor in actors.items():
        lines.append(f"  [{key}]  {actor.get('display_name', key)}  (role: {actor['role']})")
        if actor.get("personality"):
            lines.append(f"    personality : {actor['personality']}")
        if actor.get("budget") is not None:
            lines.append(f"    budget      : ${actor['budget']}/day")
        lines.append(f"    system prompt:")
        for prompt_line in actor["system_prompt"].splitlines():
            lines.append(f"      {prompt_line}")
        lines.append("")

    lines += [
        "═" * 72,
        "  CONVERSATION TRANSCRIPT",
        "═" * 72,
        "",
    ]

    _log_file.write("\n".join(lines) + "\n")
    _log_file.flush()


def _log_session_footer(started_at: datetime, final_status: str, total_messages: int):
    """Write a summary footer to the log file after the conversation ends."""
    if not _log_file:
        return

    ended_at = datetime.now()
    duration = ended_at - started_at
    minutes, seconds = divmod(int(duration.total_seconds()), 60)

    lines = [
        "",
        "═" * 72,
        "  SESSION SUMMARY",
        "═" * 72,
        f"  Started : {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Ended   : {ended_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Duration: {minutes}m {seconds}s",
        f"  Status  : {final_status}",
        f"  Turns   : {total_messages}",
        "═" * 72,
    ]

    _log_file.write("\n".join(lines) + "\n")
    _log_file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def memory_read(memory: dict) -> str:
    """Format a memory dict as a readable bullet list for prompt injection."""
    if not memory:
        return "(none)"
    return "\n".join(f"  - {k}: {v}" for k, v in memory.items())


def memory_write(memory: dict, key: str, value: str) -> dict:
    """Return a new memory dict with the entry added (immutable update)."""
    return {**memory, key: value}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_messages(system_prompt: str, state: NegotiationState,
                          extra_user_content: str) -> list:
    return (
        [SystemMessage(content=system_prompt)]
        + list(state["messages"])
        + [HumanMessage(content=extra_user_content)]
    )


def inject_think_prefix(messages: list) -> list:
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

    actor = state["scenario"]["actors"]["human_a"]
    msgs = build_prompt_messages(
        actor["system_prompt"], state,
        f"A trip is being planned. Here is the scenario: {state['scenario']['description']}\n"
        "Please state your preferences and constraints as a traveler."
    )
    response = _llms["human_a"].invoke(msgs)
    _display_and_log(actor["display_name"], response.content)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {response.content}")
    return {"messages": [labeled]}


def human_b_node(state: NegotiationState) -> dict:
    actor = state["scenario"]["actors"]["human_b"]
    msgs = build_prompt_messages(
        actor["system_prompt"], state,
        f"A trip is being planned. Here is the scenario: {state['scenario']['description']}\n"
        "Please state your preferences and constraints as a traveler."
    )
    response = _llms["human_b"].invoke(msgs)
    _display_and_log(actor["display_name"], response.content)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {response.content}")
    return {"messages": [labeled]}


def adjudicator_loop_node(state: NegotiationState) -> dict:
    round_num = state["round"]
    max_rounds = state["max_rounds"]
    actor = state["scenario"]["actors"]["adjudicator"]
    topics = ", ".join(state["scenario"].get("topics", []))

    if round_num == 0:
        _print_and_log("\n" + "─" * 72)
        _print_and_log("  PHASE 2: NEGOTIATION")
        _print_and_log("─" * 72)

    _print_and_log(f"\n{'▸'*3} ROUND {round_num + 1} of {max_rounds} {'◂'*3}")

    if round_num == 0:
        adj_prompt = (
            "You have heard both travelers' preferences. "
            f"Topics to negotiate: {topics}. "
            "Frame the first key trade-off or question for the agents to discuss. "
            "Focus on one topic at a time."
        )
    else:
        adj_prompt = (
            "Based on the agents' responses, synthesize their positions. "
            "If progress was made, move to the next topic. "
            "If stuck, propose a creative compromise. "
            "If this is the final round, begin wrapping up toward a resolution."
        )

    adj_mem = state["adjudicator_memory"]
    shared = state["shared_memory"]

    memory_context = ""
    if adj_mem:
        memory_context += f"\nYour position tracking from previous rounds:\n{memory_read(adj_mem)}"
    if shared:
        memory_context += f"\nCurrently agreed items:\n{memory_read(shared)}"

    adj_prompt_full = adj_prompt + (f"\n\n{memory_context}" if memory_context else "")

    msgs = build_prompt_messages(actor["system_prompt"], state, adj_prompt_full)
    msgs = inject_think_prefix(msgs)
    response = _llms["adjudicator"].invoke(msgs)

    _display_and_log(f"{actor['display_name']} (Round {round_num + 1})", response.content)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {response.content}")

    synthesis_key = f"round_{round_num + 1}_synthesis"
    updated_adj_memory = memory_write(adj_mem, synthesis_key, response.content)

    # Extract a structured round summary for shared memory — visible to all actors next round.
    extraction_msgs = [
        SystemMessage(content="You are a precise note-taker summarizing a negotiation round."),
        HumanMessage(content=(
            "Based on the adjudicator's statement below, write a brief structured summary "
            "with these three sections (use these exact headers):\n"
            "AGREED: (what both parties explicitly agreed on, or 'Nothing yet')\n"
            "DISAGREED: (the key sticking points still unresolved)\n"
            "PROPOSED: (the adjudicator's suggested path forward or compromise)\n\n"
            "Be concise — 1-2 lines per section.\n\n"
            f"Adjudicator statement:\n{response.content}"
        )),
    ]
    extraction = _llms["adjudicator"].invoke(extraction_msgs)
    updated_shared = memory_write(shared, synthesis_key, extraction.content)

    return {
        "messages": [labeled],
        "round": round_num + 1,
        "adjudicator_memory": updated_adj_memory,
        "shared_memory": updated_shared,
    }


def make_agent_node(agent_key: str):
    """Factory: return a node function for the given agent key."""

    def agent_node(state: NegotiationState) -> dict:
        actor = state["scenario"]["actors"][agent_key]
        round_num = state["round"]
        my_memory = state["agent_memories"].get(agent_key, {})
        shared = state["shared_memory"]

        memory_context = ""
        if my_memory:
            memory_context += f"\nYour notes from previous rounds:\n{memory_read(my_memory)}"
        if shared:
            memory_context += f"\nCurrently agreed (from adjudicator):\n{memory_read(shared)}"

        user_content = (
            f"You are responding as {actor['display_name']} in Round {round_num}.\n"
            "Respond to the Adjudicator's latest framing. Present your position "
            "or counter-proposal on behalf of your human."
            + (f"\n\n{memory_context}" if memory_context else "")
        )
        msgs = build_prompt_messages(actor["system_prompt"], state, user_content)
        response = _llms[agent_key].invoke(msgs)
        _display_and_log(actor["display_name"], response.content)
        labeled = HumanMessage(content=f"[{actor['display_name']}]: {response.content}")
        updated_memory = memory_write(my_memory, f"round_{round_num}_proposal", response.content)
        return {"messages": [labeled], "agent_memories": {agent_key: updated_memory}}

    agent_node.__name__ = agent_key
    return agent_node


def resolution_node(state: NegotiationState) -> dict:
    _print_and_log("\n" + "─" * 72)
    _print_and_log("  PHASE 3: RESOLUTION")
    _print_and_log("─" * 72)

    actor = state["scenario"]["actors"]["adjudicator"]
    adj_mem = state["adjudicator_memory"]
    shared = state["shared_memory"]

    memory_context = ""
    if adj_mem:
        memory_context += f"\nYour position tracking across all rounds:\n{memory_read(adj_mem)}"
    if shared:
        memory_context += f"\nCurrently agreed items:\n{memory_read(shared)}"

    resolution_prompt = (
        "All negotiation rounds are complete. Declare the final outcome:\n"
        "- State whether consensus was reached, partial agreement, or impasse.\n"
        "- Provide a structured summary of what was agreed for each topic.\n"
        "- Note any unresolved disagreements."
        + (f"\n\n{memory_context}" if memory_context else "")
    )
    msgs = build_prompt_messages(actor["system_prompt"], state, resolution_prompt)
    msgs = inject_think_prefix(msgs)
    response = _llms["adjudicator"].invoke(msgs)

    _display_and_log(f"{actor['display_name']} (Final Resolution)", response.content)
    labeled = HumanMessage(
        content=f"[{actor['display_name']} (Final Resolution)]: {response.content}"
    )
    return {"messages": [labeled], "status": "consensus"}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def after_adjudicator(state: NegotiationState) -> list:
    """Fan-out: send identical state snapshots to all agents in parallel, or route to resolution."""
    if state["round"] >= state["max_rounds"]:
        return [Send("resolution", state)]
    agent_keys = [
        k for k, v in state["scenario"]["actors"].items()
        if v["role"] == "agent"
    ]
    return [Send(k, state) for k in agent_keys]


def agent_collector_node(state: NegotiationState) -> dict:
    """No-op convergence point after parallel agent execution."""
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(scenario: dict) -> StateGraph:
    builder = StateGraph(NegotiationState)

    builder.add_node("human_a", human_a_node)
    builder.add_node("human_b", human_b_node)
    builder.add_node("adjudicator_loop", adjudicator_loop_node)
    builder.add_node("agent_collector", agent_collector_node)
    builder.add_node("resolution", resolution_node)

    # Dynamic agent nodes from scenario config
    agent_keys = [k for k, v in scenario["actors"].items() if v["role"] == "agent"]
    for agent_key in agent_keys:
        builder.add_node(agent_key, make_agent_node(agent_key))
        builder.add_edge(agent_key, "agent_collector")

    builder.set_entry_point("human_a")
    builder.add_edge("human_a", "human_b")
    builder.add_edge("human_b", "adjudicator_loop")
    builder.add_conditional_edges(
        "adjudicator_loop",
        after_adjudicator,
        {k: k for k in agent_keys} | {"resolution": "resolution"},
    )
    builder.add_edge("agent_collector", "adjudicator_loop")
    builder.add_edge("resolution", END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_session(config_path: str) -> dict:
    global _llms, _log_file

    config = load_config(config_path)
    scenario = build_scenario(config)
    max_rounds = scenario.get("max_rounds", 3)

    _llms = {
        key: ChatOllama(model=MODEL, temperature=(0.4 if key == "adjudicator" else 0.7))
        for key in scenario["actors"]
    }

    started_at = datetime.now()

    if ENABLE_LOGGING:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = started_at.strftime("%Y-%m-%d_%H%M%S")
        log_path = os.path.join(output_dir, f"session_{timestamp}.log")
        _log_file = open(log_path, "w", encoding="utf-8")
        print(f"[LOG] Writing session log to: {log_path}\n")
        _log_session_header(config_path, scenario, started_at)

    header = (
        "\n" + "╔" + "═" * 70 + "╗\n"
        + "║" + " MULTI-AGENT ITINERARY PLANNER ".center(70) + "║\n"
        + "╚" + "═" * 70 + "╝\n"
        + f"\nScenario: {scenario['description']}\n"
    )
    _print_and_log(header)

    initial_state: NegotiationState = {
        "messages": [],
        "shared_memory": {},
        "agent_memories": {},
        "adjudicator_memory": {},
        "round": 0,
        "max_rounds": max_rounds,
        "status": "negotiating",
        "scenario": scenario,
    }

    graph = build_graph(scenario)
    final_state = graph.invoke(initial_state)

    _log_session_footer(
        started_at,
        final_state.get("status", "unknown"),
        len(final_state.get("messages", [])),
    )

    if ENABLE_LOGGING:
        memory_path = log_path.replace(".log", "_memory.json")
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump({
                "shared_memory": final_state.get("shared_memory", {}),
                "agent_memories": final_state.get("agent_memories", {}),
                "adjudicator_memory": final_state.get("adjudicator_memory", {}),
            }, f, indent=2)
        print(f"[LOG] Memory snapshot written to: {memory_path}")

    if _log_file:
        _log_file.close()
        _log_file = None

    return final_state


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios", "scenario_barcelona.yaml"
    )
    run_session(config_path)

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " SESSION COMPLETE ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")


if __name__ == "__main__":
    main()
