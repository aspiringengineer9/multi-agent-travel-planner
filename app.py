"""
Streamlit UI for the Multi-Agent Travel Negotiation System — Phase 1f.

Phase state machine:  setup → streaming → interrupted → streaming → ... → done

Human-in-the-loop: the graph pauses after each adjudicator round synthesis
(via interrupt_after=["adjudicator_loop"]). The human can optionally inject
a message before agents respond, or continue without input.
"""

import os
import sys
import uuid

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import after path setup
import negotiation_graph as ng
from negotiation_graph import (
    load_config,
    build_scenario,
    build_graph,
    MODEL,
    ADJUDICATOR_TOOLS,
    AGENT_TOOLS,
)
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# ── Scenario options ──────────────────────────────────────────────────────────
SCENARIOS = {
    "Barcelona — 3-day trip (3 rounds)": os.path.join(
        SCRIPT_DIR, "scenarios", "scenario_barcelona.yaml"
    ),
    "Los Angeles — 1-day trip (2 rounds)": os.path.join(
        SCRIPT_DIR, "scenarios", "scenario_la.yaml"
    ),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Travel Planner",
    page_icon="✈",
    layout="wide",
)
st.title("Multi-Agent Travel Negotiation")
st.caption("LangGraph · Ollama · Human-in-the-Loop")

# ── Session state initialization ──────────────────────────────────────────────
_DEFAULTS = {
    "phase": "setup",        # "setup" | "streaming" | "interrupted" | "done"
    "graph": None,
    "thread_cfg": None,
    "messages": [],          # list[dict]: {"role", "speaker", "content"}
    "interrupt_round": None, # int: round number at current pause
    "interrupt_max": None,   # int: max_rounds
    "interrupt_humans": [],  # list[dict]: {"key", "display_name", "name"} for each human actor
    "final_status": None,    # str: negotiation outcome
    "_is_resume": False,
    "_resume_cmd": None,     # None | Command(...)
    "_pending_state": None,  # initial NegotiationState for first run
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _init_llms(scenario: dict) -> None:
    """Populate ng._llms — mirrors run_session() LLM setup."""
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "2")
    ng._llms = {}
    for key, actor in scenario["actors"].items():
        if key == "adjudicator":
            adj_base = ChatOllama(model=MODEL, temperature=0.4)
            ng._llms[key] = adj_base.bind_tools(ADJUDICATOR_TOOLS)
            ng._llms["adjudicator_think"] = adj_base
        elif actor["role"] == "agent":
            base = ChatOllama(model=MODEL, temperature=0.7)
            ng._llms[key] = base.bind_tools(AGENT_TOOLS)
        else:
            ng._llms[key] = ChatOllama(model=MODEL, temperature=0.7)


def _parse_speaker(raw_content: str, fallback: str) -> tuple[str, str]:
    """Parse '[Speaker Name]: body' format used throughout negotiation_graph.py."""
    if raw_content.startswith("[") and "]: " in raw_content:
        bracket_end = raw_content.index("]: ")
        return raw_content[1:bracket_end], raw_content[bracket_end + 3:]
    return fallback, raw_content


def _extract_messages_from_chunk(chunk: dict) -> list[dict]:
    """Pull new messages out of a stream_mode='updates' chunk."""
    result = []
    for node_name, delta in chunk.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(delta, dict):
            continue
        for msg in delta.get("messages") or []:
            content = getattr(msg, "content", None)
            if not content or not content.strip():
                continue
            speaker, body = _parse_speaker(content, node_name)
            # Determine chat role for display
            role = "user" if "Human" in speaker and "Observer" not in speaker else "assistant"
            result.append({"role": role, "speaker": speaker, "content": body})
    return result


def _render_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"**{msg['speaker']}**")
            st.markdown(msg["content"])


def _run_stream(stream_input) -> bool:
    """
    Run graph.stream() until interrupt or completion.
    Accumulates messages in st.session_state.messages.
    Returns True if interrupted, False if completed.
    """
    graph = st.session_state.graph
    thread_cfg = st.session_state.thread_cfg
    interrupted = False

    for chunk in graph.stream(stream_input, config=thread_cfg, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupted = True
            break
        new_msgs = _extract_messages_from_chunk(chunk)
        st.session_state.messages.extend(new_msgs)

    return interrupted


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: setup
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "setup":
    st.subheader("Configure Negotiation")

    scenario_label = st.selectbox("Select scenario:", list(SCENARIOS.keys()))
    st.info(
        "The negotiation pauses after each adjudicator round. "
        "You can optionally inject a message before agents respond."
    )

    if st.button("Start Negotiation", type="primary"):
        config_path = SCENARIOS[scenario_label]
        config = load_config(config_path)
        scenario = build_scenario(config)
        _init_llms(scenario)

        checkpointer = MemorySaver()
        graph = build_graph(scenario, checkpointer=checkpointer)
        thread_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

        initial_state = {
            "messages": [],
            "shared_memory": {},
            "agent_memories": {},
            "adjudicator_memory": {},
            "round": 0,
            "max_rounds": scenario.get("max_rounds", 3),
            "status": "negotiating",
            "scenario": scenario,
        }

        st.session_state.graph = graph
        st.session_state.thread_cfg = thread_cfg
        st.session_state.messages = []
        st.session_state.phase = "streaming"
        st.session_state._is_resume = False
        st.session_state._pending_state = initial_state
        st.session_state._resume_cmd = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: streaming
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "streaming":
    _render_messages()
    status_slot = st.empty()
    status_slot.info("Negotiation in progress…  (this may take a minute per round)")

    is_resume = st.session_state._is_resume
    if is_resume:
        stream_input = st.session_state._resume_cmd  # None or Command(...)
    else:
        stream_input = st.session_state._pending_state

    interrupted = _run_stream(stream_input)
    status_slot.empty()

    if interrupted:
        # Read round info and human actors from checkpointed state
        snapshot = st.session_state.graph.get_state(st.session_state.thread_cfg)
        st.session_state.interrupt_round = snapshot.values.get("round", "?")
        st.session_state.interrupt_max = snapshot.values.get("max_rounds", "?")
        actors = snapshot.values.get("scenario", {}).get("actors", {})
        st.session_state.interrupt_humans = [
            {"key": k, "display_name": v.get("display_name", k), "name": v.get("name", k)}
            for k, v in actors.items()
            if v.get("role") == "human"
        ]
        st.session_state.phase = "interrupted"
    else:
        snapshot = st.session_state.graph.get_state(st.session_state.thread_cfg)
        st.session_state.final_status = snapshot.values.get("status", "unknown")
        st.session_state.phase = "done"

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: interrupted
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "interrupted":
    _render_messages()

    st.divider()
    round_num = st.session_state.interrupt_round
    max_rounds = st.session_state.interrupt_max
    humans = st.session_state.interrupt_humans  # [{"key", "display_name", "name"}, ...]

    st.subheader(f"Round {round_num}/{max_rounds} — Your turn")
    st.info(
        "The adjudicator has framed the next question. "
        "Each traveler can now weigh in directly before their agent responds. "
        "Leave a field blank to let the agent speak on their own."
    )

    # One text area per human
    inputs: dict[str, str] = {}
    for h in humans:
        inputs[h["key"]] = st.text_area(
            f"{h['display_name']}",
            key=f"human_input_{h['key']}",
            height=80,
            placeholder=f"What does {h['name']} want to say? (optional)",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue (no input)", type="secondary", use_container_width=True):
            st.session_state.phase = "streaming"
            st.session_state._is_resume = True
            st.session_state._resume_cmd = None
            st.session_state.interrupt_round = None
            st.rerun()
    with col2:
        if st.button("Submit input", type="primary", use_container_width=True):
            new_msgs = []
            for h in humans:
                text = inputs[h["key"]].strip()
                if text:
                    labeled = f"[{h['display_name']}]: {text}"
                    new_msgs.append(HumanMessage(content=labeled))
                    st.session_state.messages.append({
                        "role": "user",
                        "speaker": h["display_name"],
                        "content": text,
                    })
            resume_cmd = Command(update={"messages": new_msgs}) if new_msgs else None
            st.session_state.phase = "streaming"
            st.session_state._is_resume = True
            st.session_state._resume_cmd = resume_cmd
            st.session_state.interrupt_round = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: done
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "done":
    _render_messages()

    st.divider()
    final_status = st.session_state.final_status or "unknown"
    label = final_status.upper().replace("_", " ")

    if final_status == "consensus":
        st.success(f"Negotiation Complete — {label}")
    elif final_status == "partial_consensus":
        st.warning(f"Negotiation Complete — {label}")
    else:
        st.error(f"Negotiation Complete — {label}")

    if st.button("Start New Negotiation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
