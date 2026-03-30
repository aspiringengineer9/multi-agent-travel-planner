# Design Decision 001: Orchestration Framework

**Date**: 2026-03-29
**Status**: Decided
**Decision**: LangGraph — chosen for learning value + future-proofing (checkpointing, human-in-the-loop for Streamlit UI in Phase 4)

---

## The Question

Should this project use an orchestration framework (LangGraph, CrewAI, AutoGen, etc.) or continue with plain Python for multi-agent coordination?

---

## What We Actually Need (Phase 2 Requirements)

| Requirement | Description | Complexity Without Framework |
|-------------|-------------|------------------------------|
| Turn-taking | Fixed cycle: adjudicator → agent_a → agent_b, with early exit | A `for` loop + `if/break` |
| State | Shared history, per-agent memory, round counter | Fields on a class |
| Tool calling | Agents optionally invoke budget_calculator, memory tools | ~40 lines (parse response, execute, re-submit) |
| Config loading | YAML → personas, constraints, scenario | `yaml.safe_load()` + dataclass |
| Checkpointing | Optional pause before final outcome | `json.dump(state)` at this scale |

**Key observation**: There is no dynamic routing (LLM doesn't pick which agent goes next), no parallel branches, no subgraphs. The control flow is a for-loop.

---

## Framework Comparison

### LangChain (abstraction layer, not orchestration)

**What it is**: Standardized interface over LLM APIs. Wraps model calls, tool definitions, output parsing.

**What you get**:
- `ChatOllama` instead of raw httpx calls
- `@tool` decorator for tool definitions with auto-generated JSON schemas
- Standardized message types (HumanMessage, AIMessage, ToolMessage)
- Output parsers for structured responses

**What you lose**:
- Direct control over HTTP payloads (e.g., qwen3 thinking mode `/think` needs workarounds)
- Simplicity — adds abstraction over something that's already simple (one HTTP POST)
- Dependency weight: pulls in pydantic, tenacity, jsonpatch, langsmith, etc.

**Verdict**: Useful if you plan to swap models frequently. Overhead if you're committed to Ollama + qwen3.

---

### LangGraph (graph-based state machine, built on LangChain)

**What it is**: Models agent workflows as a directed graph. Nodes = functions, edges = transitions, state = TypedDict flowing through the graph.

**Requires**: LangChain (specifically `langchain-core`)

**Genuine strengths for this project**:
- Checkpointing (save/resume state at any point) — useful for Phase 3+ human-in-the-loop
- `interrupt_before`/`interrupt_after` — pause execution for human review
- Built-in `ToolNode` for automatic tool execution loops
- State reducers (e.g., `add_messages`) for clean message accumulation
- LangSmith tracing for observability

**Weaknesses for this project**:
- Encodes a for-loop as a graph — over-engineered for the current control flow
- Forces you into LangChain's message/tool abstractions
- Heavy dependency tree (~20+ packages)
- API has changed significantly over time (churn risk)
- Features you'd pay for but not use: parallel execution, subgraphs, map-reduce, streaming through graph

**When it becomes clearly worth it**: Dynamic routing (LLM decides which agent speaks next), complex checkpoint/replay, parallel agent execution.

---

### AutoGen (Microsoft → ecosystem split)

**What it is**: Multi-agent conversation framework. `ConversableAgent` + `GroupChat` + `GroupChatManager`.

**Fit**: Actually the closest conceptual match — designed for agents having conversations, not completing tasks. The `GroupChatManager` handles turn-taking with customizable speaker selection.

**Critical problem**: Ecosystem split in 2024. Microsoft rewrote it as `autogen-agentchat` v0.4+ with a completely different API. The community forked the original as "AG2." Two competing versions, unclear winner. Betting on either is risky.

**Ollama support**: Via OpenAI-compatible config, but indirect.

**Verdict**: Right paradigm, wrong time. Ecosystem instability is a dealbreaker for applied research.

---

### CrewAI (role-based task pipelines)

**What it is**: Agents with roles complete tasks in sequence or hierarchy. "Researcher → Writer → Reviewer."

**Fundamental mismatch**: CrewAI is task-oriented. Your system is conversation-oriented. Your agents don't complete independent tasks — they take turns in a shared negotiation. Forcing negotiation into CrewAI's task model means fighting the framework at every step.

**Ollama support**: Exists but primarily designed for OpenAI/Anthropic models.

**Verdict**: Wrong paradigm. Solves a different problem.

---

### Smolagents (HuggingFace)

**What it is**: Lightweight single-agent tool-use framework.

**Problem**: Single-agent, not multi-agent. You'd build the orchestration yourself on top, at which point you might as well use plain Python.

**Verdict**: Wrong scope.

---

### Plain Python (extend current approach)

**What it is**: Keep procedural code, add tool calling and memory as plain functions.

**Strengths**:
- Zero framework overhead — every line serves the system
- Full control over Ollama payloads (thinking mode, tool format, etc.)
- No dependency risk or API churn
- Phase 2 adds ~150 lines, system stays under 600 lines total
- Easiest to debug and understand

**Weaknesses**:
- No built-in checkpointing (but `json.dump(state)` works at this scale)
- No human-in-the-loop primitives (but the system runs locally, so you can just add input() calls)
- If Phase 3+ needs complex orchestration, you'd retrofit or rewrite

**Verdict**: Best engineering choice for Phase 2. But doesn't teach you frameworks.

---

## Reference: Does Claude Code use any of these?

**No.** Claude Code (Anthropic's CLI) calls the Claude API directly with custom orchestration code. No LangChain, no LangGraph, no third-party agent framework. This demonstrates that sophisticated agent systems don't require frameworks — but Claude Code also has a dedicated engineering team maintaining that custom code.

---

## Decision Factors

| Factor | Plain Python | LangGraph | AutoGen | CrewAI |
|--------|-------------|-----------|---------|--------|
| Phase 2 fit | Best | Over-engineered | Decent | Poor |
| Phase 3+ fit | Needs retrofit | Strong | Uncertain (split) | Poor |
| Learning value | Low | High | Medium | Low |
| Dependency weight | 1 pkg (httpx) | ~20+ pkgs | ~15+ pkgs | ~15+ pkgs |
| Ollama/qwen3 compat | Full control | Via ChatOllama | Indirect | Indirect |
| Ecosystem stability | N/A | Active but churning | Split/unstable | Stable but niche |
| Checkpointing | DIY (easy at this scale) | Built-in | Limited | None |

---

## Open Questions

1. How important is the learning objective vs. shipping a clean Phase 2?
2. If we go with LangGraph, should we keep `multi_agent.py` as a plain-Python reference alongside?
3. Is there value in building Phase 2 in plain Python first, then doing a LangGraph port — so you understand exactly what the framework abstracts away?
