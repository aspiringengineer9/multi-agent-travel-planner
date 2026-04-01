#!/usr/bin/env python3
"""
Multi-Agent Itinerary Planning System — LangGraph rewrite (Phase 1c)
Config-driven + per-actor memory: agents track private proposals across rounds;
adjudicator tracks position deltas; shared memory surfaces agreed items to all.
"""

import os
import re
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

print = functools.partial(print, flush=True)

MODEL = "qwen3:8b"


def _load_dotenv():
    """Load .env file from the same directory as this script into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


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
- Use the proposal_scorer tool to quantitatively compare competing proposals when agents are stuck.
- Frame specific questions or trade-offs for agents to respond to.
- Call for decisions when discussion stalls.

CONSENSUS DETECTION:
- If both agents have explicitly agreed on all negotiation topics, call the declare_outcome tool with status='consensus'.
- If agents have agreed on some topics but remain stuck on others after substantive discussion, call declare_outcome with status='partial_consensus'.
- If agents are repeating positions with no movement after 2+ rounds of the same arguments, call declare_outcome with status='impasse'.
- Do NOT declare consensus unless agents have EXPLICITLY agreed. Silence or lack of objection is not agreement.

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
                  all_agent_meta: dict, scenario_desc: str, topics: list,
                  deliberation_points: list | None = None,
                  travel_context: dict | None = None) -> str:
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

    delib_text = ""
    if deliberation_points:
        delib_text = "\nSpecific decisions to negotiate:\n" + "\n".join(
            f"  - {dp}" for dp in deliberation_points
        ) + "\n"

    return (
        f"=== NEGOTIATION SETUP ===\n"
        f"Scenario: {scenario_desc}\n"
        f"Topics under negotiation: {', '.join(topics)}\n"
        f"{delib_text}"
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
        f"7. Use the budget_calculator tool to verify whether proposals fit within your human's budget.\n"
        f"8. Use flight_search and hotel_search to ground your proposals in real options — "
        f"always search before recommending specific flights or hotels.\n"
        f"9. Use date_feasibility_check to verify travel dates before committing to them.\n"
        f"10. If a tool returns an error, retry with adjusted parameters (different date format, "
        f"airport code instead of city name, etc.) before falling back to estimates.\n"
        f"11. Keep responses to 2-3 paragraphs max.\n"
        f"\n"
        f"IMPORTANT: You are {agent_display}. Speak in your own voice based on "
        f"{human['name']}'s priorities. Do not copy or paraphrase other agents' positions."
        + (
            f"\n\n=== TRIP DETAILS ===\n"
            f"Destination: {travel_context.get('destination', 'N/A')}\n"
            f"Departure city: {travel_context.get('departure_city', 'N/A')}\n"
            f"Outbound: {travel_context.get('outbound_date', 'N/A')} "
            f"({travel_context.get('departure_airport', '')} → {travel_context.get('arrival_airport', '')})\n"
            f"Return: {travel_context.get('return_date', 'N/A')}\n"
            f"Hotel check-in: {travel_context.get('check_in', 'N/A')} / check-out: {travel_context.get('check_out', 'N/A')}"
            if travel_context else ""
        )
    )


def build_scenario(config: dict) -> dict:
    """Enrich raw config with generated system_prompt and display_name for every actor."""
    actors = config["actors"]
    humans = {k: v for k, v in actors.items() if v["role"] == "human"}
    scenario_desc = config["scenario"]
    topics = config.get("topics", [])
    deliberation_points = config.get("deliberation_points")
    travel_context = config.get("travel_context")

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
                agent_meta, scenario_desc, topics, deliberation_points, travel_context,
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
_events_file = None


def _log_event(event: dict) -> None:
    """Append a JSON-lines event to the events log for metric collection."""
    if _events_file:
        _events_file.write(json.dumps(event) + "\n")
        _events_file.flush()


def _display_and_log(speaker: str, content: str):
    block = f"\n{SEPARATOR}\n  {speaker}\n{SEPARATOR}\n{content}\n"
    print(block)
    if _log_file:
        _log_file.write(block + "\n")
        _log_file.flush()
    _log_event({"type": "turn", "speaker": speaker})


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
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def budget_calculator(items: list[dict], daily_budget: float, num_days: int = 1) -> str:
    """Calculate whether proposed itinerary items fit within a daily budget.

    Args:
        items: List of items, each with 'name' (str), 'cost_per_day' (float),
               and 'category' (str — e.g. accommodation, food, activity, transport).
        daily_budget: The daily budget cap in dollars.
        num_days: Number of trip days (default 1).
    """
    by_category: dict[str, float] = {}
    for item in items:
        cat = item.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + item.get("cost_per_day", 0)

    total_daily = sum(by_category.values())
    total_trip = total_daily * num_days
    budget_trip = daily_budget * num_days
    remaining = budget_trip - total_trip
    verdict = "WITHIN_BUDGET" if remaining >= 0 else "OVER_BUDGET"

    lines = [f"Daily cost breakdown:"]
    for cat, cost in sorted(by_category.items()):
        lines.append(f"  {cat}: ${cost:.2f}")
    lines.append(f"Total daily cost: ${total_daily:.2f}")
    lines.append(f"Daily budget: ${daily_budget:.2f}")
    lines.append(f"Remaining: ${remaining:.2f}")
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


@tool
def proposal_scorer(
    proposal: str,
    budget_a: float,
    budget_b: float | None,
    estimated_daily_cost: float,
    comfort_level: int,
    experience_richness: int,
) -> str:
    """Score a proposal on weighted criteria to help break deadlocks.

    Args:
        proposal: Brief description of the proposal being scored.
        budget_a: Human A's daily budget.
        budget_b: Human B's daily budget (None if flexible).
        estimated_daily_cost: Estimated daily cost of this proposal.
        comfort_level: 1-10 rating of comfort/luxury level.
        experience_richness: 1-10 rating of cultural/experiential value.
    """
    # Cost fit (40%): penalize if over strict budget, reward middle ground
    budgets = [b for b in [budget_a, budget_b] if b is not None]
    if budgets:
        min_budget = min(budgets)
        if estimated_daily_cost <= min_budget:
            cost_score = 10
        elif estimated_daily_cost <= min_budget * 1.2:
            cost_score = 6
        elif estimated_daily_cost <= min_budget * 1.5:
            cost_score = 3
        else:
            cost_score = 1
    else:
        cost_score = 8  # both flexible

    comfort_score = max(1, min(10, comfort_level))
    experience_score = max(1, min(10, experience_richness))

    weighted = (cost_score * 0.4 + comfort_score * 0.3 + experience_score * 0.3) * 10
    weighted = round(weighted, 1)

    return (
        f"Proposal: {proposal}\n"
        f"  Cost fit:    {cost_score}/10 (weight 40%) — est. ${estimated_daily_cost:.0f}/day\n"
        f"  Comfort:     {comfort_score}/10 (weight 30%)\n"
        f"  Experience:  {experience_score}/10 (weight 30%)\n"
        f"  TOTAL SCORE: {weighted}/100"
    )


@tool
def declare_outcome(status: str, reasoning: str) -> str:
    """Declare the negotiation outcome. Call this when consensus is reached or impasse is clear.

    Args:
        status: One of 'consensus', 'partial_consensus', 'impasse'.
        reasoning: Brief explanation of why this outcome was declared.
    """
    valid = {"consensus", "partial_consensus", "impasse"}
    if status not in valid:
        return f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid))}"
    return f"Outcome declared: {status}. Reason: {reasoning}"


@tool
def flight_search(origin: str, destination: str, outbound_date: str, return_date: str | None = None) -> str:
    """Search for available flights using real-time data.

    Args:
        origin: Departure city or airport code (e.g. 'New York' or 'JFK').
        destination: Arrival city or airport code (e.g. 'Barcelona' or 'BCN').
        outbound_date: Departure date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format (omit for one-way).

    Returns:
        Top 3 flight options with airline, price, duration, and stops.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return "Tool error: SERPAPI_KEY not set in environment."
    try:
        from serpapi import GoogleSearch
        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": outbound_date,
            "currency": "USD",
            "hl": "en",
            "api_key": api_key,
            "type": "1" if return_date else "2",
        }
        if return_date:
            params["return_date"] = return_date
        results = GoogleSearch(params).get_dict()
        if "error" in results:
            return f"Flight search error: {results['error']}"
        flights = results.get("best_flights") or results.get("other_flights") or []
        if not flights:
            return f"No flights found from {origin} to {destination} on {outbound_date}."
        lines = [f"Flights from {origin} to {destination} on {outbound_date}:"]
        for i, group in enumerate(flights[:3], 1):
            legs = group.get("flights", [{}])
            airline = legs[0].get("airline", "Unknown") if legs else "Unknown"
            price = group.get("price", "N/A")
            duration = group.get("total_duration", "N/A")
            stops = len(legs) - 1
            stop_text = "nonstop" if stops == 0 else f"{stops} stop(s)"
            lines.append(f"  {i}. {airline} — ${price} — {duration} min — {stop_text}")
        return "\n".join(lines)
    except Exception as e:
        return f"Flight search error: {e}"


@tool
def hotel_search(destination: str, check_in: str, check_out: str, max_price_per_night: float | None = None) -> str:
    """Search for available hotels at the destination.

    Args:
        destination: City or area to search hotels in (e.g. 'Barcelona, Spain').
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        max_price_per_night: Optional maximum price per night in USD.

    Returns:
        Top 3 hotel options with name, price per night, rating, and amenities.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return "Tool error: SERPAPI_KEY not set in environment."
    try:
        from serpapi import GoogleSearch
        params = {
            "engine": "google_hotels",
            "q": f"hotels in {destination}",
            "check_in_date": check_in,
            "check_out_date": check_out,
            "currency": "USD",
            "hl": "en",
            "api_key": api_key,
        }
        results = GoogleSearch(params).get_dict()
        if "error" in results:
            return f"Hotel search error: {results['error']}"
        properties = results.get("properties", [])
        if not properties:
            return f"No hotels found in {destination} for {check_in} to {check_out}."
        if max_price_per_night:
            filtered = [
                p for p in properties
                if (p.get("rate_per_night") or {}).get("extracted_lowest", float("inf")) <= max_price_per_night
            ]
            if filtered:
                properties = filtered
        lines = [f"Hotels in {destination} ({check_in} to {check_out}):"]
        for i, hotel in enumerate(properties[:3], 1):
            name = hotel.get("name", "Unknown hotel")
            rate = (hotel.get("rate_per_night") or {}).get("lowest", "N/A")
            rating = hotel.get("overall_rating", "N/A")
            amenities = ", ".join((hotel.get("amenities") or [])[:3]) or "N/A"
            lines.append(f"  {i}. {name} — {rate}/night — {rating}★ — {amenities}")
        return "\n".join(lines)
    except Exception as e:
        return f"Hotel search error: {e}"


@tool
def date_feasibility_check(outbound_date: str, return_date: str, constraints: list[str] | None = None) -> str:
    """Check whether proposed travel dates are feasible.

    Args:
        outbound_date: Departure date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format.
        constraints: Optional list of blocked date ranges, e.g. ['2025-09-01 to 2025-09-05'].

    Returns:
        FEASIBLE: <summary> or INFEASIBLE: <reason>
    """
    try:
        from datetime import date as date_type
        outbound = datetime.strptime(outbound_date, "%Y-%m-%d").date()
        ret = datetime.strptime(return_date, "%Y-%m-%d").date()
        today = date_type.today()
        if outbound <= today:
            return "INFEASIBLE: Outbound date must be in the future."
        if ret < outbound:
            return "INFEASIBLE: Return date must be on or after outbound date."
        duration = (ret - outbound).days
        if duration > 30:
            return f"INFEASIBLE: Trip of {duration} days exceeds 30-day limit."
        if constraints:
            for constraint in constraints:
                parts = constraint.split(" to ")
                if len(parts) == 2:
                    try:
                        block_start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").date()
                        block_end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").date()
                        if not (ret < block_start or outbound > block_end):
                            return f"INFEASIBLE: Dates overlap with blocked period '{constraint}'."
                    except ValueError:
                        pass
        return f"FEASIBLE: {duration}-day trip from {outbound_date} to {return_date}."
    except ValueError as e:
        return f"INFEASIBLE: Invalid date format — {e}. Use YYYY-MM-DD."


AGENT_TOOLS = [budget_calculator, flight_search, hotel_search, date_feasibility_check]
ADJUDICATOR_TOOLS = [proposal_scorer, declare_outcome]

_AGENT_TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}
_ADJUDICATOR_TOOLS_BY_NAME = {t.name: t for t in ADJUDICATOR_TOOLS}


def _build_retry_hint(tool_name: str, result_str: str, args: dict) -> str | None:
    """Return a targeted, actionable retry hint for a failed tool call, or None if not applicable.

    The hint is appended to the ToolMessage so the LLM sees exactly what to change on the
    next ReAct iteration — deterministic error classification, LLM-driven corrective action.
    """
    low = result_str.lower()

    if tool_name == "hotel_search":
        if "no hotels found" in low:
            dest = args.get("destination", "")
            return (
                f"[RETRY HINT] No results for '{dest}'. Try: "
                f"(1) append country e.g. '{dest}, Spain' or '{dest}, USA', "
                f"(2) use a broader area name, "
                f"(3) remove max_price_per_night filter if set."
            )
        if "hotel search error" in low or "tool error" in low:
            return (
                "[RETRY HINT] API error on hotel_search. Try: "
                "(1) use 'City, Country' format for destination, "
                "(2) confirm dates are YYYY-MM-DD and in the future."
            )

    if tool_name == "flight_search":
        if "no flights found" in low:
            origin = args.get("origin", "")
            dest = args.get("destination", "")
            return (
                f"[RETRY HINT] No flights for '{origin}'→'{dest}'. Try: "
                f"(1) use IATA airport codes instead of city names "
                f"(e.g. 'JFK' not 'New York', 'BCN' not 'Barcelona', 'LAX' not 'Los Angeles'), "
                f"(2) try a nearby major airport."
            )
        if "flight search error" in low or "tool error" in low:
            return (
                "[RETRY HINT] API error on flight_search. Try: "
                "(1) use IATA airport codes (JFK, LAX, BCN, LHR, ORD, MCO, SFO, RNO), "
                "(2) confirm date format is YYYY-MM-DD."
            )

    if tool_name == "date_feasibility_check":
        if "outbound date must be in the future" in low:
            return (
                "[RETRY HINT] Dates are in the past. Use the future dates shown in your "
                "trip details (outbound_date / return_date fields in your system prompt)."
            )
        if "invalid date format" in low:
            return (
                "[RETRY HINT] Wrong date format. Use YYYY-MM-DD "
                "(e.g. '2026-06-10', not '06/10/2026' or 'June 10, 2026')."
            )

    return None


def _execute_tool_calls(response: AIMessage, tools_by_name: dict,
                         _ctx_agent: str = "", _ctx_round: int = 0) -> list[ToolMessage]:
    """Extract and execute tool calls from an AIMessage. Returns list of ToolMessage."""
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return []
    results = []
    for tc in response.tool_calls:
        tool_fn = tools_by_name.get(tc["name"])
        if tool_fn:
            try:
                result = tool_fn.invoke(tc["args"])
                result_str = str(result)
                success = not result_str.lower().startswith(
                    ("tool error:", "flight search error:", "hotel search error:", "infeasible:")
                )
                _log_event({"type": "tool_call", "agent": _ctx_agent, "tool": tc["name"],
                            "round": _ctx_round, "success": success})
                if not success:
                    hint = _build_retry_hint(tc["name"], result_str, tc.get("args") or {})
                    if hint:
                        result_str = result_str + "\n" + hint
                        _log_event({"type": "retry_hint", "agent": _ctx_agent,
                                    "tool": tc["name"], "round": _ctx_round})
                results.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
            except Exception as e:
                _log_event({"type": "tool_call", "agent": _ctx_agent, "tool": tc["name"],
                            "round": _ctx_round, "success": False, "error": str(e)})
                results.append(ToolMessage(content=f"Tool error: {e}", tool_call_id=tc["id"]))
        else:
            results.append(ToolMessage(
                content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]
            ))
    return results


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


def _extract_round_summary(text: str) -> str:
    """Parse AGREED/DISAGREED/PROPOSED sections from adjudicator response.

    Falls back to storing the full response if structured sections aren't found.
    """
    sections = {}
    for header in ("AGREED", "DISAGREED", "PROPOSED"):
        match = re.search(
            rf"{header}\s*:\s*(.+?)(?=\n(?:AGREED|DISAGREED|PROPOSED)\s*:|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if match:
            sections[header] = match.group(1).strip()
    if sections:
        return "\n".join(f"{k}: {v}" for k, v in sections.items())
    return text  # fallback: store full response


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def _static_human_statement(actor: dict) -> str:
    """Build a preference statement directly from config — no LLM call needed."""
    parts = [actor.get("personality", "").strip()]
    budget = actor.get("budget")
    if budget:
        parts.append(f"My budget is strictly ${budget}/day.")
    else:
        parts.append("My budget is flexible — comfort matters more than cost.")
    return " ".join(parts)


def human_a_node(state: NegotiationState) -> dict:
    _print_and_log("\n" + "─" * 72)
    _print_and_log("  PHASE 1: PREFERENCE GATHERING")
    _print_and_log("─" * 72)

    actor = state["scenario"]["actors"]["human_a"]
    statement = _static_human_statement(actor)
    _display_and_log(actor["display_name"], statement)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {statement}")
    return {"messages": [labeled]}


def human_b_node(state: NegotiationState) -> dict:
    actor = state["scenario"]["actors"]["human_b"]
    statement = _static_human_statement(actor)
    _display_and_log(actor["display_name"], statement)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {statement}")
    return {"messages": [labeled]}


def adjudicator_loop_node(state: NegotiationState) -> dict:
    round_num = state["round"]
    max_rounds = state["max_rounds"]
    actor = state["scenario"]["actors"]["adjudicator"]
    topics = ", ".join(state["scenario"].get("topics", []))
    delib_points = state["scenario"].get("deliberation_points")

    if round_num == 0:
        _print_and_log("\n" + "─" * 72)
        _print_and_log("  PHASE 2: NEGOTIATION")
        _print_and_log("─" * 72)

    _print_and_log(f"\n{'▸'*3} ROUND {round_num + 1} of {max_rounds} {'◂'*3}")

    if round_num == 0:
        delib_text = ""
        if delib_points:
            delib_text = "\nSpecific decisions to resolve:\n" + "\n".join(
                f"  - {dp}" for dp in delib_points
            ) + "\n"
        adj_prompt = (
            "You have heard both travelers' preferences. "
            f"Topics to negotiate: {topics}. "
            f"{delib_text}"
            "Frame the first key trade-off or question for the agents to discuss. "
            "Focus on one topic at a time.\n\n"
            "End your response with a status block in this exact format:\n"
            "AGREED: [what both agents explicitly agreed on, or 'Nothing yet']\n"
            "DISAGREED: [key sticking points still unresolved]\n"
            "PROPOSED: [your suggested compromise or next step]"
        )
    elif round_num == max_rounds - 1:
        adj_prompt = (
            "This is the FINAL round. You must now declare an outcome.\n\n"
            "1. Optionally call proposal_scorer to evaluate the best compromise on the table.\n"
            "2. Then call declare_outcome with the appropriate status:\n"
            "   - 'consensus' if a viable compromise was reached (agents converging, even implicitly)\n"
            "   - 'partial_consensus' if some topics resolved but others remain stuck\n"
            "   - 'impasse' if positions are irreconcilable\n"
            "You MUST call declare_outcome — do not end this round without it.\n\n"
            "End your response with a status block in this exact format:\n"
            "AGREED: [what was resolved]\n"
            "DISAGREED: [what remains unresolved, or 'None']\n"
            "PROPOSED: [the final agreed plan, or best available compromise]"
        )
    else:
        adj_prompt = (
            "Based on the agents' responses, synthesize their positions. "
            "If progress was made, move to the next topic. "
            "If stuck, propose a creative compromise or use the proposal_scorer tool to compare options.\n\n"
            "End your response with a status block in this exact format:\n"
            "AGREED: [what both agents explicitly agreed on, or 'Nothing yet']\n"
            "DISAGREED: [key sticking points still unresolved]\n"
            "PROPOSED: [your suggested compromise or next step]"
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
    response = _llms["adjudicator"].invoke(msgs)

    # Inline tool execution: handle proposal_scorer and declare_outcome calls
    declared_status = None
    tool_msgs = _execute_tool_calls(response, _ADJUDICATOR_TOOLS_BY_NAME,
                                     _ctx_agent="adjudicator", _ctx_round=round_num + 1)
    if tool_msgs:
        for tc in (response.tool_calls or []):
            if tc["name"] == "declare_outcome":
                declared_status = tc["args"].get("status")
                _print_and_log(f"  [TOOL] declare_outcome → {declared_status}: {tc['args'].get('reasoning', '')}")
            else:
                _print_and_log(f"  [TOOL] {tc['name']} called")

    # Path A: if tools fired or content is empty, re-invoke adjudicator_think with tool results in context
    if tool_msgs or not response.content or not response.content.strip():
        msgs_with_tools = msgs + [response] + tool_msgs
        response = _llms["adjudicator_think"].invoke(msgs_with_tools)

    _display_and_log(f"{actor['display_name']} (Round {round_num + 1})", response.content)
    labeled = HumanMessage(content=f"[{actor['display_name']}]: {response.content}")

    synthesis_key = f"round_{round_num + 1}_synthesis"
    updated_adj_memory = memory_write(adj_mem, synthesis_key, response.content)

    # Parse structured round summary from adjudicator's response (no extra LLM call)
    summary = _extract_round_summary(response.content)
    updated_shared = memory_write(shared, synthesis_key, summary)

    result = {
        "messages": [labeled],
        "round": round_num + 1,
        "adjudicator_memory": updated_adj_memory,
        "shared_memory": updated_shared,
    }
    if declared_status:
        result["status"] = declared_status
    return result


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

        # ReAct loop: reason → act → observe → repeat until no more tool calls
        MAX_REACT_ITERS = 4
        messages = msgs
        prev_failed_tools: set[str] = set()
        for iteration in range(MAX_REACT_ITERS):
            response = _llms[agent_key].invoke(messages)
            tool_msgs = _execute_tool_calls(response, _AGENT_TOOLS_BY_NAME,
                                             _ctx_agent=agent_key, _ctx_round=round_num)
            if not tool_msgs:
                break
            for tc, tm in zip(response.tool_calls or [], tool_msgs):
                _print_and_log(
                    f"  [ReAct {iteration + 1}] {actor['display_name']} → {tc['name']}"
                )
                result_lower = tm.content.lower()
                failed_now = result_lower.startswith(
                    ("tool error:", "no hotels", "no flights found", "infeasible:",
                     "flight search error:", "hotel search error:")
                )
                if not failed_now and tc["name"] in prev_failed_tools:
                    _log_event({"type": "retry_success", "agent": agent_key,
                                "tool": tc["name"], "round": round_num})
                    _print_and_log(
                        f"  [ReAct ✓] {actor['display_name']} retried {tc['name']} → success"
                    )
                if failed_now:
                    prev_failed_tools.add(tc["name"])
                else:
                    prev_failed_tools.discard(tc["name"])
            messages = messages + [response] + tool_msgs

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

    # If status was already set by declare_outcome, preserve it; otherwise determine from response
    current_status = state["status"]
    status_context = ""
    if current_status in ("consensus", "partial_consensus", "impasse"):
        status_context = f"\nThe negotiation was concluded with status: {current_status}.\n"

    resolution_prompt = (
        "All negotiation rounds are complete. Declare the final outcome:\n"
        f"{status_context}"
        "- State whether consensus was reached, partial agreement, or impasse.\n"
        "- Provide a structured summary of what was agreed for each topic.\n"
        "- Note any unresolved disagreements."
        + (f"\n\n{memory_context}" if memory_context else "")
    )
    msgs = build_prompt_messages(actor["system_prompt"], state, resolution_prompt)
    # Use adjudicator_think (no tools bound) so /think prefix works for deeper reasoning
    msgs = inject_think_prefix(msgs)
    response = _llms["adjudicator_think"].invoke(msgs)

    _display_and_log(f"{actor['display_name']} (Final Resolution)", response.content)
    labeled = HumanMessage(
        content=f"[{actor['display_name']} (Final Resolution)]: {response.content}"
    )

    # Use declared status if available; otherwise infer from response text
    if current_status not in ("consensus", "partial_consensus", "impasse"):
        text_lower = response.content.lower()
        if "consensus" in text_lower and "partial" not in text_lower:
            final_status = "consensus"
        elif "impasse" in text_lower:
            final_status = "impasse"
        else:
            final_status = "partial_consensus"
    else:
        final_status = current_status

    return {"messages": [labeled], "status": final_status}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def after_adjudicator(state: NegotiationState) -> list:
    """Fan-out: send identical state snapshots to all agents in parallel, or route to resolution."""
    # Early termination: adjudicator declared outcome via declare_outcome tool
    if state["status"] in ("consensus", "partial_consensus", "impasse"):
        _print_and_log(f"  ⇒ Early termination: {state['status']}")
        return [Send("resolution", state)]
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

def build_graph(scenario: dict, checkpointer=None):
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

    interrupt_after = ["adjudicator_loop"] if checkpointer is not None else []
    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_session(config_path: str) -> dict:
    global _llms, _log_file, _events_file

    config = load_config(config_path)
    scenario = build_scenario(config)
    max_rounds = scenario.get("max_rounds", 3)

    # Enable parallel Ollama requests — M3 Air 16GB can handle 2 concurrent requests
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "2")

    _llms = {}
    for key, actor in scenario["actors"].items():
        if key == "adjudicator":
            adj_base = ChatOllama(model=MODEL, temperature=0.4)
            _llms[key] = adj_base.bind_tools(ADJUDICATOR_TOOLS)  # tools, no /think
            _llms["adjudicator_think"] = adj_base                 # /think, no tools
        elif actor["role"] == "agent":
            base = ChatOllama(model=MODEL, temperature=0.7)
            _llms[key] = base.bind_tools(AGENT_TOOLS)
        else:
            _llms[key] = ChatOllama(model=MODEL, temperature=0.7)

    started_at = datetime.now()

    if ENABLE_LOGGING:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = started_at.strftime("%Y-%m-%d_%H%M%S")
        log_path = os.path.join(output_dir, f"session_{timestamp}.log")
        _log_file = open(log_path, "w", encoding="utf-8")
        _events_file = open(log_path.replace(".log", "_events.jsonl"), "w", encoding="utf-8")
        print(f"[LOG] Writing session log to: {log_path}\n")
        _log_session_header(config_path, scenario, started_at)
        _log_event({"type": "session_start", "scenario": os.path.basename(config_path),
                    "timestamp": started_at.isoformat(), "max_rounds": max_rounds})

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

    _AGREEMENT_SCORES = {"consensus": 1.0, "partial_consensus": 0.5, "impasse": 0.0}
    _log_event({
        "type": "session_end",
        "status": final_state.get("status", "unknown"),
        "rounds": final_state.get("round", 0),
        "max_rounds": max_rounds,
        "duration_seconds": round((datetime.now() - started_at).total_seconds(), 1),
        "agreement_score": _AGREEMENT_SCORES.get(final_state.get("status", ""), 0.0),
        "scenario": os.path.basename(config_path),
    })

    if _log_file:
        _log_file.close()
        _log_file = None
    if _events_file:
        _events_file.close()
        _events_file = None

    return final_state


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scenarios", "scenario_la.yaml"
    )
    run_session(config_path)

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " SESSION COMPLETE ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")


if __name__ == "__main__":
    main()
