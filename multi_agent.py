#!/usr/bin/env python3
"""
Multi-Agent Itinerary Planning System
Collaborative negotiation between simulated travelers via mediated agents.
"""

import httpx
import json
import os
import sys
import time
import functools
from datetime import datetime

# Force unbuffered output so we can see progress in real time
print = functools.partial(print, flush=True)

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3:8b"
TIMEOUT = 300.0
ENABLE_LOGGING = True


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
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
# Actor Classes
# ─────────────────────────────────────────────────────────────────────────────

class Actor:
    """Base class wrapping Ollama API calls with a persona."""

    def __init__(self, name: str, system_prompt: str, use_thinking: bool = False,
                 temperature: float = 0.7):
        self.name = name
        self.system_prompt = system_prompt
        self.use_thinking = use_thinking
        self.temperature = temperature

    def respond(self, conversation_history: list[dict]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]

        for entry in conversation_history:
            messages.append({"role": entry["role"], "content": entry["content"]})

        # For thinking mode, prepend /think to the last user message
        if self.use_thinking and messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": "/think " + messages[-1]["content"],
            }

        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(OLLAMA_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            print("\n[ERROR] Cannot connect to Ollama at", OLLAMA_URL)
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"\n[ERROR] Ollama returned status {e.response.status_code}")
            print(e.response.text)
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            sys.exit(1)


class HumanSimulator(Actor):
    """Simulates a traveler stating preferences."""
    pass


class Agent(Actor):
    """Advocates for a human's preferences in negotiation."""
    pass


class Adjudicator(Actor):
    """Mediates between agents, proposes compromises, declares outcomes."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Manager
# ─────────────────────────────────────────────────────────────────────────────

class ConversationManager:
    """Orchestrates turn-taking and conversation flow."""

    SEPARATOR = "═" * 72

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.log_file = None

        self.human_a = HumanSimulator("Human A", HUMAN_A_PROMPT, temperature=0.7)
        self.human_b = HumanSimulator("Human B", HUMAN_B_PROMPT, temperature=0.7)
        self.agent_a = Agent("Agent A", AGENT_A_PROMPT, temperature=0.7)
        self.agent_b = Agent("Agent B", AGENT_B_PROMPT, temperature=0.7)
        self.adjudicator = Adjudicator(
            "Adjudicator", ADJUDICATOR_PROMPT, use_thinking=True, temperature=0.4
        )

        # Shared conversation history visible to all actors
        self.history: list[dict] = []

    def _add_message(self, speaker: str, content: str):
        """Add a labeled message to the shared history."""
        self.history.append({
            "role": "user",
            "content": f"[{speaker}]: {content}",
        })

    def _log(self, text: str):
        """Write text to the log file if logging is enabled."""
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def _display(self, speaker: str, content: str):
        """Pretty-print a speaker's message."""
        block = f"\n{self.SEPARATOR}\n  {speaker}\n{self.SEPARATOR}\n{content}\n"
        print(block)
        self._log(block)

    def _get_response(self, actor: Actor, prompt: str | None = None) -> str:
        """Get a response from an actor, optionally with an extra prompt."""
        history = list(self.history)
        if prompt:
            history.append({"role": "user", "content": prompt})
        response = actor.respond(history)
        return response

    def _print_and_log(self, text: str):
        """Print to stdout and log simultaneously."""
        print(text)
        self._log(text)

    def run_planning_session(self, trip_scenario: str) -> dict:
        """Run the full planning session and return a summary."""

        # Set up logging
        if ENABLE_LOGGING:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(output_dir, f"session_{timestamp}.log")
            self.log_file = open(log_path, "w", encoding="utf-8")
            print(f"[LOG] Writing session log to: {log_path}\n")

        header = (
            "\n" + "╔" + "═" * 70 + "╗\n"
            + "║" + " MULTI-AGENT ITINERARY PLANNER ".center(70) + "║\n"
            + "╚" + "═" * 70 + "╝\n"
            + f"\nScenario: {trip_scenario}\n"
        )
        self._print_and_log(header)

        # ── Phase 1: Setup ────────────────────────────────────────────────
        self._print_and_log("\n" + "─" * 72)
        self._print_and_log("  PHASE 1: PREFERENCE GATHERING")
        self._print_and_log("─" * 72)

        # Human A states preferences
        ha_response = self._get_response(
            self.human_a,
            f"A trip is being planned. Here is the scenario: {trip_scenario}\n"
            "Please state your preferences and constraints as a traveler."
        )
        self._display("Human A (Budget Backpacker)", ha_response)
        self._add_message("Human A (Budget Backpacker)", ha_response)

        # Human B states preferences
        hb_response = self._get_response(
            self.human_b,
            f"A trip is being planned. Here is the scenario: {trip_scenario}\n"
            "Please state your preferences and constraints as a traveler."
        )
        self._display("Human B (Luxury Seeker)", hb_response)
        self._add_message("Human B (Luxury Seeker)", hb_response)

        # ── Phase 2: Negotiation ──────────────────────────────────────────
        self._print_and_log("\n" + "─" * 72)
        self._print_and_log("  PHASE 2: NEGOTIATION")
        self._print_and_log("─" * 72)

        final_adjudicator_response = ""

        for round_num in range(1, self.max_rounds + 1):
            self._print_and_log(f"\n{'▸'*3} ROUND {round_num} of {self.max_rounds} {'◂'*3}")

            # Adjudicator frames the discussion
            if round_num == 1:
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

            adj_response = self._get_response(self.adjudicator, adj_prompt)
            self._display(f"Adjudicator (Round {round_num})", adj_response)
            self._add_message("Adjudicator", adj_response)

            # Agent A responds to the Adjudicator
            aa_response = self._get_response(
                self.agent_a,
                "Respond to the Adjudicator's latest framing. Present your position "
                "or counter-proposal on behalf of Human A (Budget Backpacker)."
            )
            self._display("Agent A (Budget Advocate)", aa_response)
            self._add_message("Agent A (Budget Advocate)", aa_response)

            # Agent B responds to the Adjudicator
            ab_response = self._get_response(
                self.agent_b,
                "Respond to the Adjudicator's latest framing. Present your position "
                "or counter-proposal on behalf of Human B (Luxury Seeker)."
            )
            self._display("Agent B (Luxury Advocate)", ab_response)
            self._add_message("Agent B (Luxury Advocate)", ab_response)

            final_adjudicator_response = adj_response

        # ── Phase 3: Resolution ───────────────────────────────────────────
        self._print_and_log("\n" + "─" * 72)
        self._print_and_log("  PHASE 3: RESOLUTION")
        self._print_and_log("─" * 72)

        resolution = self._get_response(
            self.adjudicator,
            "All negotiation rounds are complete. Declare the final outcome:\n"
            "- State whether consensus was reached, partial agreement, or impasse.\n"
            "- Provide a structured summary of what was agreed for each topic "
            "(accommodation, daily activities, dinners).\n"
            "- Note any unresolved disagreements."
        )
        self._display("Adjudicator (Final Resolution)", resolution)
        self._add_message("Adjudicator (Final Resolution)", resolution)

        result = {
            "scenario": trip_scenario,
            "rounds": self.max_rounds,
            "resolution": resolution,
            "full_history": self.history,
        }

        if self.log_file:
            self.log_file.close()
            self.log_file = None

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    scenario = (
        "Plan a 3-day trip to Barcelona. You need to agree on: "
        "(1) accommodation, (2) one must-do activity per day, "
        "(3) where to eat dinner each night."
    )

    manager = ConversationManager(max_rounds=3)
    result = manager.run_planning_session(scenario)

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " SESSION COMPLETE ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")


if __name__ == "__main__":
    main()
