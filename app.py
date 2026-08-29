"""
Khidmah AI — an AI operations manager for mosques and Islamic community organizations.

Core insight (verified, see research/notes.md): American mosques aren't missing dashboards,
they're missing staff — 76% are run entirely by volunteers (ISPU American Mosque Survey 2020).
So this doesn't try to be another place to enter events/announcements/donations. It tries to
actually DO the operational work an admin would otherwise have to do by hand.

Architecture — this is genuinely agentic, not a fixed pipeline:
  A Planner agent (claude-sonnet-4-6) reads the admin's goal and DECIDES for itself, via real
  tool-use, which specialist agents this specific goal needs and in what order:
    - run_event_agent       -> logistics: schedule, checklist, budget, supplies
    - run_volunteer_agent   -> assigns real volunteers (from the roster) to the roles the
                               event agent identified
    - run_comms_agent       -> drafts WhatsApp/email/Instagram/volunteer-thankyou copy
  A simple announcement might only need run_comms_agent; a full event usually needs all three,
  called in sequence with each one's output feeding the next. The planner is not told a fixed
  order in code — it decides, and you can see it deciding in the live status log.

  Once the planner is done, a separate Audit agent (claude-sonnet-4-6) reviews the *combined*
  plan for the specific gaps volunteer-run mosques hit in real life (salah-time conflicts,
  accessibility, parental consent for youth events, unassigned roles, unrealistic budgets) —
  agent-to-agent review, not the planner grading its own homework.

Cost/model strategy: the two steps that are real judgment calls (deciding what work is needed,
and critiquing the finished plan) use the stronger model. The three specialist agents, once
told exactly what to produce, use the cheap model — that's most of the tool calls in a typical
run, which is where the cost savings actually matter.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # then paste your real ANTHROPIC_API_KEY into .env
    streamlit run app.py

Deploy: push this repo to GitHub (public), then on share.streamlit.io point at it.
Set ANTHROPIC_API_KEY in that app's Settings -> Secrets instead of committing .env.
"""

import os
import streamlit as st
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# --- Model choice is the whole "cost & model strategy" story for judges. ---
CHEAP_MODEL = "claude-haiku-4-5-20251001"     # specialist agents: execution, once told what to do
STRONG_MODEL = "claude-sonnet-4-6"            # planner + audit: the actual judgment calls
# Double-check these are still current before you rely on them:
# https://platform.claude.com/docs/en/about-claude/models/overview

PRICE_PER_MTOK = {
    CHEAP_MODEL: {"input": 1.00, "output": 5.00},
    STRONG_MODEL: {"input": 3.00, "output": 15.00},
}

# Single shared ceiling for every API call in the app (planner, each specialist, audit) —
# raise this one constant, not individual call sites, if a response still truncates.
MAX_TOKENS = 4096

TRUNCATION_WARNING = (
    "\n\n⚠️ **This response was cut off (hit the model's output limit) — consider a "
    "shorter goal or a smaller event.**"
)


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found. Locally: put it in a .env file. "
            "On Streamlit Cloud: add it under Settings -> Secrets."
        )
        st.stop()
    # accept-encoding: gzip works around a brotli-decompression bug some environments hit
    # with the default client — same fix used in Halal Scout, carried over here.
    # timeout/max_retries are explicit (not just relying on the SDK defaults) so a hung
    # request fails in ~60s instead of the SDK's 10-minute default read timeout.
    return Anthropic(
        api_key=api_key,
        default_headers={"accept-encoding": "gzip"},
        timeout=60.0,
        max_retries=2,
    )


class AgentCallError(Exception):
    """A clean, user-facing message for a failed Anthropic API call — raised once the
    SDK's own retries (rate limits / 5xx / 529 overloaded) are exhausted, or immediately
    for a non-retryable failure like a bad API key. Never contains a raw traceback."""


def _clean_api_error(exc: Exception, step_label: str) -> AgentCallError:
    if isinstance(exc, anthropic.AuthenticationError):
        return AgentCallError(
            f"{step_label} failed: the Anthropic API key was rejected. "
            "Check ANTHROPIC_API_KEY and try again."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AgentCallError(f"{step_label} failed: this API key doesn't have permission for this request.")
    if isinstance(exc, anthropic.APIConnectionError):
        return AgentCallError(
            f"{step_label} failed: couldn't reach the Anthropic API (network issue or timeout). "
            "Check your connection and try again."
        )
    if isinstance(exc, anthropic.APIStatusError):
        return AgentCallError(
            f"{step_label} failed: the Anthropic API returned an error ({exc.status_code}). "
            "Try again in a moment."
        )
    return AgentCallError(f"{step_label} failed unexpectedly ({type(exc).__name__}). Try again in a moment.")


def create_message(client: Anthropic, step_label: str, **kwargs):
    """The one choke point every Anthropic API call in the app goes through: applies the
    shared MAX_TOKENS ceiling, converts SDK errors into a clean AgentCallError (see
    _clean_api_error), and reports whether the response was truncated (stop_reason ==
    "max_tokens") so callers can warn instead of silently treating cut-off text as
    complete."""
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.AnthropicError as exc:
        raise _clean_api_error(exc, step_label) from exc
    return resp, resp.stop_reason == "max_tokens"


def call_model(client: Anthropic, model: str, system: str, user: str, step_label: str = "Agent call") -> tuple[str, dict]:
    resp, truncated = create_message(
        client, step_label, model=model, system=system, messages=[{"role": "user", "content": user}]
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    if truncated:
        text += TRUNCATION_WARNING
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "model": model}
    return text, usage


def estimate_cost(usages: list[dict]) -> float:
    total = 0.0
    for u in usages:
        p = PRICE_PER_MTOK.get(u["model"])
        if not p:
            continue
        total += (u["input_tokens"] / 1_000_000) * p["input"]
        total += (u["output_tokens"] / 1_000_000) * p["output"]
    return total


# --- Specialist agents (called as tools by the planner) --------------------------------

EVENT_AGENT_SYSTEM = """You are the logistics specialist for a mosque's operations team. Given \
an event goal, produce: a schedule/timeline, a task checklist (a bullet list, including things \
people forget: setup, cleanup, first aid, prayer/salah scheduling around the event, \
accessibility), a budget breakdown that fits any stated budget, and a supplies list. Be \
concrete and mosque-specific — assume volunteer labor, not paid staff, unless told otherwise. \
End with a short line listing the distinct ROLES that will need volunteers, so another agent \
can staff them."""

VOLUNTEER_AGENT_SYSTEM = """You are the volunteer coordination specialist. Given a list of \
roles/shifts needed and a roster of available volunteers with their availability and skills, \
assign volunteers to roles sensibly, matching skills to tasks where possible. If there are not \
enough volunteers for a role, say so explicitly rather than inventing people who aren't on the \
roster. Output as a clear list: ROLE — assigned volunteer(s) — any gap/shortage noted."""

COMMS_AGENT_SYSTEM = """You draft community communications for a mosque. Given an event \
summary, produce four short pieces, clearly labeled: WHATSAPP (casual, brief, with an Islamic \
greeting), EMAIL (slightly more formal, with a subject line), INSTAGRAM CAPTION (short, with \
2-4 relevant hashtags), VOLUNTEER THANK-YOU MESSAGE (warm, brief). Keep each under 80 words."""

TOOLS = [
    {
        "name": "run_event_agent",
        "description": (
            "Plan the logistics for an event or program: schedule/timeline, task checklist, "
            "budget breakdown, and supplies needed. Call this whenever the goal involves "
            "organizing an event, program, or gathering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_goal": {
                    "type": "string",
                    "description": "The event goal and any known details (headcount, budget, date, etc.)",
                }
            },
            "required": ["event_goal"],
        },
    },
    {
        "name": "run_volunteer_agent",
        "description": (
            "Assign volunteers to roles/shifts for an event, based on their availability and "
            "skills. Call this whenever the goal needs people coordinated, not just planned — "
            "usually after run_event_agent has identified what roles are needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks_needed": {
                    "type": "string",
                    "description": "The roles/shifts that need volunteers, e.g. from the event agent's checklist",
                },
                "volunteer_roster": {
                    "type": "string",
                    "description": "The list of available volunteers with their availability and skills",
                },
            },
            "required": ["tasks_needed", "volunteer_roster"],
        },
    },
    {
        "name": "run_comms_agent",
        "description": (
            "Draft community communications for an event: WhatsApp announcement, email, "
            "Instagram caption, and a volunteer thank-you message. Call this whenever the "
            "community or volunteers need to be informed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_summary": {
                    "type": "string",
                    "description": "A summary of the event/program to communicate about",
                }
            },
            "required": ["event_summary"],
        },
    },
]

STEP_LABELS = {
    "run_event_agent": "📅 Event/Logistics Agent — building the schedule, checklist, and budget...",
    "run_volunteer_agent": "🧑‍🤝‍🧑 Volunteer Coordinator Agent — assigning roles from the roster...",
    "run_comms_agent": "📣 Communications Agent — drafting WhatsApp / email / Instagram copy...",
}


def execute_tool(client: Anthropic, name: str, tool_input: dict) -> tuple[str, dict]:
    def require(field: str) -> str:
        value = tool_input.get(field)
        if value is None:
            raise AgentCallError(f"{name} was called without the required '{field}' field — can't proceed.")
        return value

    if name == "run_event_agent":
        return call_model(
            client, CHEAP_MODEL, EVENT_AGENT_SYSTEM, require("event_goal"), step_label="Event/logistics agent"
        )
    if name == "run_volunteer_agent":
        user_msg = (
            f"Roles/shifts needed:\n{require('tasks_needed')}\n\n"
            f"Available volunteers:\n{require('volunteer_roster')}"
        )
        return call_model(client, CHEAP_MODEL, VOLUNTEER_AGENT_SYSTEM, user_msg, step_label="Volunteer coordinator agent")
    if name == "run_comms_agent":
        return call_model(
            client, CHEAP_MODEL, COMMS_AGENT_SYSTEM, require("event_summary"), step_label="Communications agent"
        )
    raise AgentCallError(f"Planner requested an unknown tool: {name}")


# --- Planner (orchestrator) — this is the actual agentic loop -------------------------------

PLANNER_SYSTEM = """You are the operations planner for Khidmah AI, an AI operations manager \
for mosques and Islamic community organizations. An admin will describe something their \
community needs — an event, a program, an announcement, a volunteer effort.

You have three specialist agents available as tools:
- run_event_agent: logistics, scheduling, checklists, budget, supplies
- run_volunteer_agent: assigns people to roles/shifts (needs a task list AND the volunteer roster)
- run_comms_agent: drafts announcements/messages once the event/program is defined

Decide for yourself which of these THIS SPECIFIC goal actually needs — not every goal needs \
all three. A pure announcement about something already planned might only need \
run_comms_agent. A full new event usually needs run_event_agent first (so you know what roles \
exist), then run_volunteer_agent (using those roles + the roster you were given), then \
run_comms_agent last (once the event is fully defined). Call one tool at a time and use each \
result to inform the next call's input.

Once you have everything the goal requires, STOP calling tools and instead write ONE complete, \
well-organized final operations plan for the admin, with headings: Overview, Task Checklist, \
Volunteer Assignments (only if you ran the volunteer agent), Communications Drafts (only if you \
ran the comms agent). Never invent volunteer names that weren't in the roster you were given."""

AUDIT_SYSTEM = """You are the audit/review agent for a mosque operations plan. Given a \
complete plan (checklist, volunteer assignments, communications), review it critically for \
gaps commonly missed by volunteer-run mosques: conflicts with prayer/salah times, \
accessibility needs, parental consent for youth-focused events, insufficient cleanup crew, a \
budget that doesn't add up or is unrealistic for the stated amount, food-safety/allergy notes, \
and any role mentioned in the checklist that was never actually assigned a volunteer. Output a \
short bullet list of flags (⚠ each one), or state clearly "✅ No major gaps found" if there \
truly are none — don't invent issues to seem thorough. End with exactly one line: \
"READY TO PUBLISH: Yes" or "READY TO PUBLISH: Needs attention"."""


def run_orchestrator(client: Anthropic, goal: str, volunteer_roster: str, on_step=None):
    """The planner decides which specialist agents to call, in what order, via real tool-use.
    Returns (final_plan_text, tool_call_log, usages)."""
    usages: list[dict] = []
    tool_log: list[tuple[str, dict, str]] = []
    messages = [
        {
            "role": "user",
            "content": f"Community goal: {goal}\n\nAvailable volunteers:\n{volunteer_roster}",
        }
    ]

    for _ in range(6):  # safety cap so a confused model can't loop forever
        resp, truncated = create_message(
            client, "Planner agent", model=STRONG_MODEL, system=PLANNER_SYSTEM, tools=TOOLS, messages=messages
        )
        usages.append(
            {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "model": STRONG_MODEL}
        )

        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            if truncated:
                final_text += TRUNCATION_WARNING
            return final_text, tool_log, usages

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if on_step:
                on_step(block.name, block.input)
            result_text, usage = execute_tool(client, block.name, block.input)
            usages.append(usage)
            tool_log.append((block.name, block.input, result_text))
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
            )
        messages.append({"role": "user", "content": tool_results})

    return "The planner hit its step limit — showing what it gathered so far.", tool_log, usages


def audit_step(client: Anthropic, plan_text: str) -> tuple[str, dict]:
    return call_model(client, STRONG_MODEL, AUDIT_SYSTEM, plan_text, step_label="Audit agent")


def escape_markdown_dollars(text: str) -> str:
    """Streamlit's markdown renders "$...$" as LaTeX math, which mangles agent-generated
    budget figures like "$3,000 for 300 guests ($10/person)". Escape literal dollar
    signs before display so they render as plain text instead."""
    return (text or "").replace("$", r"\$")


def is_finished_plan(tool_log: list, final_text: str) -> bool:
    """True only if the planner actually finished a plan, rather than stopping to ask a
    clarifying question (e.g. a vague goal, or a full event with no volunteer roster to
    work with). PLANNER_SYSTEM requires a finished plan to open with an "Overview"
    section, so its absence — combined with the text trailing off into a question, the
    way a request for more info does — is a reasonably reliable signal the planner isn't
    done yet. Auditing a clarifying question instead of a plan produces a confusing,
    useless review, so this errs toward treating ambiguous output as unfinished."""
    if not tool_log:
        return False
    text = (final_text or "").strip()
    if not text:
        return False
    mentions_overview = "overview" in text[:400].lower()
    ends_like_a_question = text.rstrip("*_>`\" ").endswith("?")
    return mentions_overview and not ends_like_a_question


# --- UI ---------------------------------------------------------------------------------

DEFAULT_ROSTER = """Ahmed - evenings, food service experience
Sara - weekends, youth coordinator
Omar - 4 hours available, good at setup/teardown
Fatima - registration and front-desk experience"""

st.set_page_config(page_title="Khidmah AI", page_icon="🕌", layout="wide")
st.title("🕌 Khidmah AI")
st.caption(
    "Your mosque's AI operations manager. 76% of U.S. mosques are run entirely by volunteers "
    "(American Mosque Survey, ISPU 2020) — the problem isn't a missing dashboard, it's missing "
    "hands. Khidmah doesn't ask an admin to organize information; it plans, staffs, and drafts "
    "the communications for them, then checks its own work before anything goes out."
)

goal = st.text_area(
    "What does your community need?",
    placeholder=(
        "e.g. 'Plan a Ramadan community iftar for 300 people, budget $3,000' or "
        "'We need a youth basketball tournament next Saturday' or "
        "'Send an announcement that Friday khutbah is starting 15 minutes early this week'"
    ),
    height=90,
)

with st.expander("Volunteer roster (demo data pre-filled — edit it to try your own)"):
    volunteer_roster = st.text_area("Available volunteers", value=DEFAULT_ROSTER, height=130)

goal_is_empty = not goal.strip()
generate_clicked = st.button("Generate Operations Plan", type="primary", disabled=goal_is_empty)
if goal_is_empty:
    st.caption("⬆️ Describe what your community needs above to enable this button.")

if generate_clicked and goal.strip():
    client = get_client()
    status_box = st.status("Khidmah AI is working...", expanded=True)
    status_box.write("🧠 Planner Agent — reading the goal and deciding what's actually needed...")

    def on_step(tool_name, _tool_input):
        status_box.write(f"→ {STEP_LABELS.get(tool_name, tool_name)}")

    try:
        final_plan, tool_log, usages = run_orchestrator(client, goal, volunteer_roster, on_step=on_step)

        plan_is_finished = is_finished_plan(tool_log, final_plan)
        audit_result = None
        if plan_is_finished:
            status_box.write("🔍 Audit Agent — reviewing the finished plan for gaps before it goes live...")
            audit_result, audit_usage = audit_step(client, final_plan)
            usages.append(audit_usage)
    except AgentCallError as exc:
        status_box.update(label="Failed", state="error")
        st.error(f"⚠️ {exc}")
        st.stop()

    status_box.update(label=f"Done — {len(tool_log)} specialist agent(s) called", state="complete")

    st.divider()

    if not plan_is_finished:
        st.subheader("💬 Khidmah AI needs more information")
        st.markdown(escape_markdown_dollars(final_plan))
    else:
        st.subheader("📋 Operations Plan")
        st.markdown(escape_markdown_dollars(final_plan))

        st.subheader("⚠️ Needs Attention")
        st.markdown(escape_markdown_dollars(audit_result))

    if tool_log:
        with st.expander("See each specialist agent's raw output (the reasoning trail)"):
            for name, _inp, result in tool_log:
                st.markdown(f"**{STEP_LABELS.get(name, name)}**")
                st.markdown(escape_markdown_dollars(result))
                st.divider()
    else:
        st.info("The planner decided no specialist agents were needed for this goal — it answered directly.")

    cost = estimate_cost(usages)
    st.caption(
        f"Estimated cost for this run: **${cost:.4f}** across {len(usages)} model call(s) — "
        f"planner + audit use {STRONG_MODEL} (the two judgment calls: deciding what work is "
        f"needed, and critiquing the finished plan); each specialist agent uses {CHEAP_MODEL} "
        "(execution, once told exactly what to produce). Swap in real per-token pricing before "
        "you quote this to judges."
    )

st.divider()
st.caption(
    "⚠️ Hackathon MVP: this demos one flow end-to-end (goal → plan → staffing → comms → audit) "
    "with a demo volunteer roster. It is not a religious authority — an imam/board should "
    "review anything before it's actually published to a community."
)
