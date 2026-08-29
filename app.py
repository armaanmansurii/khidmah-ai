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
import re
import uuid
from datetime import date, datetime, timedelta, time as dt_time, timezone

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

# The 5 prep-timeline phases the event agent's checklist must be organized under, and how many
# days before the event each one lands — drives the .ics export (see build_prep_timeline_ics).
EVENT_PHASES = [
    ("4 Weeks Before", 28),
    ("3 Weeks Before", 21),
    ("2 Weeks Before", 14),
    ("1 Week Before", 7),
    ("Day-Of Timeline", 0),
]


def build_event_agent_system(today: date) -> str:
    """EVENT_AGENT_SYSTEM is date-aware (built fresh per request) because it must resolve
    relative dates ("next Friday") against the actual current date, and its output is
    parsed by parse_event_timeline() to build a downloadable prep-timeline calendar — so
    the format contract here (EVENT_DATE/EVENT_TIME/EVENT_TITLE lines, exact phase
    headings) is load-bearing, not just stylistic."""
    return f"""You are the logistics specialist for a mosque's operations team. Today's date \
is {today.isoformat()} ({today.strftime("%A, %B %d, %Y")}).

First, resolve the event goal's date/time to an absolute calendar date using today's date \
(e.g. "next Friday" or "this Saturday at 6pm" both resolve to a specific date). If the event \
goal does not mention any date, day, or timeframe at all, respond with EXACTLY the single line \
NEEDS_DATE and nothing else — do not guess a date.

Otherwise, your response MUST begin with these lines, in this exact order, before anything \
else (no preamble):
EVENT_DATE: YYYY-MM-DD
EVENT_TIME: HH:MM
EVENT_TITLE: <short event name, 3-6 words>
(Omit the EVENT_TIME line entirely if no time was given or clearly implied.)

Then produce a task checklist organized under EXACTLY these five section headings, each on \
its own line starting with "## ", in this exact order, with every task as a bullet point \
under whichever phase it actually needs to happen in (setup, cleanup, first aid, \
prayer/salah scheduling, accessibility, and anything else people forget):
## 4 Weeks Before
## 3 Weeks Before
## 2 Weeks Before
## 1 Week Before
## Day-Of Timeline
If a phase genuinely has nothing to do, still include its heading with a single bullet \
"- Nothing specific this phase." — all five headings must always be present.

After Day-Of Timeline, add a same-day schedule/timeline, a budget breakdown that fits any \
stated budget, and a supplies list. Be concrete and mosque-specific — assume volunteer labor, \
not paid staff, unless told otherwise. End with a short line listing the distinct ROLES that \
will need volunteers, so another agent can staff them."""

VOLUNTEER_AGENT_SYSTEM = """You are the volunteer coordination specialist. Given a list of \
roles/shifts needed and a roster of available volunteers with their availability and skills, \
assign volunteers to roles sensibly, matching skills to tasks where possible. If there are not \
enough volunteers for a role, say so explicitly rather than inventing people who aren't on the \
roster. Output as a clear list: ROLE — assigned volunteer(s) — any gap/shortage noted."""

COMMS_AGENT_SYSTEM = """You draft community communications for a mosque. Given an event \
summary, produce four short pieces, clearly labeled: WHATSAPP (casual, brief, with an Islamic \
greeting), EMAIL (slightly more formal, with a subject line), INSTAGRAM CAPTION (short, with \
2-4 relevant hashtags), VOLUNTEER THANK-YOU MESSAGE (warm, brief). Keep each under 80 words."""

RECRUITMENT_COMMS_SYSTEM = """You draft a "Call for Volunteers" recruitment announcement for a \
mosque event that has NO confirmed volunteers yet — do not draft a thank-you message, there is \
no one to thank. Given a summary of the event and the specific roles/shifts still needed \
(with headcounts), produce three short pieces, clearly labeled: WHATSAPP (casual, brief, with \
an Islamic greeting), EMAIL (slightly more formal, with a subject line), INSTAGRAM CAPTION \
(short, with 2-4 relevant hashtags). Each piece must name the actual roles/shifts and \
headcounts needed (e.g. "2 volunteers for setup, Friday 2-4pm") — never a vague "we need \
help" — and tell people how to sign up (reply to this message/email). Never invent roles that \
weren't given to you. Keep each under 80 words."""

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
                },
                "is_recruitment_call": {
                    "type": "boolean",
                    "description": (
                        "Set to true only in recruitment mode (no volunteers confirmed yet). "
                        "This drafts a 'Call for Volunteers' announcement asking people to sign "
                        "up for specific roles, instead of a normal event announcement — the "
                        "event_summary should then explicitly list the roles/headcounts still "
                        "needed (from run_event_agent's checklist), not just describe the event."
                    ),
                },
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


def execute_tool(client: Anthropic, name: str, tool_input: dict, today: date) -> tuple[str, dict]:
    def require(field: str) -> str:
        value = tool_input.get(field)
        if value is None:
            raise AgentCallError(f"{name} was called without the required '{field}' field — can't proceed.")
        return value

    if name == "run_event_agent":
        return call_model(
            client,
            CHEAP_MODEL,
            build_event_agent_system(today),
            require("event_goal"),
            step_label="Event/logistics agent",
        )
    if name == "run_volunteer_agent":
        user_msg = (
            f"Roles/shifts needed:\n{require('tasks_needed')}\n\n"
            f"Available volunteers:\n{require('volunteer_roster')}"
        )
        return call_model(client, CHEAP_MODEL, VOLUNTEER_AGENT_SYSTEM, user_msg, step_label="Volunteer coordinator agent")
    if name == "run_comms_agent":
        is_recruitment_call = bool(tool_input.get("is_recruitment_call", False))
        system = RECRUITMENT_COMMS_SYSTEM if is_recruitment_call else COMMS_AGENT_SYSTEM
        return call_model(client, CHEAP_MODEL, system, require("event_summary"), step_label="Communications agent")
    raise AgentCallError(f"Planner requested an unknown tool: {name}")


# --- Planner (orchestrator) — this is the actual agentic loop -------------------------------

PLANNER_SYSTEM = """You are the operations planner for Khidmah AI, an AI operations manager \
for mosques and Islamic community organizations. An admin will describe something their \
community needs — an event, a program, an announcement, a volunteer effort.

You have three specialist agents available as tools:
- run_event_agent: logistics, scheduling, checklists, budget, supplies
- run_volunteer_agent: assigns people to roles/shifts (needs a task list AND the volunteer roster)
- run_comms_agent: drafts announcements/messages once the event/program is defined; set \
is_recruitment_call=true instead of a normal announcement when there are no volunteers yet

Decide for yourself which of these THIS SPECIFIC goal actually needs — not every goal needs \
all three. A pure announcement about something already planned might only need \
run_comms_agent. A full new event usually needs run_event_agent first (so you know what roles \
exist), then run_volunteer_agent (using those roles + the roster you were given), then \
run_comms_agent last (once the event is fully defined). Call one tool at a time and use each \
result to inform the next call's input.

RECRUITMENT MODE: if the admin's message says no volunteers are confirmed yet, do NOT call \
run_volunteer_agent — there is no one to assign. Instead, once run_event_agent has identified \
the roles/shifts needed, call run_comms_agent with is_recruitment_call=true and an \
event_summary that explicitly lists those specific roles and headcounts, so it drafts a real \
call for volunteers instead of a generic announcement.

DATE REQUIRED: this app builds a downloadable prep-timeline calendar from run_event_agent's \
output, which requires a real event date — never guess one yourself. If run_event_agent's \
result is exactly "NEEDS_DATE", stop calling tools immediately (do not call run_volunteer_agent \
or run_comms_agent) and ask the admin what date/day the event is, the same way you'd ask about \
any other missing detail.

Once you have everything the goal requires, STOP calling tools and instead write ONE complete, \
well-organized final operations plan for the admin, with headings: Overview, Task Checklist, \
then either Volunteer Assignments (only if you ran the volunteer agent — named volunteers per \
role) or Volunteers Needed (recruitment mode — the roles/headcounts still needed, no names), \
then Communications Drafts (only if you ran the comms agent). Never invent volunteer names \
that weren't in the roster you were given."""

AUDIT_SYSTEM = """You are the audit/review agent for a mosque operations plan. Given a \
complete plan (checklist, either volunteer assignments or a volunteers-needed list, and \
communications), review it critically for gaps commonly missed by volunteer-run mosques: \
conflicts with prayer/salah times, accessibility needs, parental consent for youth-focused \
events, insufficient cleanup crew, a budget that doesn't add up or is unrealistic for the \
stated amount, food-safety/allergy notes, and any role mentioned in the checklist that was \
never actually assigned a volunteer.

If the plan has a "Volunteers Needed" section instead of "Volunteer Assignments", that means \
recruitment mode was deliberately chosen — there are no volunteers yet by design, so do NOT \
flag unassigned or unnamed roles as a defect. Instead check that the Communications Drafts are \
an actual call for volunteers (not a generic announcement) and that every role/headcount from \
the Task Checklist is named somewhere in that call — flag any checklist role the recruitment \
call fails to mention.

Output a short bullet list of flags (⚠ each one), or state clearly "✅ No major gaps found" if \
there truly are none — don't invent issues to seem thorough. End with exactly one line: \
"READY TO PUBLISH: Yes" or "READY TO PUBLISH: Needs attention"."""


def run_orchestrator(client: Anthropic, goal: str, volunteer_roster: str, recruitment_mode: bool = False, on_step=None):
    """The planner decides which specialist agents to call, in what order, via real tool-use.
    Returns (final_plan_text, tool_call_log, usages)."""
    usages: list[dict] = []
    tool_log: list[tuple[str, dict, str]] = []
    today = date.today()
    if recruitment_mode:
        # Whatever's in the roster textbox is irrelevant here — deliberately not passed to
        # the planner at all, since there's no one to assign yet.
        roster_section = (
            "RECRUITMENT MODE — no volunteers are confirmed yet. Do NOT call "
            "run_volunteer_agent (there is no one to assign). Once run_event_agent has "
            "identified the roles/shifts needed, call run_comms_agent with "
            "is_recruitment_call=true and list those specific roles and headcounts so it can "
            "draft a real call for volunteers."
        )
    else:
        roster_section = f"Available volunteers:\n{volunteer_roster}"
    messages = [
        {
            "role": "user",
            "content": f"Community goal: {goal}\n\nToday's date: {today.isoformat()}\n\n{roster_section}",
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
            result_text, usage = execute_tool(client, block.name, block.input, today)
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


def parse_event_timeline(event_agent_output: str) -> dict | None:
    """Extract the resolved event date/time/title and the 5 prep phases (with computed
    dates) from run_event_agent's raw output (see build_event_agent_system for the format
    contract). Returns None if there's no usable EVENT_DATE line — e.g. the event agent
    asked for a date instead (NEEDS_DATE) — so callers just skip offering the .ics
    download rather than guess or build a broken calendar file."""
    text = event_agent_output or ""
    date_match = re.search(r"(?m)^EVENT_DATE:\s*(\d{4}-\d{2}-\d{2})\s*$", text)
    if not date_match:
        return None
    try:
        event_date = date.fromisoformat(date_match.group(1))
    except ValueError:
        return None

    time_match = re.search(r"(?m)^EVENT_TIME:\s*(\d{1,2}):(\d{2})\s*$", text)
    event_time = dt_time(int(time_match.group(1)), int(time_match.group(2))) if time_match else None

    title_match = re.search(r"(?m)^EVENT_TITLE:\s*(.+?)\s*$", text)
    event_title = title_match.group(1) if title_match else "Event"

    phases = []
    for label, days_before in EVENT_PHASES:
        heading = re.search(rf"(?im)^#{{1,3}}[^\n]*\b{re.escape(label)}\b[^\n]*$", text)
        if not heading:
            continue
        start = heading.end()
        next_heading = re.search(r"(?m)^#{1,3}[^\n]*$", text[start:])
        end = start + next_heading.start() if next_heading else len(text)
        body = text[start:end].strip()
        if body:
            phases.append({"label": label, "date": event_date - timedelta(days=days_before), "items": body})

    if not phases:
        return None
    return {"event_date": event_date, "event_time": event_time, "event_title": event_title, "phases": phases}


def _ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _ics_fold(line: str) -> str:
    """RFC 5545: content lines SHOULD be folded at 75 octets; continuation lines start
    with a single space. A simple character-based fold (not exact octet-counting) is fine
    for the mostly-ASCII text this app generates."""
    limit = 74
    if len(line) <= limit:
        return line
    chunks = [line[:limit]]
    rest = line[limit:]
    while rest:
        chunks.append(" " + rest[: limit - 1])
        rest = rest[limit - 1 :]
    return "\r\n".join(chunks)


def build_prep_timeline_ics(timeline: dict) -> bytes:
    """Build an RFC 5545 .ics calendar: one all-day VEVENT per prep phase on its computed
    date, plus one VEVENT for the event itself (timed if EVENT_TIME was given, else
    all-day)."""
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Khidmah AI//Prep Timeline//EN", "CALSCALE:GREGORIAN"]

    def add_vevent(uid_suffix: str, summary: str, description: str, start_date: date, start_time: dt_time | None):
        lines.append("BEGIN:VEVENT")
        lines.append(_ics_fold(f"UID:{uuid.uuid4()}-{uid_suffix}@khidmah.ai"))
        lines.append(f"DTSTAMP:{dtstamp}")
        if start_time is None:
            lines.append(f"DTSTART;VALUE=DATE:{start_date.strftime('%Y%m%d')}")
            lines.append(f"DTEND;VALUE=DATE:{(start_date + timedelta(days=1)).strftime('%Y%m%d')}")
        else:
            start_dt = datetime.combine(start_date, start_time)
            end_dt = start_dt + timedelta(hours=3)  # reasonable default event duration
            lines.append(f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}")
            lines.append(f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}")
        lines.append(_ics_fold(f"SUMMARY:{_ics_escape(summary)}"))
        lines.append(_ics_fold(f"DESCRIPTION:{_ics_escape(description)}"))
        lines.append("END:VEVENT")

    event_title = timeline["event_title"]
    for phase in timeline["phases"]:
        add_vevent(
            uid_suffix=phase["label"].lower().replace(" ", "-"),
            summary=f"{event_title} Prep: {phase['label']}",
            description=phase["items"],
            start_date=phase["date"],
            start_time=None,  # prep reminders are all-day
        )

    add_vevent(
        uid_suffix="event",
        summary=event_title,
        description="The event itself.",
        start_date=timeline["event_date"],
        start_time=timeline["event_time"],
    )

    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


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
    recruitment_mode = st.checkbox(
        "We don't have volunteers yet — generate a recruitment call instead of assignments."
    )
    volunteer_roster = st.text_area(
        "Available volunteers", value=DEFAULT_ROSTER, height=130, disabled=recruitment_mode
    )

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
        final_plan, tool_log, usages = run_orchestrator(
            client, goal, volunteer_roster, recruitment_mode=recruitment_mode, on_step=on_step
        )

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

        event_agent_output = next((result for name, _inp, result in tool_log if name == "run_event_agent"), None)
        timeline = parse_event_timeline(event_agent_output) if event_agent_output else None
        if timeline:
            st.download_button(
                "📅 Download prep timeline (.ics) — import into Google Calendar or any calendar app",
                data=build_prep_timeline_ics(timeline),
                file_name="khidmah_prep_timeline.ics",
                mime="text/calendar",
            )
            st.caption(
                "Google Calendar: Settings (gear icon) → Import & export → Import, then select this file."
            )

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
