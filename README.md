# Khidmah AI

An AI operations manager for mosques — not another dashboard to update, an agent that plans,
staffs, and drafts communications for a community event, then reviews its own work.

Built for the Niya Summit "Founder in a Day" hackathon, Aug 29 2026.

## The problem

76% of U.S. mosques are run entirely by volunteers (American Mosque Survey, ISPU 2020) —
see `research/notes.md` for the sourced numbers. The gap isn't missing software for prayer
times, announcements, or donations — several products already do that. The gap is that nobody
is actually doing the *work* of planning, staffing, and communicating an event for an
under-staffed volunteer team.

## How it works

A genuinely agentic pipeline (see `app.py`) — not a fixed script:

1. **Planner agent** (`claude-sonnet-4-6`) reads the admin's goal and decides, via real
   tool-use, which specialist agents this specific goal needs and in what order. A simple
   announcement might only need the comms agent; a full event usually needs all three.
2. **Specialist agents** (`claude-haiku-4-5`), called as tools by the planner:
   - `run_event_agent` — schedule, task checklist, budget, supplies
   - `run_volunteer_agent` — assigns real volunteers from the roster to the roles above
   - `run_comms_agent` — WhatsApp / email / Instagram / volunteer thank-you drafts
3. **Audit agent** (`claude-sonnet-4-6`) reviews the *combined* plan for gaps volunteer-run
   mosques actually hit — salah-time conflicts, accessibility, parental consent for youth
   events, unassigned roles, unrealistic budgets — before anything is marked ready to publish.

## Model stack & cost per task

- Planner: `claude-sonnet-4-6` (the judgment call: deciding what work is needed)
- Specialist agents: `claude-haiku-4-5-20251001` (execution, once told what to produce)
- Audit: `claude-sonnet-4-6` (the other judgment call: critiquing the finished plan)
- The app prints an estimated cost per run live in the UI. Fill in the actual number you
  observe during testing here before you submit: **Estimated cost per run: $____**

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # paste your real ANTHROPIC_API_KEY into .env (same key as any other project)
streamlit run app.py
```

## Deploying (required — judges need a real URL, not localhost)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy this repo (`app.py` as the entry point).
3. In the deployed app's **Settings -> Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key"
   ```
4. Never put the real key in this repo — `.env` is git-ignored on purpose.

## Research & validation

See `research/notes.md` — keep adding to it throughout the day.

## Disclaimer

Hackathon MVP demoing one flow end-to-end. Not a religious authority — an imam/board should
review anything before it's actually published to a real community.
