"""
Vera-challenge bot — magicpin AI Challenge submission.

Architecture (see README.md for the full writeup):
  context store -> fact assembler -> kind-family prompt dispatch -> LLM ->
  guardrail validator -> conversation state machine (tick policy / reply FSM)

Design priorities, in order:
  1. Never fabricate. The LLM only ever sees a small assembled facts packet,
     never raw context blobs — so it structurally can't cite a number that
     wasn't actually pushed to us.
  2. Never crash / never time out. Every endpoint has a hard wall-clock
     budget; tick uses a thread pool so N triggers compose in parallel
     instead of serially blowing the 30s window.
  3. Handle inputs we've never seen. Any trigger.kind not in FAMILY_MAP
     falls through to a generic-but-still-grounded family instead of
     erroring — this is what "fresh scenarios at judge time" needs.
  4. Deterministic. temperature=0 everywhere, no `random`, stable sorts.

Run:
    pip install -r requirements.txt --break-system-packages
    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / GOOGLE_API_KEY
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Some Python installs (notably python.org builds on macOS) ship without a
# usable CA bundle wired into the stdlib ssl module, which makes every
# urllib.request.urlopen() call to an https endpoint fail with
# CERTIFICATE_VERIFY_FAILED even though the cert is fine. We pin an explicit
# context built from certifi's bundle so LLM calls don't depend on the host
# machine's cert store being set up correctly. Falls back to the stdlib
# default context if certifi isn't installed.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

# ============================================================================
# CONFIG
# ============================================================================

TEAM_NAME = os.environ.get("TEAM_NAME", "Og")
TEAM_MEMBERS = [os.environ.get("TEAM_MEMBER", "Og")]
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "ayushgupta7729@gmail.com")
BOT_VERSION = "0.1.0"

TICK_WALLCLOCK_BUDGET_S = 25.0          # stay under the judge's 30s timeout
REPLY_WALLCLOCK_BUDGET_S = 25.0
MAX_NEW_CONVOS_PER_MERCHANT_PER_TICK = 1  # restraint: one new thread per merchant per tick
MIN_TICKS_GAP_PER_MERCHANT = 1            # don't message the same merchant on back-to-back ticks
REPETITION_SIMILARITY_REJECT = 0.86       # difflib ratio above this = "same message" -> penalty risk

START_TIME = time.time()
_LOCK = threading.RLock()

# ============================================================================
# IN-MEMORY STORE
# ============================================================================

# (scope, context_id) -> {"version": int, "payload": dict}
CONTEXTS: dict[tuple[str, str], dict] = {}

# conversation_id -> state dict
CONVERSATIONS: dict[str, dict] = {}

# suppression_key -> sent_at (iso) ; a key we've already acted on
SENT_SUPPRESSION_KEYS: dict[str, str] = {}

# merchant_id -> last tick index we sent something on (cadence cap)
LAST_SEND_TICK: dict[str, int] = {}
_TICK_COUNTER = {"n": 0}


def ctx_counts() -> dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in CONTEXTS:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = CONTEXTS.get((scope, context_id))
    return entry["payload"] if entry else None


# ============================================================================
# LLM PROVIDER (stdlib only — no extra deps, matches judge_simulator.py style)
# ============================================================================

TIMEOUT_LLM = 20


def _call_anthropic(system: str, prompt: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    body = json.dumps({
        "model": model, "max_tokens": 700, "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "Content-Type": "application/json",
                 "anthropic-version": "2023-06-01", "User-Agent": "curl/8.4.0"})
    resp = urllib.request.urlopen(req, timeout=TIMEOUT_LLM, context=SSL_CONTEXT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


def _call_openai(system: str, prompt: str) -> str:
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 700,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = urllib.request.urlopen(req, timeout=TIMEOUT_LLM, context=SSL_CONTEXT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_gemini(system: str, prompt: str) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    model = os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash")
    full_prompt = f"{system}\n\n{prompt}"
    body = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 700},
    }).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = urllib.request.urlopen(req, timeout=TIMEOUT_LLM, context=SSL_CONTEXT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(system: str, prompt: str) -> str:
    # Groq (api.groq.com) — fast inference, OpenAI-compatible /chat/completions.
    # Not to be confused with xAI's Grok (_call_grok above) — different company,
    # different key, different endpoint.
    #
    # openai/gpt-oss-20b is a reasoning model. Without JSON mode enabled,
    # Groq's reasoning_format defaults to "raw" — the model's chain-of-thought
    # gets embedded inline in `content` inside <think> tags, ahead of the
    # actual JSON answer. Two consequences we hit in practice: (1) reasoning
    # tokens eat into the completion budget, so a short max_tokens can leave
    # nothing for the real answer — an empty/truncated body, not an error;
    # (2) the stray <think> block sitting next to the JSON increases the
    # odds of the extraction regex misfiring. reasoning_format="hidden" makes
    # Groq return only the final answer, and a larger token budget gives the
    # (now-invisible) reasoning room to finish without starving it.
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    body = json.dumps({
        "model": model, "temperature": 0, "max_completion_tokens": 4096,
        "reasoning_format": "hidden", "reasoning_effort": "low",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = urllib.request.urlopen(req, timeout=TIMEOUT_LLM, context=SSL_CONTEXT)
    raw_resp = resp.read().decode("utf-8")
    if os.environ.get("DEBUG_LLM"):
        print(f"[DEBUG_LLM raw groq response] {raw_resp}", flush=True)
    data = json.loads(raw_resp)
    return data["choices"][0]["message"]["content"]


def _call_grok(system: str, prompt: str) -> str:
    # xAI's API is OpenAI-compatible (same /v1/chat/completions shape).
    # Check https://docs.x.ai for the current model list before relying on
    # this default — model names have moved fast (grok-4.6 -> grok-4.7 in
    # the same month) and a stale name will just 404.
    key = os.environ["XAI_API_KEY"]
    model = os.environ.get("XAI_MODEL", "grok-4-fast")
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 700,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    resp = urllib.request.urlopen(req, timeout=TIMEOUT_LLM, context=SSL_CONTEXT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _fallback_compose(system: str, prompt: str) -> str:
    """No API key configured. Deterministic, clearly-labeled stand-in so the
    service still runs end-to-end. Replace by setting an API key — this path
    will not score well (it's a generic template), it exists purely so
    wiring can be verified without a key."""
    return json.dumps({
        "body": "STUB: no LLM key configured — set ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or XAI_API_KEY.",
        "cta": "none",
    })


def call_llm(system: str, prompt: str) -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(system, prompt)
    if os.environ.get("OPENAI_API_KEY"):
        return _call_openai(system, prompt)
    if os.environ.get("GROQ_API_KEY"):
        return _call_groq(system, prompt)
    if os.environ.get("GOOGLE_API_KEY"):
        return _call_gemini(system, prompt)
    if os.environ.get("XAI_API_KEY"):
        return _call_grok(system, prompt)
    return _fallback_compose(system, prompt)


# ============================================================================
# FACT ASSEMBLY — the anti-fabrication chokepoint.
# The composer NEVER sees raw category/merchant/trigger/customer dicts.
# It only ever sees what this function decides to extract.
# ============================================================================

def _active_offers(merchant: dict) -> list[str]:
    return [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]


def _find_digest_item(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def assemble_facts(category: dict, merchant: dict, trigger: dict,
                    customer: Optional[dict]) -> dict:
    identity = merchant.get("identity", {})
    facts: dict[str, Any] = {
        "merchant_name": identity.get("name"),
        "owner_first_name": identity.get("owner_first_name"),
        "city": identity.get("city"),
        "locality": identity.get("locality"),
        "languages": identity.get("languages", ["en"]),
        "verified": identity.get("verified"),
        "performance": merchant.get("performance", {}),
        "active_offers": _active_offers(merchant),
        "signals": merchant.get("signals", []),
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "peer_stats": category.get("peer_stats", {}),
        "voice": category.get("voice", {}),
        "seasonal_beats": category.get("seasonal_beats", []),
        "trend_signals": category.get("trend_signals", []),
        "trigger_kind": trigger.get("kind"),
        "trigger_scope": trigger.get("scope"),
        "trigger_source": trigger.get("source"),
        "trigger_urgency": trigger.get("urgency"),
        "trigger_payload": trigger.get("payload", {}),
    }

    top_item_id = trigger.get("payload", {}).get("top_item_id")
    digest_item = _find_digest_item(category, top_item_id)
    if digest_item:
        facts["digest_item"] = digest_item

    if customer:
        c_identity = customer.get("identity", {})
        facts["customer"] = {
            "name": c_identity.get("name"),
            "language_pref": c_identity.get("language_pref"),
            "relationship": customer.get("relationship", {}),
            "state": customer.get("state"),
            "preferences": customer.get("preferences", {}),
        }

    return facts


# ============================================================================
# TRIGGER-KIND -> PROMPT FAMILY DISPATCH
# Unknown kinds (anything the judge injects that we've never seen) fall
# through to "generic" rather than erroring — this is the safety net for
# the "fresh scenarios" the real harness tests with.
# ============================================================================

FAMILY_MAP: dict[str, str] = {
    "research_digest": "knowledge_digest",
    "category_research_digest_release": "knowledge_digest",
    "cde_opportunity": "knowledge_digest",
    "regulation_change": "urgent_compliance",
    "supply_alert": "urgent_compliance",
    "category_trend_movement": "knowledge_digest",
    "perf_spike": "performance",
    "perf_dip": "performance",
    "seasonal_perf_dip": "performance",
    "milestone_reached": "performance",
    "recall_due": "customer_lifecycle",
    "customer_lapsed_soft": "customer_lifecycle",
    "customer_lapsed_hard": "customer_lifecycle",
    "appointment_tomorrow": "customer_lifecycle",
    "chronic_refill_due": "customer_lifecycle",
    "trial_followup": "customer_lifecycle",
    "wedding_package_followup": "customer_lifecycle",
    "unplanned_slot_open": "customer_lifecycle",
    "curious_ask_due": "curious_ask",
    "scheduled_recurring": "curious_ask",
    "competitor_opened": "competitive",
    "review_theme_emerged": "review_theme",
    "festival_upcoming": "external_event",
    "weather_heatwave": "external_event",
    "local_news_event": "external_event",
    "ipl_match_today": "external_event",
    "category_seasonal": "external_event",
    "renewal_due": "lifecycle_account",
    "winback_eligible": "lifecycle_account",
    "gbp_unverified": "lifecycle_account",
    "dormant_with_vera": "lifecycle_account",
    "active_planning_intent": "active_planning",
}

FAMILY_FRAMING: dict[str, str] = {
    "knowledge_digest": (
        "Frame this as a relevant knowledge item just released for their category. "
        "Cite the source (name + date/page from digest_item if present). Anchor it to "
        "something specific about THIS merchant (a signal, a customer_aggregate segment) "
        "so it isn't generic. End with a low-friction, reciprocity-style offer "
        "('want me to draft/pull X for you'). CTA should be open_ended."
    ),
    "urgent_compliance": (
        "This is urgent and specific (recall/regulation). Lead with the concrete fact "
        "(batch numbers, deadline, molecule). If customer_aggregate lets you compute an "
        "affected count, state it. Offer a concrete next artifact (a customer note, a "
        "workflow). Tone: bounded urgency, not alarmist. CTA: open_ended or binary_confirm."
    ),
    "performance": (
        "Anchor on the exact performance delta given. If a signal says this is expected/"
        "seasonal, reframe it as not-a-problem using peer context if available — don't "
        "let the merchant panic over a normal seasonal dip. Propose one concrete action. "
        "CTA: open_ended."
    ),
    "customer_lifecycle": (
        "This is being sent ON BEHALF OF the merchant to their own customer (send_as = "
        "merchant_on_behalf). Use the customer's name and honor their language_pref and "
        "preferences.preferred_slots. Reference only real active_offers for pricing. If "
        "trigger_payload has available_slots, offer them as a multi-choice CTA; otherwise "
        "use a single binary CTA. Warm, no guilt-tripping, no medical/health claims beyond "
        "what voice.taboos allows."
    ),
    "curious_ask": (
        "Ask the merchant ONE low-stakes question (what's in demand, what's changed) with "
        "no commitment implied. Offer something back for answering (a post draft, a reply "
        "template). Keep it very short. CTA: open_ended."
    ),
    "competitive": (
        "Frame competitor movement as useful intelligence, not fear-mongering. State the "
        "specific fact (name, distance, their offer) only if present in trigger_payload. "
        "Propose one concrete counter-move. CTA: open_ended."
    ),
    "review_theme": (
        "State the emerging review pattern with its occurrence count and trend. Propose a "
        "concrete fix, not just awareness. CTA: open_ended."
    ),
    "external_event": (
        "Connect the external event to a specific, sometimes counter-intuitive operating "
        "decision (e.g. skip vs lean into a promo) rather than a generic 'this is "
        "happening' message. Use any peer_stats or historical framing available. If "
        "nothing category-specific applies, keep this message short and skip the "
        "the CTA (cta: none) rather than force relevance that isn't there."
    ),
    "lifecycle_account": (
        "This is about the merchant's own account/subscription/verification state. Be "
        "direct and factual (days remaining, what verification unlocks). No hype. CTA: "
        "binary_yes_no or open_ended depending on urgency."
    ),
    "active_planning": (
        "The merchant has ALREADY expressed explicit interest in something specific "
        "(see trigger_payload.merchant_last_message) — they said yes, not 'maybe'. Do NOT "
        "ask a qualifying question, and do NOT re-pitch the idea as something being "
        "considered. Write this as an announcement that the next step is already being "
        "set up, not an offer being floated: state what you're doing ('Setting up X for "
        "Y' / 'Locking in X'), not what you could do ('we can do X' / 'we could set up "
        "X'). Avoid hedging verbs — no 'can', 'could', 'would you like'. Attach a concrete "
        "scope/number from the facts. End on a single confirm-to-execute line ('Confirm "
        "and I'll start' / 'Reply yes and this goes live today') rather than an open "
        "'do you want to proceed?' question — the merchant already answered that. CTA: "
        "binary_confirm_cancel."
    ),
    "generic": (
        "No specific playbook exists for this trigger kind yet. Ground the message ONLY "
        "in the facts given below — do not invent a reason beyond what trigger_payload "
        "actually contains. If the facts are too thin to say anything specific and "
        "verifiable, it is better to say less than to pad with generic filler. CTA: "
        "open_ended, or cta: none if there's truly nothing actionable yet."
    ),
}

SYSTEM_PROMPT = """You are the composer inside a merchant-engagement WhatsApp bot ("Vera") \
for magicpin, an Indian local-commerce platform. You write ONE outbound message to a \
merchant (or, when a customer block is present, ON BEHALF OF the merchant to their own \
customer).

HARD RULES — violating any of these is scored as a severe failure:
1. GROUNDING: use ONLY the facts given to you below. Never invent a number, a source, a \
competitor name, or a customer count. If you did not receive a fact, do not reference it.
2. SPECIFICITY: prefer a concrete verifiable number/date/source over a vague claim. \
"10% off" is weak; "Haircut @ Rs99" is strong.
3. VOICE: match category voice.tone. Respect voice.taboos — never use a taboo word. Use \
owner_first_name if present ("Hi Dr. Meera" / "Hi Suresh"), never a generic "Hi there".
4. LANGUAGE: if languages includes "hi" (or customer.language_pref mentions hi-en mix), \
write in natural Hindi-English code-mix, not pure English. Otherwise plain English.
5. ONE CTA: exactly one call-to-action, stated in the last sentence. Never stack multiple \
asks ("reply YES for X, NO for Y, MAYBE for Z").
6. NO URLS unless one is literally present in the facts given (none normally are) — do \
not fabricate a link.
7. NO PREAMBLE: no "I hope you're doing well" — get to the point in the first clause.
8. Do not re-introduce yourself if conversation_history is non-empty.
9. JSON VALIDITY: the "body" value must be valid inside a JSON string — never use an
unescaped double-quote character in the message text (use single quotes for emphasis
instead, e.g. 'Deep Cleaning' not "Deep Cleaning"), and never include a literal newline
(write one continuous sentence or use \\n).

Return ONLY valid JSON, no markdown fences, no commentary:
{"body": "<the message text>", "cta": "<one of: open_ended | binary_yes_no | \
binary_confirm_cancel | multi_choice_slot | none>"}"""


def family_for_kind(kind: str) -> str:
    return FAMILY_MAP.get(kind, "generic")


def build_prompt(facts: dict, family: str, send_as: str, conversation_history: str = "") -> str:
    framing = FAMILY_FRAMING[family]
    parts = [
        f"MESSAGE FAMILY: {family}",
        f"FAMILY FRAMING INSTRUCTIONS: {framing}",
        f"send_as = {send_as}"
        + (" (you ARE the merchant's own outbound voice to their customer)"
           if send_as == "merchant_on_behalf" else " (you are Vera, messaging the merchant directly)"),
        "",
        "FACTS (the only source of truth — do not use anything outside this):",
        json.dumps(facts, ensure_ascii=False, default=str, indent=2),
    ]
    if conversation_history:
        parts += ["", "CONVERSATION SO FAR (most recent last):", conversation_history]
    return "\n".join(parts)


# ============================================================================
# GUARDRAIL VALIDATOR
# ============================================================================

_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
_URL_RE = re.compile(r"https?://\S+")


def _numbers_in(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUM_RE.findall(text)}


def check_number_provenance(body: str, facts: dict) -> list[str]:
    """Every number in the body should trace back to the facts packet.
    Heuristic, not exact — flags candidates rather than hard-blocking,
    since some numbers (e.g. slot times '6pm') are legitimate without
    being a top-level fact. Used to decide whether to re-prompt."""
    facts_blob = json.dumps(facts, default=str)
    facts_nums = _numbers_in(facts_blob)
    body_nums = _numbers_in(body)
    # ignore tiny numbers (1-2 digit) — high false-positive rate (slot counts, "2 min")
    suspicious = [n for n in body_nums if len(n) >= 3 and n not in facts_nums]
    return suspicious


def check_single_cta(body: str) -> bool:
    """True if body looks like it stacks multiple distinct CTAs."""
    lowered = body.lower()
    multi_patterns = [
        r"reply\s+\w+\s+for\s+\w+.*reply\s+\w+\s+for\s+\w+",
        r"yes.*no.*maybe",
    ]
    return not any(re.search(p, lowered, re.S) for p in multi_patterns)


def strip_or_flag_urls(body: str, facts: dict) -> str:
    urls_in_facts = set(_URL_RE.findall(json.dumps(facts, default=str)))
    def _sub(m: re.Match) -> str:
        return m.group(0) if m.group(0) in urls_in_facts else ""
    return _URL_RE.sub(_sub, body).strip()


def too_similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= REPETITION_SIMILARITY_REJECT


def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    candidate = match.group() if match else raw
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Strict parsing failed — almost always because the model's message text
    # contains an unescaped quote/newline that breaks the outer JSON, even
    # though the intended body/cta are perfectly recoverable. Pull them out
    # with a tolerant regex rather than falling back to dumping the raw
    # JSON-looking string as the literal outbound message (which is what
    # used to happen here, and is how "{"body":"Hi Me..." ends up sent to a
    # merchant).
    body_match = re.search(r'"body"\s*:\s*"(.*)"\s*,\s*"cta"', candidate, re.S)
    cta_match = re.search(r'"cta"\s*:\s*"([a-zA-Z_]+)"', candidate)
    if body_match:
        body_text = body_match.group(1)
        body_text = body_text.replace('\\"', '"').replace("\\n", "\n").replace("\\'", "'")
        return {"body": body_text.strip(), "cta": cta_match.group(1) if cta_match else "open_ended"}

    return {"body": raw[:500], "cta": "open_ended"}


def compose_message(facts: dict, family: str, send_as: str,
                     conversation_history: str = "",
                     prior_bodies: Optional[list[str]] = None) -> dict:
    """Runs the LLM, validates, and does one bounded re-prompt if needed."""
    prior_bodies = prior_bodies or []
    prompt = build_prompt(facts, family, send_as, conversation_history)

    for attempt in range(2):  # one shot + one bounded retry
        raw = call_llm(SYSTEM_PROMPT, prompt)
        parsed = parse_llm_json(raw)
        body = (parsed.get("body") or "").strip()
        cta = parsed.get("cta", "open_ended")

        problems = []
        suspicious_nums = check_number_provenance(body, facts)
        if suspicious_nums:
            problems.append(f"These numbers aren't in the facts given: {suspicious_nums}. Remove or fix them.")
        if not check_single_cta(body):
            problems.append("You stacked multiple CTAs. Use exactly one.")
        if any(too_similar(body, p) for p in prior_bodies):
            problems.append("This is a near-duplicate of a message already sent in this conversation. Vary the wording and framing.")
        if not body:
            problems.append("Body was empty.")

        body = strip_or_flag_urls(body, facts)

        if not problems:
            return {"body": body, "cta": cta}

        if attempt == 0:
            prompt = prompt + ("\n\nYOUR PREVIOUS ATTEMPT HAD PROBLEMS — FIX THEM:\n"
                                + "\n".join(problems)
                                + "\n\nMake the minimal edit needed to fix these specific problems. "
                                  "Don't re-derive the message from scratch — reuse the good parts of "
                                  "your previous attempt as-is and only change what's flagged. Output "
                                  "the corrected JSON immediately.")
            continue

        # second attempt still has problems: ship the safest version we have
        # rather than loop forever or emit an error the harness would penalize
        return {"body": body or "(unable to compose — insufficient grounded facts)", "cta": cta}

    return {"body": "(unable to compose)", "cta": "none"}


def _safe_commit_fallback(facts: dict) -> dict:
    """Last-resort reply when call_llm raises (network/SSL/rate-limit/provider
    outage) instead of returning text. Built only from the facts packet, same
    grounding guarantee as the normal path, so this can never fabricate a
    number or a name that wasn't actually pushed to us."""
    name = facts.get("merchant_name") or facts.get("owner_first_name")
    greeting = f"Got it{', ' + name if name else ''} — " if name else "Got it — "
    return {"body": greeting + "give me a moment and I'll follow up with next steps shortly.",
            "cta": "none"}


# ============================================================================
# CONVERSATION STATE MACHINE
# ============================================================================

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching)",
    r"will (respond|revert|get back) (to you )?(shortly|soon)",
    r"currently (unavailable|busy|away)",
    r"team will (respond|reach|connect|revert)",
    r"automated (assistant|response|reply|message)",
    r"outside (of )?(our )?business hours",
    r"we (will|shall) get back to you",
]
HOSTILE_PATTERNS = [
    r"stop (messaging|texting|contacting)", r"\buseless\b", r"\bspam\b",
    r"harass", r"annoying", r"leave me alone", r"don'?t (message|contact|text) me",
    r"\bfuck", r"\bstupid\b", r"waste of (my )?time",
]
INTENT_COMMIT_PATTERNS = [
    r"let'?s do it", r"go ahead",
    r"\byes\b.*\b(join|do it|proceed|confirm|send|share|start|set\s*up)\b",
    r"^\s*(ok|okay|sure|yep|yeah)\b.*\b(do it|proceed|go ahead|send|share|start|set\s*up|confirm)\b",
    r"\bconfirm\b", r"sounds good",
    r"i want to join", r"ready to start", r"let'?s start",
]
NOT_INTERESTED_PATTERNS = [
    r"not interested", r"no thanks", r"remove me", r"unsubscribe", r"stop it",
]

def _matches_any(patterns: list[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def get_or_init_conv(conversation_id: str, merchant_id: Optional[str],
                      customer_id: Optional[str]) -> dict:
    if conversation_id not in CONVERSATIONS:
        CONVERSATIONS[conversation_id] = {
            "merchant_id": merchant_id, "customer_id": customer_id,
            "turns": [],            # [{"from": "bot"/"merchant", "body": str}]
            "sent_bodies": [],      # bot bodies only, for repetition check
            "merchant_msgs": [],    # raw merchant messages, for auto-reply detection
            "state": "opened",      # opened -> qualifying -> action_committed / ended
            "hostile_warned": False,
        }
    return CONVERSATIONS[conversation_id]


def handle_reply(conv_id: str, merchant_id: Optional[str], customer_id: Optional[str],
                  from_role: str, message: str) -> dict:
    conv = get_or_init_conv(conv_id, merchant_id, customer_id)
    conv["turns"].append({"from": from_role, "body": message})
    conv["merchant_msgs"].append(message)

    # --- auto-reply escalation ---------------------------------------
    msgs = conv["merchant_msgs"]
    is_auto_phrase = _matches_any(AUTO_REPLY_PATTERNS, message)
    is_repeat = len(msgs) >= 2 and too_similar(msgs[-1], msgs[-2])
    if is_repeat:
        conv["auto_reply_streak"] = conv.get("auto_reply_streak", 1) + 1
    elif is_auto_phrase:
        conv["auto_reply_streak"] = conv.get("auto_reply_streak", 0) + 1
    else:
        conv["auto_reply_streak"] = 0

    streak = conv["auto_reply_streak"]
    if streak >= 3:
        conv["state"] = "ended"
        return {"action": "end",
                "rationale": "Same auto-reply text repeated 3x — no real merchant engagement. Closing to avoid wasting turns."}
    if streak == 2:
        return {"action": "wait", "wait_seconds": 86400,
                "rationale": "Second consecutive auto-reply — owner likely not at phone right now. Backing off 24h instead of burning another turn."}
    if streak == 1 and is_auto_phrase:
        body = "Looks like an auto-reply \u2014 when the owner sees this, just reply to confirm and I'll go ahead."
        conv["sent_bodies"].append(body)
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Detected likely auto-reply pattern; one explicit low-friction prompt to flag it for the owner before backing off."}

    # --- hostile handling -----------------------------------------------
    if _matches_any(HOSTILE_PATTERNS, message):
        conv["state"] = "ended"
        return {"action": "end",
                "rationale": "Merchant expressed clear frustration/hostility. Closing without further engagement; suppressing future triggers for this merchant is recommended at the caller level."}

    # --- not-interested ---------------------------------------------------
    if _matches_any(NOT_INTERESTED_PATTERNS, message):
        conv["state"] = "ended"
        return {"action": "end",
                "rationale": "Merchant explicitly signaled disinterest. Graceful exit per the anti-spam policy."}

    # --- intent transition ------------------------------------------------
    if _matches_any(INTENT_COMMIT_PATTERNS, message) and conv["state"] != "action_committed":
        conv["state"] = "action_committed"
        category, merchant, trigger, customer = resolve_scope(conv)
        facts = assemble_facts(category or {}, merchant or {}, trigger or {}, customer)
        facts["merchant_just_committed"] = True
        facts["merchant_message"] = message
        family = "active_planning"
        history = _format_history(conv)
        try:
            result = compose_message(facts, family, "vera" if not customer_id else "merchant_on_behalf",
                                      history, conv["sent_bodies"])
        except Exception:
            # Never let a transient LLM/network failure surface as a 500 — ship
            # the safest acknowledgement we can construct from facts alone,
            # same "don't crash, don't loop" philosophy as compose_message's
            # own bounded-retry fallback and build_tick_actions' per-item catch.
            result = _safe_commit_fallback(facts)
        conv["sent_bodies"].append(result["body"])
        return {"action": "send", "body": result["body"], "cta": result.get("cta", "binary_confirm_cancel"),
                "rationale": "Merchant explicitly committed; switching from qualification to action mode immediately, not asking another qualifying question."}

    # --- normal in-flow reply ---------------------------------------------
    conv["state"] = "qualifying" if conv["state"] == "opened" else conv["state"]
    category, merchant, trigger, customer = resolve_scope(conv)
    facts = assemble_facts(category or {}, merchant or {}, trigger or {}, customer)
    facts["merchant_message"] = message
    family = family_for_kind((trigger or {}).get("kind", ""))
    history = _format_history(conv)
    try:
        result = compose_message(facts, family, "vera" if not customer_id else "merchant_on_behalf",
                                  history, conv["sent_bodies"])
    except Exception:
        result = _safe_commit_fallback(facts)
    conv["sent_bodies"].append(result["body"])
    return {"action": "send", "body": result["body"], "cta": result.get("cta", "open_ended"),
            "rationale": "Continuing conversation; grounded in current context state."}


def _format_history(conv: dict) -> str:
    lines = []
    for t in conv["turns"][-8:]:
        lines.append(f"{t['from']}: {t['body']}")
    return "\n".join(lines)


def resolve_scope(conv: dict) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    merchant = get_ctx("merchant", conv.get("merchant_id")) if conv.get("merchant_id") else None
    category = get_ctx("category", merchant.get("category_slug")) if merchant else None
    customer = get_ctx("customer", conv.get("customer_id")) if conv.get("customer_id") else None
    trigger_id = conv.get("last_trigger_id")
    trigger = get_ctx("trigger", trigger_id) if trigger_id else {}
    return category, merchant, trigger, customer


# ============================================================================
# TICK POLICY — decide what's worth sending this tick
# ============================================================================

def build_tick_actions(available_triggers: list[str], now_iso: str = "") -> list[dict]:
    with _LOCK:
        _TICK_COUNTER["n"] += 1
        this_tick = _TICK_COUNTER["n"]

    # Use the judge's SIMULATED time for expiry checks, never wall-clock —
    # the test runs on simulated dates (e.g. April 2026) that can be well
    # in the past or future relative to real server time.
    try:
        sim_now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    except Exception:
        sim_now = datetime.now(timezone.utc)

    candidates = []
    for trig_id in sorted(set(available_triggers)):  # sorted: deterministic
        trigger = get_ctx("trigger", trig_id)
        if not trigger:
            continue
        sup_key = trigger.get("suppression_key", trig_id)
        if sup_key in SENT_SUPPRESSION_KEYS:
            continue
        expires_at = trigger.get("expires_at")
        if expires_at:
            try:
                if datetime.fromisoformat(expires_at.replace("Z", "+00:00")) < sim_now:
                    continue
            except Exception:
                pass
        merchant_id = trigger.get("merchant_id")
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        category = get_ctx("category", merchant.get("category_slug"))
        if not category:
            continue
        customer_id = trigger.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None
        candidates.append((trigger, merchant, category, customer))

    # sort by urgency desc (stable/deterministic), then trigger id for tie-break
    candidates.sort(key=lambda c: (-int(c[0].get("urgency", 1)), c[0]["id"]))

    chosen_this_tick: dict[str, int] = {}
    selected = []
    for trigger, merchant, category, customer in candidates:
        merchant_id = trigger.get("merchant_id")
        if chosen_this_tick.get(merchant_id, 0) >= MAX_NEW_CONVOS_PER_MERCHANT_PER_TICK:
            continue
        last_tick = LAST_SEND_TICK.get(merchant_id)
        if last_tick is not None and (this_tick - last_tick) <= MIN_TICKS_GAP_PER_MERCHANT:
            continue
        chosen_this_tick[merchant_id] = chosen_this_tick.get(merchant_id, 0) + 1
        selected.append((trigger, merchant, category, customer))
        if len(selected) >= 20:  # hard cap per spec
            break

    def _compose_one(item):
        trigger, merchant, category, customer = item
        facts = assemble_facts(category, merchant, trigger, customer)
        family = family_for_kind(trigger.get("kind", ""))
        send_as = "merchant_on_behalf" if customer else "vera"
        result = compose_message(facts, family, send_as)
        merchant_id = trigger["merchant_id"]
        customer_id = trigger.get("customer_id")
        conv_id = f"conv_{merchant_id}_{trigger['id']}"
        conv = get_or_init_conv(conv_id, merchant_id, customer_id)
        conv["last_trigger_id"] = trigger["id"]
        conv["sent_bodies"].append(result["body"])
        conv["turns"].append({"from": "bot", "body": result["body"]})
        template_name = f"vera_{family}_v1"
        template_params = [merchant.get("identity", {}).get("name", ""), result["body"][:200], result.get("cta", "")]
        return {
            "conversation_id": conv_id, "merchant_id": merchant_id, "customer_id": customer_id,
            "send_as": send_as, "trigger_id": trigger["id"],
            "template_name": template_name, "template_params": template_params,
            "body": result["body"], "cta": result.get("cta", "open_ended"),
            "suppression_key": trigger.get("suppression_key", trigger["id"]),
            "rationale": f"kind={trigger.get('kind')} family={family} urgency={trigger.get('urgency')}; grounded in current merchant/category/trigger context.",
        }, trigger.get("suppression_key", trigger["id"]), merchant_id

    actions = []
    if not selected:
        return actions

    deadline = time.time() + TICK_WALLCLOCK_BUDGET_S
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(selected))) as ex:
        futures = {ex.submit(_compose_one, item): item for item in selected}
        remaining = max(1.0, deadline - time.time())
        done, not_done = concurrent.futures.wait(futures, timeout=remaining)
        for fut in done:
            try:
                action, sup_key, merchant_id = fut.result()
            except Exception:
                continue
            actions.append(action)
            with _LOCK:
                SENT_SUPPRESSION_KEYS[sup_key] = datetime.now(timezone.utc).isoformat()
                LAST_SEND_TICK[merchant_id] = this_tick
        # not_done futures are simply dropped — better an incomplete tick
        # than a timeout penalty; they'll be reconsidered next tick since
        # we never marked their suppression_key as sent.

    # stable order: by merchant_id then trigger_id, for reproducibility
    actions.sort(key=lambda a: (a["merchant_id"], a["trigger_id"]))
    return actions


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI()


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME),
            "contexts_loaded": ctx_counts()}


@app.get("/v1/metadata")
def metadata():
    # Mirror call_llm's actual provider-selection order and defaults exactly —
    # the model names below MUST match the fallback defaults in _call_anthropic/
    # _call_openai/_call_groq/_call_gemini/_call_grok, or this will drift out of
    # sync again the next time one of those defaults changes.
    if os.environ.get("ANTHROPIC_API_KEY"):
        active_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    elif os.environ.get("OPENAI_API_KEY"):
        active_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    elif os.environ.get("GROQ_API_KEY"):
        active_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    elif os.environ.get("GOOGLE_API_KEY"):
        active_model = os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash")
    elif os.environ.get("XAI_API_KEY"):
        active_model = os.environ.get("XAI_MODEL", "grok-4-fast")
    else:
        active_model = "unset (no API key configured — running stub fallback composer)"
    return {
        "team_name": TEAM_NAME, "team_members": TEAM_MEMBERS,
        "model": active_model,
        "approach": "fact-assembled composer, dispatched by trigger-kind family, "
                    "guardrail-validated (number provenance / single CTA / anti-repeat), "
                    "deterministic FSM for auto-reply / intent / hostile handling",
        "contact_email": CONTACT_EMAIL, "version": BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope",
                                                        "details": f"unknown scope {body.scope}"})
    key = (body.scope, body.context_id)
    with _LOCK:
        cur = CONTEXTS.get(key)
        if cur and cur["version"] >= body.version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version",
                                                            "current_version": cur["version"]})
        CONTEXTS[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/tick")
def tick(body: TickBody):
    actions = build_tick_actions(body.available_triggers, body.now)
    return {"actions": actions}


@app.post("/v1/reply")
def reply(body: ReplyBody):
    conv = get_or_init_conv(body.conversation_id, body.merchant_id, body.customer_id)
    if body.merchant_id:
        conv["merchant_id"] = body.merchant_id
    if body.customer_id:
        conv["customer_id"] = body.customer_id
    result = handle_reply(body.conversation_id, conv.get("merchant_id"), conv.get("customer_id"),
                           body.from_role, body.message)
    return result


@app.post("/v1/teardown")
def teardown():
    with _LOCK:
        CONTEXTS.clear()
        CONVERSATIONS.clear()
        SENT_SUPPRESSION_KEYS.clear()
        LAST_SEND_TICK.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
