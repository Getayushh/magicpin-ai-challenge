# Vera-challenge bot — AYUSH GUPTA

## Approach

`compose()` is split into three stages instead of one LLM call, on purpose:

1. **Fact assembler** (`assemble_facts`, pure code) — resolves category + merchant +
   trigger + customer from the context store into a small, explicit facts packet.
   The LLM **never sees the raw context blobs** — only this packet. That's the main
   anti-fabrication mechanism: it's structural, not just a prompt instruction.
2. **Kind-family dispatch** (`FAMILY_MAP` / `FAMILY_FRAMING`) — ~30 known trigger kinds
   route to one of 10 framing families (knowledge digest, performance, customer
   lifecycle, urgent compliance, curious-ask, competitive, review-theme, external
   event, account lifecycle, active-planning). **Any kind not in the map falls
   through to a `generic` family** that's still fact-grounded rather than erroring —
   this is the safety net for trigger kinds the judge injects that weren't in the
   30 canonical pairs.
3. **Guardrail validator** (`compose_message`) — after the LLM returns, checks number
   provenance against the facts packet, single-CTA shape, and anti-repetition against
   that conversation's sent history. One bounded re-prompt with the specific problem
   named; if still broken, ships the safest available version rather than looping or
   erroring (an empty/malformed action is worse than an imperfect one).

**Conversation state** (`handle_reply`) is a small deterministic FSM, not "ask the LLM
to remember": auto-reply detection escalates `send → wait(24h) → end` on repeated
canned text (regex phrase match + similarity-based repeat detection); explicit
commitment language forces a hard transition out of qualifying mode into an
action-mode prompt (no further qualifying questions); hostile language ends the
conversation immediately. These are cheap and provably correct on the exact
scenarios the replay test scripts, rather than "probably handled well" by an LLM.

**Tick policy** (`build_tick_actions`) ranks available triggers by urgency, respects
each trigger's own `suppression_key` and `expires_at` (checked against the judge's
*simulated* `now`, not wall-clock time), and caps sends to one new conversation per
merchant per tick with a minimum gap between ticks per merchant — restraint is
scored, so the policy is built to not fire on everything available.

## Why this should hold up against fresh/unseen scenarios

The brief is explicit that scoring depends on how the bot handles context it hasn't
seen before, not the 30 known pairs. Concretely, that's addressed by:
- facts are re-assembled fresh from the context store on **every** compose call, so a
  mid-test version bump (new digest item, updated performance) is picked up
  automatically — nothing is cached into a stale prompt.
- unknown `trigger.kind` values fall through to the grounded `generic` family instead
  of crashing or ignoring the trigger.
- the reply FSM keys off message *patterns* (auto-reply phrasing, commitment
  language, hostility), not scripted turn numbers, so it isn't tied to the specific
  wording in the 3 documented replay examples.

## Run it

```bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY=sk-...      # or OPENAI_API_KEY / GOOGLE_API_KEY
uvicorn bot:app --host 0.0.0.0 --port 8080
```

With no key set, the composer falls back to a clearly-labeled stub string — the
service still runs end-to-end (useful for wiring checks) but will score badly; set a
real key before any real test.

## Test it

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py          # needs its own LLM_API_KEY configured at the top of the file
```

I manually exercised the three replay-test equivalents (auto-reply ladder, hostile,
intent transition) via curl against a live instance during development — all three
produced the documented-correct `send`/`wait`/`end` decisions.

## Known gaps / what I'd do with more time

- **`dataset/categories/*.json` here are stub CategoryContexts I authored**, not the
  official ones — I don't have the starter zip's real category files. They match the
  documented schema and use only real content from the brief/case-studies where it
  exists (dentists is the most complete, since real examples were provided). **Swap
  these for the official files if/when available** — nothing else in the bot depends
  on their exact content, only their shape.
- Number-provenance checking is a heuristic (substring match against the facts
  packet), not a hard guarantee — it catches obvious fabrication, not every case.
- The intent-transition and hostile detectors are regex/pattern-based for speed.
  Given more time I'd add an LLM-based classifier as a fallback for phrasing the
  regex misses, with the regex as a fast-path.
- Customer-facing (`merchant_on_behalf`) composition is wired through the same
  pipeline but has had less live testing than the merchant-facing path in the time
  available — worth extra manual checks against `customers_seed.json` before the
  real run.
- No persistence beyond process memory — matches the spec ("in-memory is fine"), but
  means a process restart mid-test loses state. Deploy somewhere that won't
  autoscale-to-zero mid-test.

## Deploy

Any host that gives a public URL works. Fastest options for a same-day deploy:
Railway or Render (connect the repo, set the env var, done), or a Fly.io app. Set
`PORT` if your platform requires it (the app reads `$PORT`, defaults to 8080).
