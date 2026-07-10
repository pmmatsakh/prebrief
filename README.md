# Pre-Meeting Brief Agent

A deterministic, multi-step agent with tool-calling, structured output, and a
content gate. Give it a name + company and it produces a one-page meeting brief:
background, dated recent activity, inferred priorities, three questions, and — if
you supply a profile — how you're connected.

## Architecture

**Deterministic orchestration.** `pipeline.py` is fixed code: screen → search →
gate → synthesize → gate → render. The LLM is confined to one step and never
drives control flow.

**Content gate** (`safety.py`). Untrusted web text is screened before it reaches
the model: blocked-domain policy, prompt-injection detection, optional screening by
a small safety model, and a protected-characteristic scan of the *output*. The
deterministic checks always run; if the screening model is unavailable the brief
continues on them rather than failing open or falling over.

**Signal typing is a compliance control.** Every fact is typed `professional` /
`personal` / `civic` / `protected` / `uncertain`. Facts typed `protected` never
reach output. `likely_priorities` draws only on `professional`. Enforced in
`schema.gate_facts()` — in code, not by asking the model nicely.

**Connections are set intersection, not generation** (`affinity.py`). Employer,
school, location, and second-degree overlaps are computed deterministically against
`profile.yaml` and `network.yaml`. No LLM. It cannot hallucinate, because it never
generates. Second-degree ties are co-employment inference from public history —
LinkedIn does not export second-degree connections and scraping violates its ToS.

**Absent input is never invention.** No `profile.yaml` → the connections section is
omitted. This rule exists because v1, given an empty user field, invented a user by
borrowing the subject's attributes.

**Recency.** Sources carry dates into the prompt, are sorted by date *before*
truncation, and `recent_activity` requires a date — undated material is demoted to
background.

**Interface seam.** `pipeline.generate_brief()` is a pure function with no UI
dependency: the surface to expose as an MCP tool or HTTP endpoint.

## Stack

Python · Streamlit · any OpenAI-compatible LLM endpoint (default: Groq serving
`openai/gpt-oss-120b`) · Tavily search · Pydantic.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY and TAVILY_API_KEY
cp profile.yaml.example profile.yaml   # optional: enables connections
cp network.yaml.example network.yaml   # optional: enables second-degree
streamlit run app.py
```

## Tests

```bash
python -m pytest test_deterministic.py -q
```

17 tests over every path that doesn't touch the network — which is where the
compliance controls live.

## Swapping the model or host

```bash
LLM_BASE_URL=https://api.together.xyz/v1   # or a self-hosted vLLM endpoint
LLM_MODEL=<model-id>
```

## Known limitations

- Subject employment history is supplied by hand, not extracted. Deliberate:
  connections computed from LLM-extracted history would forfeit the
  "cannot hallucinate" guarantee.
- No org-context tier yet — people at large companies get the same shape as
  startup founders.
- The protected-characteristic output scan is a regex backstop. It will over- and
  under-fire. The real control is the `protected` type in `gate_facts()`.
- `published_date` is reliable on Tavily's news topic, spotty on general results.
- No eval set has been run yet. See `eval_cases.yaml`.
