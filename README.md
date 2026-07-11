# Pre-Meeting Brief Agent

Input a name and company; get a one-page brief: who they are, dated recent
activity, what it adds up to, three questions worth asking, and — with an optional
job spec and résumé — how a candidate maps to the role.

Runs locally or deployed. OpenAI-compatible LLM endpoint (model and host are
config), Tavily search, Streamlit UI, Pydantic-validated output.

---

## Architecture

> **Keep this current.** This section is the map of the system. If you add, remove,
> or repurpose a module or a gate, update the matching row here in the same commit.
> A stale map is worse than none. The two tables below should always match the
> actual files (`ls *.py`) and the actual gate list in `schema.py` / `safety.py`.

**One idea runs through everything:** a strong model will state a rule correctly and
then break it. So the LLM is confined to a single synthesis step, and every
correctness property is enforced by deterministic code *around* it — not by asking
the model nicely. Those enforcement points are called **gates**. Adding a
capability usually means adding a gate and a test, not changing the prompt.

### The pipeline (control flow)

`pipeline.py` is the spine. Fixed order, LLM touches one step:

```
parse input
  → search (concurrent: company + person + social passes)
  → identity filter        (drop wrong-person / uncorroborated sources)
  → screen sources         (blocked domains, injection, optional LLM screen)
  → date-sort + truncate
  → SYNTHESIZE  ← the only LLM step
  → gate facts             (grounding, signal-type, résumé-leak, recency)
  → gate role sections     (proof points need a role spec + background)
  → check questions        (repair inverted questions)
  → org context            (company: cached/shared · unit: per-person)
  → protected-output scan
  → render markdown + audit trail
```

### Modules (one concern each — 10 files)

| File | Owns | LLM? |
|---|---|---|
| `pipeline.py` | Orchestration, retrieval, identity filter. The spine. | no |
| `schema.py` | The `Brief` shape **and** the fact-level gates (`gate_facts`, `gate_role_sections`, grounding, résumé-overlap, render). | no |
| `safety.py` | Content gate: domain policy, injection, protected-characteristic scan, identity confidence. | no |
| `affinity.py` | Connections by set intersection over `profile.yaml` / `network.yaml`. Cannot hallucinate. | no |
| `orgcontext.py` | Company context (cached, **shared** per firm) and business unit (**per-person**, never cached). | yes |
| `llm.py` | OpenAI-compatible client; synthesis, screening, and the inverted-question / empty-section repair passes. | yes |
| `tools.py` | Web search + date parsing. The seam for swapping search providers. | no |
| `config.py` | Every external dependency as an env var. Model/host/search are one-line swaps. | no |
| `app.py` | Streamlit UI. Thin — all logic is in `pipeline.py`. | no |
| `test_deterministic.py` | 105 tests over every non-network path — i.e. every gate. | no |

### The gates (correctness properties, enforced in code)

Each row is a real defect found against a real person, now prevented by code and
covered by a test. **This is the actual IP.** When you add a gate, add a row.

| Gate | Prevents | Where |
|---|---|---|
| Grounding | A fact about the subject with no real source | `schema.gate_facts` |
| Identity corroboration | Merging a different person who shares a name | `pipeline.subject_match_strength` |
| Self-authored trust | Dropping the subject's own posts/profile for not naming the company | `pipeline._is_self_authored` |
| Roster corroboration | Dropping a donor list / directory that names the subject but no employer | `pipeline._is_roster` |
| Résumé-leak | The reader's résumé rendered as facts about the subject | `schema._overlaps_background` |
| Recency | Undated/old material read as current activity | `schema.gate_facts` + date sort |
| Signal typing | Protected characteristics reaching output; civic ≠ professional | `schema` + `safety.scan_protected` |
| Social-activity | A *like* read as biography ("liked HBS post" → "attended HBS") | `pipeline` SOCIAL tag + prompt |
| Min-synthesis | An empty observations/priorities section shipping | `llm._missing_sections` + repair |
| Question direction | Questions aimed *at* the candidate instead of the subject | `llm._inverted_questions` + repair |
| Unit scoping | One person's business unit shown for every colleague at the firm | `orgcontext.resolve_unit` (per-person, uncached) |
| Refusal floor | A confident brief on a wrong/misspelled name with nothing corroborating | `pipeline` refusal + spelling flag |

### Caching rule (learned the hard way)

Anything cached must be keyed at the level it actually varies. **Company** facts
(industry, company situation) are shared across everyone at a firm → cached by
company. **Person** facts (role, business unit) vary per individual → never cached
by company. Violating this is what once showed every Citi contact in the same unit.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # add TAVILY_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
cp profile.yaml.example profile.yaml     # optional: enables connections
cp network.yaml.example network.yaml     # optional: enables second-degree
streamlit run app.py
```

## Tests

```bash
python -m pytest test_deterministic.py -q     # expect: 105 passed
```

Every gate above has a regression test. A green run means no gate regressed.

## Swapping model, host, or search provider

All are env vars — no code change:

```bash
LLM_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_MODEL=accounts/fireworks/routers/glm-5p2-fast
LLM_REASONING_EFFORT=none          # reasoning models: cut latency
```

## Known limitations

- Industry-vs-company separation in org context is prompt-enforced, not code-enforced.
- Thin-footprint subjects yield thin briefs; starved retrieval can surface a
  wrong-person name (flagged in caveats).
- No scored eval set yet — gates are regression tests, not a graded benchmark.
