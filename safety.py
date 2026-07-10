"""Content gate. Runs before untrusted web text reaches the model, and again on output.

Two threats, deliberately separated:
  - prompt injection: text trying to COMMAND the model. Needs an adversary.
  - contaminated content: text the model faithfully SUMMARISES. Needs none.

Design rule: the deterministic checks always run. The LLM screening pass is an
ADDITIONAL layer that may be unavailable (rate limit, bad model id, outage). If it
fails, we log and continue on the deterministic layer -- we never silently skip
both, and we never block the whole brief on a screening outage.
"""

import re

import config

# --- Source policy -----------------------------------------------------------

BLOCKED_DOMAIN_PATTERNS = [
    r"\b4chan\b", r"\b8kun\b", r"\bkiwifarms\b", r"\bstormfront\b",
    r"\bgab\.com\b", r"\bthedirty\b", r"\bcheaterland\b",
    r"\bmugshots?\.(com|zone)\b", r"\bthatsthem\b", r"\bspokeo\b",
    r"\bwhitepages\b", r"\bbeenverified\b", r"\bradaris\b", r"\bpeoplefinders\b",
]

# Doxxing / people-search aggregators are blocked not because they're offensive but
# because they surface home addresses and relatives -- PII with no business purpose.

INJECTION_PATTERNS = [
    r"ignore (all |your |previous |prior )*(instructions|prompts?)",
    r"disregard (the )?(above|previous|prior|system)",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"</?(system|assistant|instructions?)>",
]

# Backstop only. A keyword scan is crude and will both over- and under-fire.
# The real control is the `protected` SignalType enforced in schema.gate_facts.
#
# SCOPE, deliberately narrow. Protected characteristics are: race, ethnicity,
# religion, national origin (where someone is FROM), sex, sexual orientation,
# gender identity, disability, health, age, pregnancy, citizenship status.
#
# NOT protected, and NOT blocked -- these are skills and experience:
#   language proficiency ("fluent in Spanish", "bilingual", "multilingual")
#   international experience ("lived on three continents", "worked in Singapore")
#   competencies ("cross-cultural communication", "cultural awareness")
#   descriptions of a clientele ("serves a diverse client base")
#
# Overblocking is not a safe default. A gate that suppresses legitimate
# qualifications gets switched off, and then there is no gate.
PROTECTED_TERMS = [
    # religion
    r"\b(jewish|muslim|christian|hindu|buddhist|catholic|evangelical|atheist)\b",
    r"\b(religious|religion) (background|belief|affiliation)\b",
    # race / ethnicity as identity
    r"\b(black|white|asian|hispanic|latino|latina)\s+(man|woman|men|women|executive|candidate)\b",
    r"\b(ethnic|racial)\s+(background|heritage|identity)\b",
    # sexual orientation / gender identity
    r"\b(gay|lesbian|bisexual|transgender|queer)\b",
    # disability / health
    r"\b(disabled|disability|wheelchair|autistic|adhd|bipolar|depression|cancer survivor)\b",
    # pregnancy
    r"\b(pregnant|pregnancy|maternity leave)\b",
    # national origin: where someone is FROM, and immigration status.
    # "native speaker" is here because EEOC treats native-speaker requirements as a
    # national-origin proxy -- unlike fluency, which is a skill.
    r"\b(immigrant|undocumented|visa status|green card|naturalized|citizenship status)\b",
    r"\bnative (speaker|language)\b",
    r"\bcountry of (origin|birth)\b",
    r"\b(born|raised) in [A-Z][a-z]+\b",
    # age
    r"\b\d{1,2} years old\b", r"\bborn in (19|20)\d{2}\b",
]


class SafetyResult:
    def __init__(self) -> None:
        self.dropped_sources: list[str] = []
        self.flags: list[str] = []
        self.screening_available: bool = True


def domain_blocked(url: str) -> bool:
    u = (url or "").lower()
    return any(re.search(p, u) for p in BLOCKED_DOMAIN_PATTERNS)


def has_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def scan_protected(text: str) -> list[str]:
    """Return matched protected-characteristic patterns. Backstop, not the control."""
    t = (text or "").lower()
    return [p for p in PROTECTED_TERMS if re.search(p, t)]


def screen_sources(sources: list[dict]) -> tuple[list[dict], SafetyResult]:
    """Deterministic pre-filter. Always runs, never needs the network."""
    result = SafetyResult()
    kept = []
    for s in sources:
        if domain_blocked(s.get("url", "")):
            result.dropped_sources.append(f"{s.get('url')} (blocked domain)")
            continue
        if has_injection(s.get("content", "")):
            result.dropped_sources.append(f"{s.get('url')} (injection pattern)")
            continue
        kept.append(s)
    return kept, result


def llm_screen(sources: list[dict], result: SafetyResult, call_fn) -> list[dict]:
    """Optional second layer using a small safety model.

    ONE call for all sources, not one call per source. Per-source screening meant
    dozens of model calls per brief, which on a rate-limited tier means backoff,
    retry, and a three-minute brief. The deterministic layer already ran.

    `call_fn(text) -> str` returns a verdict. On ANY failure we mark screening
    unavailable and return sources unchanged: a screening outage must not take down
    the brief, and it must not silently approve either -- it is recorded in flags.
    """
    if not config.SAFETY_SCREENING_ENABLED or not sources:
        result.screening_available = False
        result.flags.append("LLM screening disabled by config")
        return sources

    batch = []
    for i, s in enumerate(sources[: config.SAFETY_MAX_SOURCES], 1):
        batch.append(f"[{i}] {s.get('content','')[:800]}")
    payload = "\n\n".join(batch)

    try:
        verdict = (call_fn(payload) or "").upper()
    except Exception as e:  # noqa: BLE001 - deliberate: never fail the brief
        result.screening_available = False
        result.flags.append(
            f"LLM screening unavailable ({type(e).__name__}); deterministic checks applied"
        )
        return sources

    if "SAFE" in verdict and "UNSAFE" not in verdict:
        return sources

    # Verdict names the unsafe indices, e.g. "UNSAFE: 2, 5"
    bad = {int(n) for n in re.findall(r"\d+", verdict)}
    kept = []
    for i, s in enumerate(sources, 1):
        if i in bad:
            result.dropped_sources.append(f"{s.get('url')} (screened unsafe)")
            continue
        kept.append(s)
    return kept


class ProtectedContentError(RuntimeError):
    """Raised when protected characteristics reach output in a hiring context."""


def check_output(markdown: str, hiring_context: bool = False) -> list[str]:
    """Post-generation scan. A claim can be assembled from clean sources and still
    land somewhere it shouldn't.

    Outside a hiring context this warns. Inside one -- a role spec was supplied, so a
    hiring decision is in scope -- it RAISES. A brief that infers national origin from
    someone's languages is a Title VII problem, not a quality problem, and the model
    demonstrably will not tag these `protected` on its own.
    """
    hits = scan_protected(markdown)
    msgs = [f"output matched protected-characteristic pattern: {h}" for h in hits]
    if hits and hiring_context:
        raise ProtectedContentError(
            "Blocked: the brief contains protected-characteristic language and a role "
            "spec was supplied, so a hiring decision is in scope.\n\nMatched: "
            + ", ".join(hits)
            + "\n\nThis is a compliance stop, not a bug. Remove personal or "
            "national-origin material from the background text, or generate the brief "
            "without a role spec."
        )
    return msgs


def identity_confidence(sources: list[dict], name: str) -> str:
    """How many sources corroborate that this is the SAME person.

    Counts `_match` strength set by pipeline.subject_match_strength: a source that
    only shares a surname does not raise confidence, because that is exactly what a
    different person with the same name looks like.
    """
    if not name:
        return "low"
    corroborated = sum(1 for s in sources if s.get("_match", 0) >= 2)
    if corroborated >= config.IDENTITY_MIN_SOURCES + 1:
        return "high"
    if corroborated >= config.IDENTITY_MIN_SOURCES:
        return "medium"
    return "low"
