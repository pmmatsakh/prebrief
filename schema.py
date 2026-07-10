"""The brief as a schema. Signal typing is a compliance control, not a style choice.

A fact typed `protected` never reaches output. A fact typed `personal` or `civic`
may appear in Personal notes but can never drive `likely_priorities`. Enforced in
code (gate_facts), not by asking the model nicely.
"""

import re
from enum import Enum

from pydantic import BaseModel, Field


class MeetingType(str, Enum):
    UNSPECIFIED = "unspecified"
    COFFEE_CHAT = "coffee_chat"
    INFORMATIONAL = "informational"
    RECRUITER_SCREEN = "recruiter_screen"
    HIRING_MANAGER = "hiring_manager"


class SignalType(str, Enum):
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    CIVIC = "civic"
    PROTECTED = "protected"   # race, religion, health, national origin, age, etc.
    UNCERTAIN = "uncertain"


class Fact(BaseModel):
    text: str
    type: SignalType
    source_index: int | None = Field(default=None, description="1-based index into sources")
    date: str | None = Field(default=None, description="ISO date if known, else null")


class Inference(BaseModel):
    """A claim the sources support but do not state.

    Distinct from Fact. A fact is IN a source and cites exactly one. An inference is
    a judgement ACROSS sources and cites the ones it rests on. Requiring inferences to
    be verbatim-sourced is a category error: it deletes exactly the synthesis that
    makes a brief worth more than a search result.

    `basis` is what keeps it honest -- an inference must name the evidence it stands on,
    and code checks those indices exist.
    """
    claim: str
    basis: list[int] = Field(description="1-based source indices this rests on")


class Question(BaseModel):
    question: str
    why: str


class ProofPoint(BaseModel):
    """Evidence from the candidate's background that meets a stated requirement.

    NOT deterministic, unlike Connection. The LLM does the matching, so both
    citations are mandatory: a proof point that cannot name the requirement it
    answers and the evidence it rests on is dropped by gate_role_sections().
    """
    requirement: str = Field(description="Verbatim-ish requirement from the role spec")
    evidence: str = Field(description="What in the candidate's background meets it")


class Gap(BaseModel):
    requirement: str
    note: str = Field(description="Honest framing: what's missing and how to address it")


class Connection(BaseModel):
    """Deterministically derived. Never produced by the LLM."""
    kind: str            # employer | school | location | second_degree
    detail: str
    via: str | None = None


class Brief(BaseModel):
    person_name: str
    company: str
    headline: str
    identity_confidence: str = Field(description="high | medium | low")
    background: list[Fact] = Field(default_factory=list)
    recent_activity: list[Fact] = Field(default_factory=list)
    observations: list[Inference] = Field(
        default_factory=list,
        description="Non-obvious things true across sources. The value-add.",
    )
    likely_priorities: list[Inference] = Field(default_factory=list)
    personal_notes: list[Fact] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    # --- Role-aware sections. Populated ONLY when a role spec is supplied. ---
    angle: str | None = Field(
        default=None,
        description="Why this person should care about this candidate for this role",
    )
    proof_points: list[ProofPoint] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)


def _overlaps_background(fact_text: str, background: str) -> bool:
    """True if a 'fact about the subject' is really a phrase from the READER's resume.

    GLM correctly noted in caveats that the candidate background was the reader's, then
    rendered a line of it as a personal note about the subject anyway. A strong reasoner
    that states the rule and breaks it needs a deterministic backstop, not more prompt.
    """
    if not background:
        return False
    ft = set(w for w in re.findall(r"[a-z]{4,}", fact_text.lower()))
    bg = set(w for w in re.findall(r"[a-z]{4,}", background.lower()))
    if not ft:
        return False
    overlap = len(ft & bg) / len(ft)
    return overlap >= 0.6   # most content words of the "fact" come from the resume


def gate_facts(
    brief: Brief, n_sources: int = 0, candidate_background: str = ""
) -> tuple[Brief, list[str]]:
    """Enforce the signal-type contract AND the grounding contract.

    Grounding: every fact ABOUT THE SUBJECT must cite a source index that exists.
    Uncited facts are dropped. This is what stops candidate-supplied text (a resume,
    a job spec) from being laundered into facts about the person being met -- the
    resume is not a source, so nothing in it can survive this gate.
    """
    actions: list[str] = []

    def require_source(facts: list[Fact], section: str) -> list[Fact]:
        kept = [
            f for f in facts
            if f.source_index is not None and 1 <= f.source_index <= n_sources
        ]
        if len(kept) != len(facts):
            actions.append(
                f"dropped {len(facts)-len(kept)} ungrounded fact(s) from {section} "
                "(no valid source_index)"
            )
        return kept

    for field in ("background", "recent_activity", "personal_notes"):
        setattr(brief, field, require_source(getattr(brief, field), field))

    def require_basis(items: list[Inference], section: str) -> list[Inference]:
        """An inference must cite the sources it rests on. It need not be IN them."""
        kept = []
        for inf in items:
            valid = [b for b in inf.basis if 1 <= b <= n_sources]
            if valid:
                inf.basis = valid
                kept.append(inf)
        if len(kept) != len(items):
            actions.append(f"dropped {len(items)-len(kept)} unsupported {section}")
        return kept

    brief.observations = require_basis(brief.observations, "observations")
    brief.likely_priorities = require_basis(brief.likely_priorities, "likely_priorities")

    def drop_protected(facts: list[Fact], section: str) -> list[Fact]:
        kept = [f for f in facts if f.type != SignalType.PROTECTED]
        if len(kept) != len(facts):
            actions.append(f"dropped {len(facts)-len(kept)} protected fact(s) from {section}")
        return kept

    brief.background = drop_protected(brief.background, "background")
    brief.recent_activity = drop_protected(brief.recent_activity, "recent_activity")
    brief.personal_notes = drop_protected(brief.personal_notes, "personal_notes")

    # Reject any "fact about the subject" that is really lifted from the reader's resume.
    if candidate_background:
        for field in ("background", "recent_activity", "personal_notes"):
            facts = getattr(brief, field)
            kept = [f for f in facts if not _overlaps_background(f.text, candidate_background)]
            if len(kept) != len(facts):
                actions.append(
                    f"dropped {len(facts)-len(kept)} {field} item(s) that were candidate "
                    "resume text, not facts about the subject"
                )
            setattr(brief, field, kept)

    dated, undated = [], []
    for f in brief.recent_activity:
        (dated if f.date else undated).append(f)
    if undated:
        brief.recent_activity = dated
        brief.background.extend(undated)
        actions.append(f"moved {len(undated)} undated item(s) to background")

    return brief, actions


def gate_role_sections(
    brief: Brief, has_role: bool, has_background: bool
) -> tuple[Brief, list[str]]:
    """Absent input is never invention -- the same rule that fixed the invented user.

    No role spec  -> no proof points, no gaps, no angle.
    No background -> no proof points (nothing to match against).
    A proof point missing either citation is dropped.
    """
    actions: list[str] = []

    if not has_role:
        for field in ("proof_points", "gaps"):
            if getattr(brief, field):
                actions.append(f"dropped {len(getattr(brief, field))} {field} (no role spec supplied)")
                setattr(brief, field, [])
        if brief.angle:
            actions.append("dropped angle (no role spec supplied)")
            brief.angle = None

    if not has_background and brief.proof_points:
        actions.append(f"dropped {len(brief.proof_points)} proof points (no candidate background supplied)")
        brief.proof_points = []

    before = len(brief.proof_points)
    brief.proof_points = [
        p for p in brief.proof_points if p.requirement.strip() and p.evidence.strip()
    ]
    if len(brief.proof_points) != before:
        actions.append(f"dropped {before - len(brief.proof_points)} uncited proof point(s)")

    return brief, actions


def render_markdown(brief: Brief, sources: list[dict],
                    connections: list[Connection] | None = None) -> str:
    """Deterministic render. Empty sections are omitted, never filled."""
    title = brief.person_name + (f" - {brief.company}" if brief.company else "")
    lines = [f"# {title}", ""]

    if brief.identity_confidence == "low":
        lines += ["> **Low identity confidence.** Sources may describe more than one person "
                  "with this name. Verify before relying on anything below.", ""]

    lines += [f"**{brief.headline}**", ""]

    def section(heading: str, facts: list) -> None:
        if not facts:
            return
        lines.append(f"## {heading}")
        for f in facts:
            date = f" *({f.date})*" if f.date else ""
            src = f" [{f.source_index}]" if f.source_index else ""
            lines.append(f"- {f.text}{date}{src}")
        lines.append("")

    def inferences(heading: str, items: list, note: str = "") -> None:
        if not items:
            return
        lines.append(f"## {heading}")
        if note:
            lines.append(f"*{note}*")
        for inf in items:
            cites = "".join(f" [{b}]" for b in inf.basis)
            lines.append(f"- {inf.claim}{cites}")
        lines.append("")

    section("Background", brief.background)
    section("Recent activity", brief.recent_activity)
    inferences(
        "What this adds up to",
        brief.observations,
        "Judgements across sources, not statements from them.",
    )
    inferences("Likely priorities *(inferred)*", brief.likely_priorities)

    if connections:
        lines.append("## How you're connected")
        for c in connections:
            via = f" (via {c.via})" if c.via else ""
            lines.append(f"- **{c.kind}**{via}: {c.detail}")
        lines.append("")

    if brief.angle:
        lines += ["## Your angle", brief.angle, ""]

    if brief.proof_points:
        lines.append("## Proof points")
        for p in brief.proof_points:
            lines.append(f"- **{p.requirement}** -> {p.evidence}")
        lines.append("")

    if brief.gaps:
        lines += ["## Gaps", "*What the role asks for that you can't yet evidence.*"]
        for g in brief.gaps:
            lines.append(f"- **{g.requirement}**: {g.note}")
        lines.append("")

    if brief.questions:
        lines.append("## Three questions worth asking")
        for i, q in enumerate(brief.questions, 1):
            lines += [f"{i}. **{q.question}**", f"   - *{q.why}*"]
        lines.append("")

    if brief.personal_notes:
        lines += ["## Personal notes",
                  "*Rapport material only. Not evidence of professional priorities.*"]
        lines += [f"- {f.text}" for f in brief.personal_notes]
        lines.append("")

    if brief.caveats:
        lines += ["## Caveats"] + [f"- {c}" for c in brief.caveats] + [""]

    if sources:
        lines.append("## Sources")
        for i, s in enumerate(sources, 1):
            d = f" - {s['date']}" if s.get("date") else ""
            lines.append(f"{i}. [{s['title']}]({s['url']}){d}")

    return "\n".join(lines)
