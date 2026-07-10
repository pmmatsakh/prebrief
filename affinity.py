"""Deterministic connection surfacing. No LLM. Set intersection over structured data.

This is the module that answers "how do you know this?" with a date range instead
of a shrug. It cannot hallucinate, because it never generates -- it only intersects.

Fixes three observed failures at once:
  - invented user ("Philip studies CS at Columbia")  -> facts come from profile.yaml
  - missed second-degree tie (Callender / Matsakh)   -> co-employment inference
  - missed alumni tie (NYU)                          -> school overlap
"""

from pathlib import Path

import yaml

import config
from schema import Connection


def _norm_org(name: str) -> str:
    """Accenture LLP == Accenture. Crude; never fully done."""
    s = (name or "").lower().strip()
    for suffix in (" llp", " llc", " inc.", " inc", " ltd", " limited",
                   " corp.", " corp", " group", " plc", " co.", " & co"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.replace(",", "").replace(".", "").strip()


def _overlaps(a_start, a_end, b_start, b_end) -> tuple[bool, int, int] | tuple[bool, None, None]:
    """Inclusive year-range overlap. None means open-ended."""
    a_start = a_start or -9999
    a_end = a_end or 9999
    b_start = b_start or -9999
    b_end = b_end or 9999
    lo, hi = max(a_start, b_start), min(a_end, b_end)
    if lo <= hi:
        return True, lo, hi
    return False, None, None


def load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def load_profile() -> dict:
    return load_yaml(config.PROFILE_PATH)


def load_network() -> list[dict]:
    data = load_yaml(config.NETWORK_PATH)
    return data.get("people", []) if isinstance(data, dict) else []


def _affiliations(entity: dict, key: str) -> list[dict]:
    """Tolerate hand-edited YAML.

    All of these mean the same thing:
        schools: [NYU]
        schools:
          - NYU
          - name: NYU
            start: 2023
    A bare string becomes {"name": ...} with open-ended dates. Crashing on a human's
    reasonable YAML is a bug in the reader, not the file.
    """
    raw = entity.get(key, []) or []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"name": item, "start": None, "end": None})
        elif isinstance(item, dict) and item.get("name"):
            out.append(item)
    return out


def _compare(subject: dict, other: dict, other_label: str | None) -> list[Connection]:
    """Intersect one person's affiliations against another's."""
    out: list[Connection] = []

    for kind, field in (("employer", "employers"), ("school", "schools")):
        for s in _affiliations(subject, field):
            for o in _affiliations(other, field):
                if _norm_org(s.get("name", "")) != _norm_org(o.get("name", "")):
                    continue
                ok, lo, hi = _overlaps(
                    s.get("start"), s.get("end"), o.get("start"), o.get("end")
                )
                org = s.get("name")
                if ok and lo and lo > -9999 and hi < 9999:
                    detail = f"Both at {org}, {lo}-{hi}" if lo != hi else f"Both at {org} in {lo}"
                elif ok:
                    detail = f"Both at {org}"
                else:
                    # Same institution, different eras. Still worth knowing, stated honestly.
                    detail = (f"Both {org}, different periods "
                              f"({s.get('start','?')}-{s.get('end','?')} vs "
                              f"{o.get('start','?')}-{o.get('end','?')})")
                out.append(
                    Connection(
                        kind=kind if ok else f"{kind} (no overlap)",
                        detail=detail,
                        via=other_label,
                    )
                )

    def _cities(e: dict) -> set[str]:
        raw = e.get("cities", []) or []
        if isinstance(raw, str):
            raw = [raw]
        return {str(c).lower() for c in raw if c}

    subj_cities, other_cities = _cities(subject), _cities(other)
    for city in sorted(subj_cities & other_cities):
        out.append(Connection(kind="location", detail=f"Both in {city.title()}", via=other_label))

    return out


def find_connections(subject: dict, profile: dict, network: list[dict]) -> list[Connection]:
    """subject: entities extracted about the person being briefed.

    Returns first-degree overlaps (subject <-> profile) and second-degree
    overlaps (subject <-> someone in the user's network).
    """
    if not subject:
        return []

    connections: list[Connection] = []
    if profile:
        connections.extend(_compare(subject, profile, other_label=None))

    for person in network:
        label = person.get("name", "someone in your network")
        rel = person.get("relationship")
        via = f"{label} ({rel})" if rel else label
        for c in _compare(subject, person, other_label=via):
            c.kind = f"second-degree {c.kind}"
            connections.append(c)

    # Stable, deterministic ordering: direct ties first, overlapping before non-overlapping.
    connections.sort(key=lambda c: ("second-degree" in c.kind, "no overlap" in c.kind))
    return connections


def parse_entities(text: str) -> dict:
    """Parse 'Accenture 2011-2015; NYU 2001-2005; London' into structured entities.

    Deliberately deterministic. Connections must never rest on LLM-extracted history,
    or the 'cannot hallucinate' guarantee evaporates.
    """
    import re as _re

    out: dict = {"employers": [], "schools": [], "cities": []}
    if not text:
        return out

    SCHOOL_HINTS = ("university", "college", "school", "institute", "nyu", "mit",
                    "harvard", "columbia", "stanford", "yale", "princeton")

    for part in _re.split(r"[;\n]", text):
        part = part.strip()
        if not part:
            continue
        m = _re.search(r"(.*?)\s*(\d{4})\s*[-–]\s*(\d{4}|present)?\s*$", part, _re.I)
        if m:
            name = m.group(1).strip().rstrip(",")
            start = int(m.group(2))
            end_raw = m.group(3)
            end = None if (not end_raw or end_raw.lower() == "present") else int(end_raw)
            entry = {"name": name, "start": start, "end": end}
            key = "schools" if any(h in name.lower() for h in SCHOOL_HINTS) else "employers"
            out[key].append(entry)
        else:
            if any(h in part.lower() for h in SCHOOL_HINTS):
                out["schools"].append({"name": part, "start": None, "end": None})
            elif len(part.split()) <= 3 and part[0].isupper():
                out["cities"].append(part)
            else:
                out["employers"].append({"name": part, "start": None, "end": None})
    return out
