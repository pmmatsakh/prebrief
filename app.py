"""Streamlit UI. Thin: all logic lives in pipeline.py.

Run:  streamlit run app.py
"""

import os

import streamlit as st

import affinity
import config
import pipeline
import safety
import tools
from llm import SynthesisError
from schema import MeetingType

st.set_page_config(page_title="Pre-Meeting Brief Agent", page_icon="[]", layout="centered")


def _gate() -> bool:
    """Password gate. No APP_PASSWORD set (local dev) -> open."""
    pw = config.APP_PASSWORD or os.getenv("APP_PASSWORD")
    if not pw:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("Pre-Meeting Brief Agent")
    entered = st.text_input("Password", type="password")
    if entered and entered == pw:
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("Incorrect password")
    return False


if not _gate():
    st.stop()

st.title("Pre-Meeting Brief Agent")
st.caption(
    f"Deterministic pipeline - screen, search, synthesize, gate. "
    f"The LLM ({config.LLM_MODEL}) touches one step. Connections are set intersection, not generation."
)

problems = config.validate()
if problems:
    for p in problems:
        st.error(p)
    st.stop()

raw_input = st.text_input(
    "Who are you meeting?", placeholder="Jane Smith, Acme Corp - or a LinkedIn URL"
)
if raw_input and not raw_input.startswith("http") and not any(
    sep in raw_input for sep in (",", "@", "|", " - ")
):
    st.warning(
        "No company given. Without one, a common name cannot be corroborated and the "
        "brief may describe several different people. Try `Irina Berg, Citi`."
    )

MEETING_LABELS = {
    "Not sure / not specified": MeetingType.UNSPECIFIED,
    "Coffee chat": MeetingType.COFFEE_CHAT,
    "Informational interview": MeetingType.INFORMATIONAL,
    "Recruiter screen": MeetingType.RECRUITER_SCREEN,
    "Hiring manager round": MeetingType.HIRING_MANAGER,
}

meeting_label = st.selectbox(
    "What kind of meeting is it?",
    list(MEETING_LABELS.keys()),
    help="Changes the register and what the three questions probe. Leave as-is if unsure.",
)
meeting_type = MEETING_LABELS[meeting_label]

with st.expander("Optional: the role you're interviewing for"):
    st.caption(
        "Paste the job description. Unlocks the angle, proof points and gaps sections. "
        "Leave blank and those sections are omitted - never guessed."
    )
    role_spec = st.text_area(
        "Job spec", height=140, placeholder="Paste the posting here...",
        label_visibility="collapsed",
    )

with st.expander("Optional: your background"):
    st.caption(
        "Paste your resume, or a summary of relevant projects and artifacts. Matched "
        "against the role spec above. Skills and evidence only - nothing personal."
    )
    candidate_background = st.text_area(
        "Your background", height=140,
        placeholder="Projects, roles, artifacts, coursework...",
        label_visibility="collapsed",
    )

with st.expander("Optional: what you already know about them"):
    st.caption(
        "Employers and schools with years, semicolon-separated. Used for deterministic "
        "connection surfacing against your profile.yaml. Leave blank and the connections "
        "section is omitted - never guessed."
    )
    known = st.text_input(
        "Their history", placeholder="Accenture 2011-2015; NYU 2001-2005; London"
    )
    st.caption(
        "Also used to sharpen the search. A role or domain helps most: "
        "`nonprofit commercial bank`, `head of research`."
    )

include_org = st.checkbox(
    "Include company / industry context",
    value=False,
    help="One extra call. Cached per company, so every later contact at the same firm "
         "reuses it for free.",
)

profile_loaded = bool(affinity.load_profile())
if not profile_loaded:
    st.info(
        "No `profile.yaml` found. Copy `profile.yaml.example` and fill it in to enable "
        "the connections section. Without it, nothing about you is asserted."
    )

if st.button("Generate brief", type="primary") and raw_input:
    status = st.status("Running pipeline...", expanded=True)
    try:
        entities = affinity.parse_entities(known) if known else None
        result = pipeline.generate_brief(
            raw_input,
            subject_entities=entities,
            subject_hints=known,
            include_org_context=include_org,
            role_spec=role_spec,
            candidate_background=candidate_background,
            meeting_type=meeting_type,
            on_step=lambda m: status.write(m),
        )
        status.update(label="Done", state="complete", expanded=False)

        if result.brief.identity_confidence == "low":
            st.warning(
                "Low identity confidence - sources may describe more than one person "
                "with this name. Verify before relying on this."
            )

        st.markdown(result.markdown)
        st.download_button(
            "Download as markdown",
            data=result.markdown,
            file_name=f"brief-{result.brief.person_name.lower().replace(' ', '-')}.md",
            mime="text/markdown",
        )

        with st.expander("Audit trail (what the gates did)"):
            for line in result.audit:
                st.write(("WARNING " in line and ":warning: " or "- ") + line)
            st.write("**Query plan:**", result.queries)
            st.write("**Sources kept:**", [s["url"] for s in result.sources])
            st.json(result.brief.model_dump(mode="json"))

    except safety.ProtectedContentError as e:
        status.update(label="Blocked by content gate", state="error")
        st.error(str(e))
    except tools.SearchError as e:
        status.update(label="Search failed", state="error")
        st.error(str(e))
    except SynthesisError as e:
        status.update(label="Synthesis failed", state="error")
        st.error(str(e))
