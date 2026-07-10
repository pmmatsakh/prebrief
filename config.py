"""Central configuration. Every external dependency is swappable here."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (OpenAI-compatible endpoint) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")

# --- Safety screening model (small, cheap). Falls back to deterministic if unavailable. ---
SAFETY_MODEL = os.getenv("SAFETY_MODEL", "openai/gpt-oss-20b")
SAFETY_SCREENING_ENABLED = os.getenv("SAFETY_SCREENING_ENABLED", "1") == "1"
SAFETY_MAX_SOURCES = int(os.getenv("SAFETY_MAX_SOURCES", "20"))  # one batched call, not one per source

# --- Search ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "6"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "30"))
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "86400"))  # 24h

# --- Pipeline knobs ---
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))  # trimmed adaptively vs 8k TPM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# Reasoning models (GLM-5.2, DeepSeek) "think" before answering. For summarize-and-
# structure work that thinking is mostly wasted time. "low" cuts latency sharply.
# Ignored by providers/models that do not support it.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# --- Optional-input caps (the free tier is 8k TPM; these keep the prompt in budget) ---
MAX_ROLE_CHARS = int(os.getenv("MAX_ROLE_CHARS", "2500"))
MAX_BACKGROUND_CHARS = int(os.getenv("MAX_BACKGROUND_CHARS", "2500"))

# --- Org context (shared entity, cached per company) ---
ORG_CONTEXT_CHARS = int(os.getenv("ORG_CONTEXT_CHARS", "8000"))
ORG_CACHE_TTL = int(os.getenv("ORG_CACHE_TTL", "604800"))   # 7 days: company facts move slowly

# --- Identity confidence: below this, refuse rather than guess ---
IDENTITY_MIN_SOURCES = int(os.getenv("IDENTITY_MIN_SOURCES", "2"))

# --- Data files ---
PROFILE_PATH = os.getenv("PROFILE_PATH", "profile.yaml")
NETWORK_PATH = os.getenv("NETWORK_PATH", "network.yaml")

# --- App ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")


def validate() -> list[str]:
    problems = []
    if not LLM_API_KEY:
        problems.append("Missing GROQ_API_KEY (or LLM_API_KEY) - get one at console.groq.com")
    if not TAVILY_API_KEY:
        problems.append("Missing TAVILY_API_KEY - get one at tavily.com")
    return problems
