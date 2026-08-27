from __future__ import annotations
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load .env ONCE, here, at import time. Every module that needs configuration imports
# storage (for paths / IO), so importing it populates os.environ for all of them — no
# other module calls load_dotenv() itself.
load_dotenv()


# --- LLM client + shared config (model / error policy / schema version) --- #
# The single OpenRouter client + model id + error policy, shared by both LLM stages
# (segmentation in process.py, enrichment + query distillation in embed.py). It lives
# here because storage already loads .env, so os.environ["OPENROUTER_KEY"] is ready.
# MODEL and the retry policy stay the user's tuning surface; SCHEMA_VERSION stamps the
# record shape (embed.py reads it).

SCHEMA_VERSION = 3   # bump when the scene-record shape changes (embed.py reads it).
                     # v2: + decomposed frame fields (subject/verb/object/setting).

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)


def classify_llm_error(e: Exception) -> str:
    """Classify an API error: "transient" (retry with backoff) vs "fatal" (raise now).

    transient = network / 429 / 5xx; fatal = other 4xx where retrying won't help.
    """
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "transient"
    if isinstance(e, openai.APIStatusError):
        return "transient" if (e.status_code == 429 or e.status_code >= 500) else "fatal"
    return "transient"  # unknown network-ish -> limited retry