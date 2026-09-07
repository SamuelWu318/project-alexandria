# ---- utils package: one import surface for the shared foundation ----
# Re-exports the flat helpers (paths, IO, checkpoint, tag enums, LLM client/config) so callers
# write `from utils import ...`. Heavier submodules (schema, relational, subjects, log) are
# imported by their module name, e.g. `from utils import schema`. SCHEMA_VERSION is intentionally
# NOT re-exported here — it lives solely in schema.py (the single source), read as `schema.SCHEMA_VERSION`.
from utils.checkpoint import Checkpoint
from utils.llm import MODEL, MODEL_PARAMS, CLIENT, classify_llm_error, llm_ready_up, WORKERS, PROCESS_PROMPT, EMBED_PROMPT, inject_retry_notes
from utils.storage import SrcPaths
from utils.tags import Tone, Arc, Intensity
from utils.read_write import read_text, write_text, read_json, write_json

__all__ = ["Checkpoint", "MODEL", "MODEL_PARAMS", "CLIENT", "llm_ready_up",
           "classify_llm_error", "SrcPaths", "read_json", "write_json",
           "read_text", "write_text", "Tone", "Arc", "Intensity",
           "WORKERS", "PROCESS_PROMPT", "EMBED_PROMPT", "inject_retry_notes"]
