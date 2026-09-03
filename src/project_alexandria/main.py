# ---- entry point: intentionally empty (no unified CLI yet) ----
# The pipeline is driven through its stage modules, not from here:
#   * build corpus / segment a book : tests.py  (segment_test, step_two_processing)
#   * enrich + index                : tests.py  (embed_test / step_three_embedding)  +  embed.index_scenes
#   * query the index               : search.py (search) — via tests.py, evals.py, or webtest
# Paths + JSON IO live in utils/storage.py + utils/read_write.py. Add a real entry point here if one is needed.
