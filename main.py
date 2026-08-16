# FOR CLAUDE — Intentionally empty placeholder; there is no unified CLI yet.
# -----------------------------------------------------------------------------
# The pipeline is driven directly through its stage modules:
#   * build the corpus / segment a book : tests.py  (segment_test, payload_dump_test)
#   * enrich + index                    : `python embed.py [code ...]`
#   * query the index                   : search.py  (imported by tests.py / the app)
# Paths + JSON IO live in storage.py. Add a real entry point here if one is needed.
# -----------------------------------------------------------------------------
