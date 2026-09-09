# ---- entry point: intentionally empty (no unified CLI yet) ----
# The pipeline is driven through its stage modules, not from here:
#   * build corpus / segment a book : tests.py  (segment_test, step_two_processing)  -> segment.segment_book
#   * enrich -> derive -> index     : tests.py  (embed_test / step_three_embedding)  ->  enrich.enrich_file,
#                                     derive.derive_file, index.index_records  (or index.index_scenes to rebuild)
#   * query the index               : query.run (read front door: normalize a request -> search.search) —
#                                     used by webtest; tests.query_test / evals drive it too. search.search is the raw door.
# Paths + JSON IO live in utils/storage.py + utils/read_write.py. Add a real entry point here if one is needed.
