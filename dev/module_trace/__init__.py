"""Module-by-module activation tracing for the af3 graph.

Run one input case through one model, capture every haiku module's output, and
write a fingerprinted trace to disk so two runs (before/after a refactor, our
port vs a reference, model A vs model B) can be diffed step by step.

  python -m tools.module_trace list
  python -m tools.module_trace run --model rosettafold3 --case 6mrr_template
  python -m tools.module_trace compare <trace_a> <trace_b>

See cases.py (inputs), models.py (weights), trace.py (capture), compare.py.
"""
