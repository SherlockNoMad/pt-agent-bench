#!/usr/bin/env python3
"""Aggregate solver-run metrics from solve_results.jsonl.
Usage: python3 metrics.py"""
import json, statistics as st, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

rows=[json.loads(l) for l in open(config.SOLVE_RESULTS) if l.strip()]

def stats(vals):
    vals=[v for v in vals if isinstance(v,(int,float))]
    if not vals: return "n/a"
    return f"mean={st.mean(vals):.1f} median={st.median(vals):.1f} min={min(vals):.1f} max={max(vals):.1f} (n={len(vals)})"

graded=[r for r in rows if "resolved" in r]
resolved=[r for r in graded if r.get("resolved")]
build_failed=[r for r in rows if r.get("reason")=="build_failed"]
instrumented=[r for r in graded if "t_e2e_s" in r]

print(f"=== OUTCOMES ===")
print(f"graded: {len(graded)}   resolved: {len(resolved)}   pass rate: {100*len(resolved)/len(graded):.1f}%" if graded else "no graded rows")
print(f"build_failed: {len(build_failed)}")
print(f"instrumented (with timing/cost): {len(instrumented)}")

if instrumented:
    print(f"\n=== TIMING (seconds) ===")
    print(f"  env build : {stats([r.get('t_build_s') for r in instrumented])}")
    print(f"  agent solve: {stats([r.get('t_solve_s') for r in instrumented])}")
    print(f"  grade     : {stats([r.get('t_grade_s') for r in instrumented])}")
    print(f"  e2e       : {stats([r.get('t_e2e_s') for r in instrumented])}")
    print(f"  claude wall(ms): {stats([r.get('claude_ms') for r in instrumented])}")

    costs=[r.get('cost_usd') for r in instrumented if isinstance(r.get('cost_usd'),(int,float))]
    print(f"\n=== AGENT SESSION COST ===")
    if costs:
        print(f"  per-task cost $: {stats(costs)}")
        print(f"  TOTAL solver cost: ${sum(costs):.2f}  over {len(costs)} sessions")
        rc=[r.get('cost_usd') for r in resolved if isinstance(r.get('cost_usd'),(int,float))]
        if rc: print(f"  cost per RESOLVED task: ${sum(costs)/len(resolved):.3f}")
    print(f"  num_turns: {stats([r.get('num_turns') for r in instrumented])}")

    print(f"\n=== TOKENS (per session) ===")
    print(f"  input      : {stats([r.get('in_tok') for r in instrumented])}")
    print(f"  output     : {stats([r.get('out_tok') for r in instrumented])}")
    print(f"  cache_read : {stats([r.get('cache_read_tok') for r in instrumented])}")
    print(f"  cache_create: {stats([r.get('cache_create_tok') for r in instrumented])}")

# projection to full corpus
if instrumented and costs:
    per=sum(costs)/len(costs); n=102
    print(f"\n=== PROJECTION to {n} tasks ===")
    print(f"  est. total solver cost: ${per*n:.2f}")
    e2e=[r.get('t_e2e_s') for r in instrumented if isinstance(r.get('t_e2e_s'),(int,float))]
    if e2e:
        import math
        w=6
        print(f"  est. wall-clock @ {w} workers: {st.mean(e2e)*n/w/3600:.1f} h")
