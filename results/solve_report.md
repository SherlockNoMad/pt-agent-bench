# PyTorch SWE-bench — Solve & Grade Report

**Benchmark:** 99 unique PyTorch tasks (dataset `pytorch-task-instances.jsonl` had 102 lines, 99 unique ids).
**Solver:** `claude-opus-4-8`, effort `xhigh`, **blind** (sees only `problem_statement` + repo@`base_commit`).
**Grading:** execution-based — apply model patch + gold `test_patch`, run `FAIL_TO_PASS`/`PASS_TO_PASS`; `resolved` = all pass.

## Headline
| Metric | Value |
|---|---|
| **Pass rate** | **61/99 = 61.6%** |
| Total solver cost | **$139.15** |
| Cost per task (mean / median) | $1.51 / $0.95 |
| Cost per resolved task | $2.28 |
| Per-task time (mean) | build 8.8 m · solve 9.2 m · grade 0.6 m · e2e 18.7 m |
| Wall-clock @ 6 workers | ~5.2 h (309 min); 1,852 min single-worker-equiv |
| **Cost/min (run @ 6 workers)** | **$0.45/min** |
| Throughput | ~19 tasks/hour @ 6 workers |

## Validity — anti-reward-hacking (audited)
Every solve ran under two universal controls:
1. **Tool lockdown** (`--bare --permission-mode default --allowedTools Read Edit Write Grep Glob Bash`): no web / MCP / Skill / Agent. (Bypass mode was NOT used — it ignores allow/deny.)
2. **Airtight git-history strip**: during solve the real `.git` is moved OUTSIDE the worktree and replaced by a single-commit repo of the base tree, so the future fix commit's objects are absent (`git show <fix>` → bad object, `git log --all` → one commit). Interrupt-safe (real `.git` always restorable).

**Trace audit** (`audit_traces.py` over saved `solve_logs/<id>.trace.jsonl`):
- 90 clean, 2 flagged, 7 without a saved trace (early runs / copy misses).
- The 2 flagged (`184053`, `184562`) *attempted* git archaeology (`git log --all`, `git show`) but were **blocked** — both returned only the single `base` commit. Not contaminated.
- Contamination found & remediated mid-run: opus **did** actively try to read the fix from git history; earlier weaker strips (in-worktree ref backup; ref/remote removal) leaked because the fix objects were cached locally. Those contaminated solves were cleared and re-run under the airtight strip. Also caught & fixed earlier: a web-search leak (`external_web_search3pai`) before the `--bare` lockdown.

## Caveats
- One blind, single-shot attempt per task, one model — a capability snapshot, not a leaderboard number (scaffold/attempts would move it).
- 7 solves lack a saved trace so weren't line-audited, but the lockdown + strip are universal (applied regardless of tracing), so they could not hack.
- These 99 are the v1 corpus: recent, CPU-runnable, Python-only fixes.

## Artifacts
- `solve_results.jsonl` — per-task: resolved, f2p_pass/total, **cost_usd, t_build_s/t_solve_s/t_grade_s/t_e2e_s, num_turns, tokens, session_id**.
- `solve_logs/<id>.trace.jsonl` — full solver tool-call transcript (for audit).
- `metrics.py` (aggregate), `audit_traces.py` (audit), `solve_and_grade.py` (harness).
