<h1 align="center">pt-agent-bench</h1>

<p align="center">
  <b>An execution-verified, agentic SWE benchmark built from real <a href="https://github.com/pytorch/pytorch">pytorch/pytorch</a> issues.</b>
</p>

pt-agent-bench extends the [SWE-bench](https://github.com/SWE-bench/SWE-bench) methodology to
PyTorch. Each task is a real GitHub issue plus the merged PR that fixed it: the model is given
the repository at the pre-fix commit and the issue text, and must produce a patch. It is graded
purely by **execution** — the PR's tests must go from failing to passing (`FAIL_TO_PASS`) while
existing tests stay green (`PASS_TO_PASS`).

Unlike the original Python-only SWE-bench repos, PyTorch requires a from-source C++/CUDA build,
so the harness handles per-task source builds and ships a hardened, **audited** solve loop
designed to resist reward hacking.

## Results

Blind, single-shot solver (`claude-opus-4-8`, effort `xhigh`) over the v1 corpus:

| Metric | Value |
|---|---|
| **Pass rate** | **61 / 99 = 61.6%** |
| Cost | $139 total · $1.51/task · $2.28/resolved |
| Time/task | build 8.8 m · solve 7.6 m · e2e 17.1 m |
| Run cost/throughput | $0.53/min · ~23 tasks/hr @ 6 workers |

Full breakdown + validity discussion: [`results/solve_report.md`](results/solve_report.md).

## Layout

```
pt-agent-bench/
├── problems/        # THE BENCHMARK: pt-agent-bench.jsonl (99 tasks) + instances/<id>/ (patches)
├── collect/         # build the benchmark from GitHub: discover → collect_one → run_collection
├── solver/          # solve.py — runs a blind agent per task (build → solve → capture patch)
├── grader/          # grade.py (execution grading) · validate.py · audit_traces.py
├── results/         # solve_results.jsonl · traces/ (solver transcripts) · metrics.py · report
├── docs/            # design.md · collecting.md · solving.md
├── config.py        # single source of paths (workspace overridable via PTAB_WORKSPACE)
└── setup_workspace.sh   # build the (gitignored) workspace: pytorch clone + conda env
```

## Task schema (`problems/pt-agent-bench.jsonl`, one per line)

```json
{"instance_id", "repo", "base_commit", "problem_statement",
 "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "version",
 // metadata (SWE-bench-compatible extras; harness ignores unknown keys)
 "issue_labels", "issue_created_at", "fix_commit_at", "resolution_days",
 "hints_text", "issue_url", "pr_url", "issue_numbers", "pull_number",
 "fix_files", "test_files", "f2p_count", "p2p_count", "patch_size_loc"}
```
The model sees only `repo`@`base_commit` + `problem_statement`. `patch`/`test_patch` are the
withheld gold solution + grading tests. `hints_text` is the issue conversation **before the
fix** (leakage-safe context; not fed to the solver by default). `issue_labels` enable category
analysis; `issue_created_at`/`fix_commit_at`/`resolution_days` enable time-trend analysis.
Regenerate metadata with `collect/enrich_instances.py`.

## Quickstart

```bash
# 1. Build the workspace (pytorch clone + conda env; ~one CPU-only build). Needs conda + gh.
bash setup_workspace.sh

# 2. Solve all tasks with a blind agent, then grade (6 parallel workers). Requires `claude` CLI.
python3 solver/solve.py 6

# 3. Inspect
python3 results/metrics.py            # pass rate, cost, time, cost/min
python3 grader/audit_traces.py        # reward-hacking audit of saved solver traces

# Grade a single prediction file:  python3 grader/grade.py problems/instances/<id>/instance.json preds.jsonl
```
Prediction format is SWE-bench-compatible: `{"instance_id", "model_name_or_path", "model_patch"}`.

## Anti-reward-hacking (why the numbers are trustworthy)

The solver is a real coding agent, so we sandbox the *task*, not the agent's abilities. Every
solve runs under two universal controls, and **every trace is audited**:

1. **Tool lockdown** — `--bare` + tool allowlist: no web search, MCP, sub-agents, or skills.
2. **Airtight git-history strip** — the real `.git` is moved outside the worktree and replaced
   by a single-commit repo of the base tree, so the future fix commit's objects are absent
   (`git show <fix>` → *bad object*). Interrupt-safe.
3. **`grader/audit_traces.py`** flags any solve that reached for the web, future git history,
   or answer-key files; flagged solves are excluded.

opus-4.8 was observed *actively* attempting both web lookup and `git log --all`/`git show` of
the fix — both were caught and blocked. See [`docs/solving.md`](docs/solving.md).

## Scope (v1)

Recent, **CPU-runnable, Python-only** fixes. Expanding to C++/CUDA-source and GPU tasks is on
the roadmap ([`TODO.md`](TODO.md)).

## Building your own tasks

`collect/` reproduces the benchmark from GitHub (issue → closing commit → PR → tuple →
validate). See [`docs/collecting.md`](docs/collecting.md). Key trick: PyTorch merges via
ghstack, so link issues via the **closing commit** (`Pull Request resolved: .../pull/N`), not
the PR's `closingIssuesReferences`.

## License

MIT — see [LICENSE](LICENSE).
