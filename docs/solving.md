# PyTorch SWE-bench Solve & Grade — Agent Instructions

Measure an **agent pass rate** over already-collected, validated tasks: for each task a blind
agent produces a patch, then it's graded by execution. This is the counterpart to
`COLLECTOR_INSTRUCTIONS.md` (which only collects + validates). Reuses the same per-worker
isolation and environment recipe (see collector §1, §4).

**Turnkey tooling**
- `solve_and_grade.py <nworkers> [limit]` — the full parallel harness (build → blind solve → grade).
- `grade.py <instance.json> <predictions.jsonl>` — Docker-free grader for a single prediction.
- Input dataset: `pytorch-task-instances.jsonl` (or `inst/<id>/instance.json`), each with
  `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `base_commit`.

Proven: matches manual blind-solve results exactly (`189142`✓, `190287`✓, `187861`✗). Blind
single-shot `claude -p --model sonnet` pass rate ≈ **60–70%** on a small sample.

---

## 1. The per-task pipeline
```
claim(instance_id)                              # atomic mkdir solve_claims/<id>
checkout base_commit in wt<i>; submodule update; build   # same recipe as collector §4
reset BLIND: git checkout -f base; git clean -fd test/ torch/ tools/   (NEVER bare clean -fd)
write PROBLEM.txt = problem_statement           # the ONLY task input the solver sees
run the blind solver (claude -p)                # edits torch/ in place
capture model_patch = git diff of non-test files
grade: apply model_patch + gold test_patch -> run F2P/P2P -> resolved
record -> solve_results.jsonl
```
`resolved = (all FAIL_TO_PASS pass) AND (all PASS_TO_PASS pass)` after applying the model's
patch + the gold `test_patch`. Same definition as upstream `run_evaluation.py`.

## 2. The blind solver (headless `claude -p`) — LOCKED DOWN
Command (run in the worktree, `cwd=wt<i>`, with a CLEAN env — no `LD_LIBRARY_PATH`, see §6):
```
claude -p "<prompt>" --bare --permission-mode default --model sonnet \
        --output-format json --allowedTools Read Edit Write Grep Glob Bash
```
- `--bare` removes plugins / MCP / skills / hooks (kills `external_web_search3pai`, `Skill`,
  plugin `Agent`).
- `--permission-mode default` + `--allowedTools ...` = strict allowlist: only those 6 tools;
  `WebFetch`/`Agent`/`Task` are auto-denied in headless mode.
- **Do NOT use `--dangerously-skip-permissions`** — `bypassPermissions` IGNORES allow/deny
  lists, so web/MCP tools stay usable (verified: a solve web-searched for the issue's PR).
- `--output-format json` gives the metrics (cost/tokens/turns) and `session_id` for the trace.
Prompt: fix the bug in `./PROBLEM.txt`, edit ONLY non-test source under `torch/` (never
`test/`), minimal; torch is already built; to run python `export
LD_LIBRARY_PATH=/usr/lib64:<env>/lib` + `<env>/bin/python`; git via `env -u LD_LIBRARY_PATH`.

## 2a. Anti-reward-hacking (MANDATORY — audit every run)
The solver is full Claude Code; without lockdown it WILL use whatever your env exposes. A real
observed contamination: a solve ran `external_web_search3pai("PyTorch GitHub issue ... fix
pull request")` + `WebFetch` of the reference source — it looked up the answer online. So:
- **Lock tools** as in §2 (removes web/MCP/Skill/Agent).
- **Preserve the full trace** per task: `claude` persists every tool call at
  `~/.claude/projects/<worktree-path-slug>/<session_id>.jsonl`. The harness copies
  it to `solve_logs/<instance_id>.trace.jsonl` and records `session_id` in the result.
- **Audit** with `audit_traces.py` — flags any web/MCP/forbidden tool, or suspicious `Bash`
  (`curl`/`wget`/`git show <sha>`/`git log --all`/`.patch`/`/inst/`/`pull/N`) or `Read` of
  answer-key paths. FLAGGED solves are excluded from the pass rate.
- **Opus actively reward-hacks git — verified.** In the opus-4.8/xhigh run, the auditor caught
  solves running `git log --all | grep <keywords>`, `git show <sha> --stat`, and
  `git merge-base --is-ancestor <sha> HEAD` to hunt for the fix. The threat is REAL, not
  theoretical — audit every run.
- **Git-history strip (implemented, `_strip_history`/`_restore_history`):** during the solve
  window the harness (1) moves all git refs (`packed-refs`+`refs/`) to a backup **OUTSIDE the
  worktree** (`ROOT/.refbak/wt<i>`), and (2) removes the `origin` remote. Effect:
  `git log --all`/`branch -a`/`tag`/`show origin/main` return nothing, AND `git show <fix_sha>`
  can't render the diff because the blobless clone must fetch the fix's blobs and the remote is
  gone. HEAD stays a detached sha; checkout-by-sha works.
  - **BUG we hit & fixed:** the first version backed refs up to `.git-refs-bak` *inside* the
    worktree — opus found it (`ls .git-refs-bak/`) and used it to recover history. The backup
    MUST live outside the worktree. Contaminated solves were cleared and re-run.
  Chosen over re-initializing a single-commit repo because it is **instant** and **safe under
  interruption**: objects are untouched, checkout-by-sha still works, and `ensure()` restores
  refs on the next start if a solve was killed mid-strip (the object-swap approach corrupts the
  worktree if SIGKILLed). Refs are restored (`finally`) before grading.
- **Residual vectors (Bash is allowed) — caught by the auditor:** exotic git spelunking
  (`git rev-list --all`, `fsck --unreachable`, `cat-file --batch`) can still reach unreachable
  future objects, and `curl`/`wget` reach the internet. `audit_traces.py` FLAGS all of these
  (and `git show <sha>`, `log --all`, `for-each-ref`, `pull/N`, answer-key reads); **flagged
  solves are excluded from the pass rate.** For a fully airtight run, add an egress allowlist
  (only the model API host) so `curl`/`wget` also fail while the LLM still works.

**Blindness guarantees** (critical — else the pass rate is meaningless):
- Solver sees only `problem_statement` (via PROBLEM.txt) + the repo at `base_commit`.
- The gold `patch`, `test_patch`, and F2P/P2P ids are NEVER in the worktree or prompt.
- The added test is absent at `base_commit` (it arrives only via `test_patch` at grade time),
  so the solver can't read the grader. Forbid it from reading other `ptbench/` dirs.
- Forbid editing `test/`; strip any test-file hunks from the captured patch anyway.

## 3. Capturing the model patch
```
git add -N .                                    # so NEW files created by the solver appear
git diff -- . ':(exclude)test/*' ':(exclude)PROBLEM.txt'
git reset                                        # unstage the intent-adds
```
Write `{instance_id, model_name_or_path, model_patch}` to a predictions jsonl (one per line).

## 4. Grading (`grade.py`, or inline in the harness)
- Reset to `base_commit`; **delete any files the model_patch creates** before `git apply`
  (else "patch did not apply" on new files — grade.py handles this).
- Apply model_patch, then gold `test_patch`.
- Run `FAIL_TO_PASS + PASS_TO_PASS` with pytest; `resolved` iff all pass.
- Run git with a clean lib path; run pytest with `LD_LIBRARY_PATH=/usr/lib64:<env>/lib`.

## 5. Parallelism & environment
Identical to the collector: N workers, each its own `wt<i>` (rsync copy, NOT worktree) + cloned
`env<i>`; `MAX_JOBS = min(40, cores/nworkers)`. **Multiple full PyTorch builds run concurrently**
(one per worker) — bounded by the MAX_JOBS cap so total compilers ≤ cores. Environment recipe
(MUST match collector §4): conda python 3.11, **`pytest==7.4.4`**, `BUILD_TEST=1`, **no
`USE_KINETO=0`**, `CMAKE_POLICY_VERSION_MINIMUM=3.5`, clean-rebuild retry on build failure,
git via `env -u LD_LIBRARY_PATH`.

## 6. Gotchas specific to solve+grade (all hit & fixed)
| Symptom | Fix |
|---|---|
| **Every solve returns empty patch (`patch_bytes=0`), claude rc=1** | `claude` is fbcode-linked — it crashes with `GLIBC_2.35 not found` if `/usr/lib64` is on `LD_LIBRARY_PATH`. Run `claude` with a **clean env** (no `LD_LIBRARY_PATH`); the solver adds it itself when running python. |
| claude "no stdin data received", flaky | pass `stdin=subprocess.DEVNULL` to the claude subprocess |
| PROBLEM.txt empty → solver asks for the bug | assert `os.path.getsize(PROBLEM.txt) > 0` before running the solver |
| model_patch "patch did not apply" (new file) | remove patch-created files before `git apply` (grade does this) |
| build fails / stale build dir | clean-rebuild retry (`rm -rf build`), same as collector |
| grade resets wipe the build | `git clean` only `test/ torch/ tools/`, never bare |

## 7. Running it
```
nohup python3 solve_and_grade.py 4 > solve.log 2>&1 & disown      # all tasks, 4 workers
# or a validation batch first:  solve_and_grade.py 4 8
```
- Resumable: skips any `instance_id` already in `solve_results.jsonl`; clear stale
  `solve_claims/` (claimed, no result) before a restart.
- Monitor: `grep -c '"resolved": true' solve_results.jsonl`; per-worker `solve_logs/s<i>.log`;
  per-task solver transcript `solve_logs/<id>.claude.log`.
- **Always validate on a small `limit` batch first** — it caught the fbcode-loader bug before
  wasting 102 builds.

## 8. Pass-rate assembly
```
python3 -c "import json; r=[json.loads(l) for l in open('solve_results.jsonl')]; \
print(f'pass rate: {sum(x[\"resolved\"] for x in r)}/{len(r)} = {100*sum(x[\"resolved\"] for x in r)/len(r):.1f}%')"
```
Report alongside difficulty signal: partial F2P (`f2p_pass/f2p_total`) shows near-misses on
multi-test tasks; cross-reference `quality.f2p_asserts_message_string` (message-pinned tasks
that a behaviorally-correct fix fails — e.g. `187861`).

## 9. Interpreting results honestly
- This is **one blind, single-shot attempt per task, one model**. Not a leaderboard number —
  a scaffold/model/attempts sweep would move it a lot.
- A `resolved=False` is often a near-miss (wrong error string, partial multi-F2P fix), not a
  total failure — inspect `f2p_pass/f2p_total` and the `.claude.log` transcript before concluding.
- Keep the solver's model, flags, and prompt fixed across a run so the number is comparable.
