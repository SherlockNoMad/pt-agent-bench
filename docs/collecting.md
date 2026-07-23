# PyTorch SWE-bench Collector — Agent Instructions (v3, parallel, battle-tested)

Produce **validated** SWE-bench task instances from `pytorch/pytorch`, and (optionally) have a
separate agent attempt to solve each. A task only counts if it passes the **Validation
Protocol (§5)**. Written for **many workers running in parallel**.

**Proven at scale:** an overnight run of the self-driving harness (`run_overnight.py`)
produced **102 validated tasks** → `pytorch-task-instances.jsonl`. Every fix below was learned
the hard way in that run; apply them from the start and yield/throughput are far higher.

Turnkey tooling (use these — they encode every gotcha; do not re-derive):
- `discover_deep.py [N]` → date-windowed, resumable discovery → `candidates.jsonl` (python-only, CPU-friendly).
- `collect_one.py <issue> <outdir>` → `<outdir>/instance.json` + `patches/` (or `REJECT:`).
- `validate_one.py <outdir>` → validation protocol on the **already-built** tree; writes F2P/P2P; prints `>>> VALID:`.
- `grade.py <instance.json> <predictions.jsonl>` → Docker-free grader (`resolved`).
- `run_overnight.py <nworkers> <target>` → the whole parallel harness (collect+build+validate loop).

`gh` is at `/tmp/gh` (authenticated).

---

## 0. Scope (v1 corpus) — reject anything outside
- **Python-only fix**: non-test files all `*.py`/`*.pyi`; none under `aten/`, `c10/`,
  `torch/csrc/`, `third_party/`, `caffe2/`, `cmake*`.
- **CPU-runnable test**: avoid `test/{inductor,distributed,cuda,cpp,mobile,onnx}` and MPS.
- **Has a repro test** under `test/**.py`.

---

## 1. PARALLEL EXECUTION MODEL

Each worker owns an **isolated tree + env** and processes claimed candidates concurrently.

### 1.1 Per-worker isolation — USE COPIES, NOT WORKTREES
- **DO NOT use `git worktree`** — PyTorch submodules break in linked worktrees
  (`git submodule update` → `could not get a repository handle for submodule`, build then fails
  on missing `third_party/pybind11`). Instead give each worker an **independent copy** of the
  main clone (which already has submodules populated):
  `rsync -a --exclude '/build/' src/ wt<i>/`  (exclude `build/` so cmake's cached absolute
  paths don't point back at `src`).
- **Conda env per worker via clone** (fast — ~6 s, hardlinked):
  `conda create -y -p env<i> --clone <base_env_with_all_deps>`. Needed because `setup.py
  develop` writes an egg-link into ONE env; shared envs race/collide.
- Workers never touch each other's `wt<i>`/`env<i>` → checkout/build/test/solve fully concurrent.

### 1.2 Claim ledger (dedup across workers)
Atomic claim before working an issue: `mkdir claims/<issue>` (fails if exists). A candidate is
"done" iff it appears in `results.jsonl`. On restart, clear **stale claims** (claimed but no
result) so interrupted work retries.

### 1.3 Concurrency sizing (host: 368 cores, ~2 TB RAM, 25 TB disk)
`MAX_JOBS = min(40, cores/num_workers)`. **Cap it** — uncapped `-j` (e.g. 90×4) causes
transient resource contention on the memory-heavy autograd files. (Note: on this host it was
NOT OOM — 1.9 TB stayed free — but capping still improved build reliability.) 4 workers × j40
is a good default. Plan ~5–10 GB build per copy + ~5 GB per env clone.

### 1.4 Per-worker loop
```
claim(issue) or skip
collect_one.py issue  inst/<instance_id>
STATIC pre-filters (no build!):  §5.0
checkout base_commit; submodule update; build (§4, with clean-retry)
validate_one.py                         # collector-agent verification
if VALID: reset tree blind; SOLVER agent (§6); grade.py (§7)
record → results.jsonl
```

### 1.5 Rebuild cost & locality
Each task needs its `base_commit` built. Incremental rebuild after a checkout is ~3–5 min IF
the previous base was near; a distant jump is a near-full rebuild (~5–10 min). Sort each
worker's queue by commit/date so consecutive tasks are adjacent. Reuse a build across tasks
with **zero compiled-file churn** vs the built commit (python-only delta) — the biggest speedup.

---

## 2. Linking: issue → closing commit → PR
Do NOT use PR `closingIssuesReferences` (~1% hit). Use: closed issue `stateReason==COMPLETED`,
closed **by a commit** whose message has `Pull Request resolved: .../pull/<N>`. The squash
commit *is* patch+test_patch; its **parent is `base_commit`**. `collect_one.py` does this.

## 3. Tuple extraction
`collect_one.py` writes `instance.json` + split `patches/{full,gold_patch,test_patch}.diff`.

---

## 4. Environment recipe (per worker) — VALIDATED, all fixes included

- **Conda python 3.11** (NOT the system `+meta` python — its fbcode loader breaks native libs:
  `GLIBC_2.35 not found`).
- Deps: `cmake==3.31.6 ninja pyyaml typing_extensions numpy setuptools wheel requests
  expecttest hypothesis` + `pip install -r requirements.txt`, and **`pytest==7.4.4`** (see below).
- Build env vars — **note: NO `USE_KINETO=0`**:
  ```
  USE_CUDA=0 USE_DISTRIBUTED=0 USE_MKLDNN=1 USE_FBGEMM=1 BUILD_TEST=1
  MAX_JOBS=40 CMAKE_POLICY_VERSION_MINIMUM=3.5
  ```
- **`pytest==7.4.4` is CRITICAL** — the single biggest yield-killer. PyTorch's
  `test/conftest.py` (older commits) uses the `path` arg in `pytest_pycollect_makemodule`,
  removed in **pytest 8** → every collection fails with `PluginValidationError` →
  "new tests not collected" → the task is wrongly rejected. pytest 7.4.4 works for old and
  recent commits. Overnight, this fix took the count from stuck-at-97 → 102.
- **`BUILD_TEST=1` is MANDATORY** — many test files load a C++ helper `libtorchbind_test.so`
  (`torch.ops.load_library`); without it, valid tasks fail at `setUp`. cmake CACHES this, so a
  bare env-var flip does nothing: force reconfigure by `rm build/CMakeCache.txt` if the cache
  has `BUILD_TEST:BOOL=False` (or `USE_KINETO:BOOL=OFF`).
- **Do NOT set `USE_KINETO=0`** — on some commits `torch/csrc/autograd/profiler_kineto.cpp`
  references `fmt` unguarded and fails to compile with kineto off (`'fmt' has not been
  declared`). Leave kineto at default (on).
- **Clean-rebuild retry**: `setup.py develop` incremental builds across distant commit jumps
  hit stale build-dir/codegen mismatches (`gen.py: unrecognized arguments:
  --headeronly-install-dir`, or `ninja: build stopped` with **no compiler error** = a killed
  codegen step). On ANY build failure, `rm -rf <wt>/build` and rebuild once clean. This turns
  most "build_failed" into successes. (Diagnosing: the harness only saved the last ~3 KB of
  build output — the real error is often above that; do a full-log build to see it.)
- **Runtime**: `export LD_LIBRARY_PATH=/usr/lib64:<env>/lib` (for `libgomp.so.1`) — python/pytest ONLY.
- **CRITICAL git rule**: run every `git` with `env -u LD_LIBRARY_PATH` (host git is fbcode-linked
  and crashes when `/usr/lib64` is on the path). Python WITH the path, git WITHOUT it.

---

## 5. Validation Protocol (`validate_one.py`)

### 5.0 Static pre-filters (BEFORE any build — save wasted builds)
- No `def test_...` anywhere in `test_patch` → reject `no_test_fns_static`.
- **GPU-gated**: added lines contain `requires_cuda` / `requires_gpu` / `requires_triton` /
  `onlyCUDA` / `requires_cuda_and_triton` / `skipCUDAIf` → reject `gpu_gated_static` (won't run
  on CPU; wastes a full clean build otherwise).

### 5.1 A task is invalid unless ALL hold (on the built tree at `base_commit`)
1. Apply `test_patch`. Extract **all touched test functions** — `re.findall(r'def
   (test_\w+)', test_patch)` — this catches **modified** existing tests, not just newly-added
   ones (added-only `^\+.*def test_` misses real bugfixes that edit an existing test). Collect
   them to exact node ids and run. **F2P = those that FAIL pre-fix and PASS post-fix.**
2. Require **F2P non-empty**. (If nothing fails pre-fix → can't reproduce → reject. Most common
   legit reject, often because the bug needs CUDA/py-version not present on CPU/py3.11.)
3. Apply `gold_patch`. Every F2P must PASS.
4. **P2P** = up to 20 other tests in the same file passing in BOTH states; no regressions.
5. **Determinism**: rerun F2P 2× post-fix; any flaky outcome → INVALID.

---

## 6. Solve attempt (separate agent, blind)
Reset tree to `base_commit` clean: `git checkout -f base; git clean -fd test/ torch/ tools/`
(**NEVER a bare `git clean -fd`** — it deletes `build/`). Spawn a solver subagent restricted to
that worker's copy, given ONLY the `problem_statement`, forbidden from reading any other
`ptbench/` dir (answer keys) or editing `test/`. Capture its patch: `git add -N` then
`git diff -- . ':(exclude)test/*'` (`add -N` makes NEW files appear). Write
`{instance_id, model_name_or_path, model_patch}` to a predictions jsonl.
Expect ~67% solve rate on a blind single-shot agent; partial solves on multi-F2P tasks are
normal (e.g. `#190185`: 4/7) and are useful difficulty signal.

## 7. Grading (`grade.py`)
Applies model_patch + gold test_patch to `base_commit`, runs F2P/P2P, prints `resolved`.
**Removes patch-created files before applying** (solvers legitimately add new files), runs git
with a clean lib path. Docker-free; mirrors upstream `run_evaluation.py`.

## 8. Reject checklist
Issue not COMPLETED / not closed by a commit; no `pull/N`; missing test or fix file; fix
touches C++/build; GPU/triton/distributed/MPS test (static-filtered); **pre-fix run doesn't
fail** (can't reproduce — most common); post-fix not all-pass or a regression; flaky; empty
`problem_statement`.

### 8a. Quality signal — down-rank exact-string-pinned tests
Prefer F2P asserting **behavior** (values/shapes/dtypes/exception *type*) over exact message
strings (`collect_one.py` sets `quality.f2p_asserts_message_string`). A behaviorally-correct
fix can score 0 on a message typo — observed on `#187861` (blind agent wrote a correct
`ValueError` with the wrong wording).

## 9. Gotchas (all hit & solved — apply directly)
| Symptom | Fix |
|---|---|
| "new tests not collected" everywhere | **`pip install pytest==7.4.4`** (pytest 8 breaks PyTorch conftest `path` hook) |
| `profiler_kineto.cpp: 'fmt' has not been declared` | do NOT set `USE_KINETO=0` |
| `libtorchbind_test.so: cannot open` | `BUILD_TEST=1` + `rm build/CMakeCache.txt` to reconfigure |
| `git submodule ... could not get a repository handle` | don't use worktrees; `rsync -a --exclude /build/ src/ wt<i>/` |
| `gen.py: unrecognized arguments --headeronly-install-dir` / `ninja build stopped` no error | stale build dir across commit jump → `rm -rf build` and rebuild (clean-retry) |
| `GLIBC_2.35 not found` on import/git | conda python (not `+meta`); git via `env -u LD_LIBRARY_PATH` |
| `libgomp.so.1: cannot open` | `LD_LIBRARY_PATH=/usr/lib64:<env>/lib` for python only |
| cmake rejects old `cmake_minimum_required` | `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| model_patch "patch did not apply" (new file) | grader deletes patch-created files first (grade.py does) |
| `git clean -fd` wipes the build | clean only `test/ torch/ tools/`, never bare |
| `pkill -f run_overnight` kills your own shell | the pattern self-matches your command — exclude `$$`/`$PPID`, or kill in a separate tool call |
| detached builds ignore `kill -9` | orphaned cc1plus/ninja in D-state; they drain naturally — wait, don't fight them |
| parallel workers duplicate work | atomic `mkdir claims/<issue>` before collecting |
| shared conda env egg-link collision | one cloned conda env per worker |

## 10. Discovery reality (yield & pool)
- `discover_deep.py` windows by **created-date** (monthly, going back) to beat GitHub search's
  **1000-result cap** per query; resumable via `seen_issues.txt`.
- **Don't go too far back**: old commits (roughly issue < ~130k) fail to build with the modern
  toolchain (gcc 11.5 / cmake / py3.11) — pure waste. The buildable, high-yield zone is recent
  commits.
- **Yield realities** (measured): hand-picked recent python-only ≈ 70%; a broad auto-discovered
  pool ≈ 15–50% depending on region. Dominant *legit* rejects: can't-reproduce-on-CPU,
  GPU/triton-gated, build-fail on old commits. Budget ~2–4× candidates per target task.

## 11. Output / dataset assembly
Per instance: `inst/<id>/instance.json` + `patches/` + `validation.json`. `results.jsonl` is
the authoritative ledger. Assemble the dataset by concatenating every VALID instance's
canonical fields into `pytorch-task-instances.jsonl` (the harness/leaderboard input).

## 12. Operational notes for a long unattended run
- Launch detached: `nohup python3 run_overnight.py 4 100 > overnight.log 2>&1 & disown`.
- Monitor via `results.jsonl` (`grep -c '"valid": true'`) and per-worker logs; the harness
  auto-stops when valid ≥ target.
- On restart the workers reuse `wt<i>`/`env<i>` (no re-clone/re-copy); build state persists so
  first tasks are incremental. Always clear stale claims first.
- If stuck (valid not climbing): check the **reject-reason histogram** of the last N results —
  it pinpoints the current blocker (e.g. a wave of "not collected" = pytest; "build_failed"
  with no error = stale build dir). Fix the config, clear those results+claims, restart.
