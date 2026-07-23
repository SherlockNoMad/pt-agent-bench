# SWE-bench for PyTorch — Design Doc (v1)

**Status:** Draft · **Scope:** v1 = CPU-only, Python-only fixes · **Author:** (design)

Goal: reuse the SWE-bench methodology and codebase to curate an execution-verified
benchmark of real `pytorch/pytorch` issues. Each task is the tuple:

> (environment, base commit, issue text, gold PR patch, tests added in the PR)

A model sees the repo at `base_commit` + the issue text, produces a patch, and is graded
purely by executing tests — `FAIL_TO_PASS` must flip failing→passing and `PASS_TO_PASS`
must stay green.

---

## 1. Scope decisions for v1

To keep v1 tractable and GPU-free, we **filter the candidate set at collection time**:

- **Python-only fixes.** The gold `patch` (the non-test hunks) touches only `*.py` and
  never `aten/`, `c10/`, `torch/csrc/`, `torch/_C`, or build files. This means the model's
  patch never triggers a C++/CUDA recompile — the base image is built once and reused.
- **CPU-runnable tests.** The `test_patch` targets tests that pass on a CPU-only PyTorch
  build (drop tests gated on `@onlyCUDA`, `requires_gpu`, distributed/NCCL, etc.).
- **Recent history.** Restrict to roughly the last ~2 release branches so the number of
  distinct build environments stays small.

Everything below is designed so scope can later expand to C++/CUDA-src fixes and GPU eval
without re-architecting.

Non-goals for v1: GPU evaluation, distributed/multi-process tests, C++/CUDA source fixes,
performance-regression tasks.

---

## 2. Task instance schema

Identical to upstream SWE-bench (see `docs/guides/datasets.md`) so all downstream tooling,
leaderboards, and `sb-cli` keep working:

```python
{
  "instance_id": "pytorch__pytorch-<pr_number>",
  "repo": "pytorch/pytorch",
  "base_commit": "<sha>",              # pull["base"]["sha"]
  "problem_statement": "<issue title+body>",
  "hints_text": "<pre-fix PR/issue comments>",
  "patch": "<gold code diff, *.py only>",   # hidden from model
  "test_patch": "<gold test diff>",          # applied at grading time
  "FAIL_TO_PASS": [ ... test ids ... ],
  "PASS_TO_PASS": [ ... test ids ... ],
  "version": "<release-era key, e.g. '2.4'>",
  "environment_setup_commit": "<sha>",
  "created_at": "...", "issue_numbers": [...]
}
```

---

## 3. Reuse map: what stays, what we add

The pipeline has three stages: **collect → environment/build → grade**. The collect stage
is essentially repo-agnostic and reused almost verbatim; the other two need PyTorch adapters.

| Stage | File(s) | v1 change |
|---|---|---|
| Scrape merged PRs | `swebench/collect/print_pulls.py` | none |
| Build candidates | `swebench/collect/build_dataset.py` (`create_instance` L331) | none to core; add filters (§4) |
| Issue↔PR linking | `swebench/collect/utils.py` `extract_resolved_issues` (L586) | **add PyTorch branch** (§4.2) |
| Patch/test split | `swebench/collect/utils.py` `extract_patches` (L820) | **tighten test-path rule** (§4.3) |
| Problem statement | `swebench/collect/utils.py` `extract_problem_statement_and_hints` (L743) | reuse (django-style hook available at "MARK: Repo Specific Parsing", L844) |
| Version resolution | `swebench/versioning/` | **add PyTorch resolver** (§5.1) |
| Build/test env spec | `swebench/harness/constants/python.py` (`SPECS_*`) | **add `SPECS_PYTORCH`** (§5.2) |
| Log → per-test status | log parsers (cf. `tests/test_log_parsers_java.py`) | **add PyTorch parser** (§6.2) |
| Run + grade | `swebench/harness/run_evaluation.py`, `grading.py` (resolved logic L290) | none |

---

## 4. Collection stage

Command (works today for scraping):

```bash
export GITHUB_TOKENS=<tok>
python -m swebench.collect.get_tasks_pipeline \
  --repos pytorch/pytorch --path_prs ./prs --path_tasks ./tasks \
  --cutoff_date 20230101
```

`is_valid_pull` (`build_dataset.py:361`) already keeps only **merged PRs with ≥1 linked
issue**; `has_test_patch` keeps only those whose diff adds tests. We add three filters.

### 4.1 Python-only filter (new)
After `extract_patches`, reject the instance unless every non-test hunk path ends in `.py`
and none start with `aten/`, `c10/`, `torch/csrc/`, `torch/_C`, `cmake`, `setup.py`, or
`third_party/`. Implemented as an extra predicate alongside `is_valid_instance`
(`build_dataset.py:377`).

### 4.2 Issue↔PR linking for PyTorch (new)
PyTorch merges via **ghstack/Phabricator**, so PR bodies frequently use
`Pull Request resolved: <url>` / `ghstack-source-id:` rather than `Fixes #NNNN`. The stock
regex `(\w+)\s+#(\d+)` + `PR_KEYWORDS` (`utils.py:596`) under-catches. Add a PyTorch branch
that additionally:
- reads GitHub's **"closing issues" GraphQL edges** for the PR (authoritative link), and
- scans the **squashed merge commit message** for `Fixes/Closes/Resolves #NNNN`.
Union the results with the existing regex output.

### 4.3 Test-path detection (tighten)
`extract_patches` currently routes any hunk whose path merely *contains* `test` to the test
patch (`utils.py:834`). In PyTorch this wrongly captures `torch/testing/` (shipping library
code). Change the rule to **`path.startswith("test/")`** (PyTorch's test root) plus explicit
allowlist for `test/**`. This keeps `torch/testing/` in the fix patch where it belongs.

### 4.4 Quality gates (collection-time)
Drop instances where: `problem_statement` is empty/very short, the issue is a
feature-request (no failing behavior), or the linked issue is actually a tracking/meta
issue. These are heuristics; the authoritative filter is validation (§7).

---

## 5. Environment / build (CPU-only)

Key idea: the expensive PyTorch source build happens **once per release-era base image**,
and — because v1 fixes are Python-only — the model's patch is applied *on top* of an
already-built tree without recompiling C++/CUDA.

### 5.1 Version resolution (`versioning/`)
Map each `base_commit` to a coarse **release-era key** (e.g. `"2.3"`, `"2.4"`) rather than a
pip version. Resolve by finding the nearest release tag/branch ancestor of `base_commit`
(`git describe --tags`), or by date buckets. This key selects the `SPECS_PYTORCH` entry.

### 5.2 `SPECS_PYTORCH` (`harness/constants/python.py`)
Mirror `SPECS_SKLEARN`'s shape, one entry per era:

```python
SPECS_PYTORCH = {
  "2.4": {
    "python": "3.11",
    "base_image": "pytorch-cpu-build:2.4",      # prebuilt: toolchain + deps, CPU only
    "packages": "requirements.txt",
    # editable dev install; sccache mounted so any (future) rebuild is incremental
    "install": "USE_CUDA=0 USE_DISTRIBUTED=0 python setup.py develop",
    "pip_packages": ["numpy", "expecttest", "hypothesis", "pytest"],
    "test_cmd": "python test/run_test.py --continue-on-error -i",
  },
  # ... "2.3": {...}
}
```

Because installs are per-era, most instances share an image (`docker_build.py`,
`prepare_images.py` already layer env → repo → instance).

### 5.3 CPU-only build knobs
`USE_CUDA=0 USE_DISTRIBUTED=0 USE_MKLDNN=1 BUILD_TEST=0`, `MAX_JOBS` capped, `sccache`
mounted. This is the single most impactful cost lever and the reason v1 is GPU-free.

---

## 6. Grading

### 6.1 Selective test execution
Never run the full suite. From `test_patch`, extract the touched test files and run only
those (`run_test.py -i <file>` / pytest node ids). `FAIL_TO_PASS`/`PASS_TO_PASS` are drawn
from these files only.

### 6.2 PyTorch log parser (new)
Add a parser (unit-tested like `tests/test_log_parsers_java.py`) that converts PyTorch test
output into `{test_id: PASSED|FAILED|SKIPPED}`. Handle both `run_test.py` and raw pytest
output; normalize node ids to stable `test_module::TestClass::test_name` form. Register it in
the repo→parser map so `grading.py` can compute `FAIL_TO_PASS`/`PASS_TO_PASS`
(`grading.py:198+`, resolved logic at L290).

### 6.3 Resolved definition
Unchanged: instance is resolved iff **all** `FAIL_TO_PASS` pass **and** all `PASS_TO_PASS`
still pass after applying (model patch + gold `test_patch`).

---

## 7. Validation & flakiness (critical for PyTorch)

Before an instance enters the benchmark it must pass an automated validation run:

1. Apply gold `patch` + `test_patch` at `base_commit`; confirm every `FAIL_TO_PASS`
   test **fails without** the fix and **passes with** it.
2. Confirm `PASS_TO_PASS` is green with the fix.
3. **Flakiness screen:** repeat the affected tests N times (e.g. 3–5) with fixed seeds
   (`PYTORCH_TEST_WITH_*` off, `--seed`); discard any instance with non-deterministic
   outcomes. PyTorch's CUDA/threading/seed non-determinism makes this mandatory even for
   CPU tests.

Only instances that reproduce deterministically are kept.

---

## 8. Filter summary (candidate → benchmark)

A PR becomes a task iff **all** hold:
- merged, and closes ≥1 issue (stock + PyTorch linking §4.2);
- adds/edits tests under `test/` (§4.3);
- gold fix patch is Python-only (§4.1);
- non-empty, bug-describing `problem_statement` (§4.4);
- affected tests are CPU-runnable;
- passes deterministic validation (§7).

---

## 9. Deliverables (file-level)

New/changed within this repo, following existing conventions:
1. `swebench/collect/utils.py` — PyTorch `extract_resolved_issues` branch; tighten
   `extract_patches` test-path rule.
2. `swebench/collect/build_dataset.py` — Python-only + CPU-runnable filters.
3. `swebench/versioning/` — PyTorch era resolver.
4. `swebench/harness/constants/python.py` — `SPECS_PYTORCH` + repo→spec/parser registration.
5. PyTorch log parser + `tests/test_log_parsers_pytorch.py`.
6. Dockerfiles for `pytorch-cpu-build:<era>` base images (sccache, toolchain, deps).
7. A validation runner config to produce the deterministic, filtered final set.

---

## 10. Milestones

1. **M1 — Collect:** scrape + link + Python-only filter → candidate JSONL; report volume &
   link-hit rate.
2. **M2 — Environment:** one `pytorch-cpu-build` base image + `SPECS_PYTORCH` for a single
   era; `setup.py develop` succeeds.
3. **M3 — Grade:** PyTorch log parser + selective test run on one gold instance end-to-end.
4. **M4 — Validate:** run validation+flakiness over the candidate set → v1 dataset.
5. **M5 — Baselines:** run a couple of agents/models to sanity-check difficulty spread.

---

## 11. Risks & open questions

- **Link recall:** how many PyTorch bug-fix PRs actually close a discoverable issue? M1
  measures this; if low, we may relax to "PR title/body as problem statement" (a deviation
  from strict SWE-bench).
- **Python-only volume:** is the Python-only subset large enough for a useful benchmark? If
  not, C++/CUDA-src fixes (still CPU-built) are the first expansion.
- **Test isolation:** some PyTorch tests mutate global state / require `run_test.py`
  sharding semantics; the parser and selective-run logic must handle this.
- **Base-image drift:** third-party deps for old eras may be hard to pin; date-bucketing may
  need manual fixups per era.
