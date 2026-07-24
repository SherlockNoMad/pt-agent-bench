# pt-agent-bench — Project TODO

Living checklist for the pt-agent-bench repo.
Docs: `docs/design.md`, `docs/collecting.md`, `docs/solving.md`, `results/solve_report.md`.

## ✅ Done
- [x] Validated methodology on 1 task end-to-end (`pytorch__pytorch-187861`).
- [x] Parallel collection harness → **99 validated tasks** (`problems/pt-agent-bench.jsonl`).
- [x] Docker-free grader + validation protocol; solve harness instrumented (cost/time/tokens/session_id).
- [x] Anti-reward-hacking: `--bare` lockdown + airtight git-history strip + `grader/audit_traces.py`.
- [x] Full blind run: **opus-4.8/xhigh = 61/99 (61.6%)**, $139, $0.53/min @ 6 workers.
- [x] **Public release prep**: renamed `pt-agent-bench`, logical folders, `config.py`, README/LICENSE/.gitignore, `setup_workspace.sh`, git init.
- [x] **Dedup** dataset 102→99 unique.

## 🔧 Open — hygiene / release
- [ ] Push to a GitHub repo (remote + `git push`); consider publishing dataset to HuggingFace too.
- [ ] Retire the old scratch/backup workspace once pt-agent-bench is verified; reclaim ~66 GB (worktrees, envs, venv).
- [ ] Re-run the 2 flagged-but-blocked tasks (`184053`, `184562`) for a spotless audit.
- [ ] 7 solves lack a saved trace — re-run or make trace-copy robust.
- [ ] Rebuild + smoke-test the fresh `workspace/` (via `setup_workspace.sh`) end-to-end on 1 task.

## 🧪 Open — benchmark rigor
- [ ] **Persist exact agent patch text** (⏳ *after the in-flight solve run finishes* — don't edit
      `solve.py` or trigger a re-run mid-flight). `solve.py` computes `mp` (`git diff`) but stores only
      `patch_bytes=len(mp)`; the text is discarded and worktrees are `git clean`ed per task, so it's
      unrecoverable. Write `mp` to `results/multi/patches/<iid>__<backend>.patch` (mirror `traces/`),
      then have the explorer "vs gold" tab load it → byte-exact instead of trace-reconstructed.
      NOTE: only helps *future* runs; re-running re-invokes agents → different patches.
- [ ] Explorer "vs gold" compare tab currently *reconstructs* the agent patch from traces (some
      backends persist full edit content; others only file+line counts, no hunk text). Once patches
      are persisted, switch the agent column to the stored patch and drop the "not byte-exact" caveat.
- [ ] Multiple attempts / pass@k + a second model (e.g. sonnet) for a comparable number, not a snapshot.
- [ ] Egress allowlist (only the model-API host) so Bash `curl`/`wget` also fail — fully airtight network.
- [ ] Per-instance difficulty tiering; down-rank message-string-pinned F2P tests (see collector §8a).

## 🚀 Open — scale / reusability
- [ ] Harness integration: `SPECS_PYTORCH` + per-era Docker base image + PyTorch log parser so
      upstream `swebench.harness.run_evaluation` grades these (the real reusable path).
- [ ] Expand corpus beyond v1 scope: C++/CUDA-source fixes (CPU build), then GPU-required tasks.
- [ ] Publish dataset (HF) + leaderboard via `sb-cli`, mirroring SWE-bench Multimodal's private test split.
- [ ] Overnight collection at larger scale (deeper discovery; the pipeline is resumable).

## ⚠️ Known gotchas (encoded in the runbooks — don't re-derive)
- Use a clean conda python (the system python's custom loader can break the build's native libs); if the host `git`/CLIs are linked against a custom libc, run them without the build `LD_LIBRARY_PATH`.
- Build: `BUILD_TEST=1`, NOT `USE_KINETO=0`, `pytest==7.4.4`, clean-rebuild-retry, `MAX_JOBS` cap.
- Solver reward-hacks git history — the airtight object-strip (real `.git` moved outside worktree) is required.
