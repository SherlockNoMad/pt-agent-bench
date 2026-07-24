"""Central path config for pt-agent-bench. All scripts import this so the layout has one
source of truth. Source-controlled artifacts live under the repo; heavy/regeneratable runtime
(pytorch checkout, conda envs, worktrees, logs, claim ledgers) lives under WORKSPACE, which is
gitignored and overridable via the PTAB_WORKSPACE env var."""
import os

REPO = os.path.dirname(os.path.abspath(__file__))

# --- source-controlled (committed) ---
PROBLEMS   = os.path.join(REPO, "problems")
INSTANCES  = os.path.join(PROBLEMS, "instances")            # per-task instance.json + patches/
DATASET    = os.path.join(PROBLEMS, "pt-agent-bench.jsonl") # the benchmark (one task per line)
RESULTS    = os.path.join(REPO, "results")
# TRACES / SOLVE_RESULTS are env-overridable so a scratch/dry run can be isolated from the
# committed corpus results (e.g. PTAB_TRACES / PTAB_SOLVE_RESULTS pointing at results/dryrun/).
TRACES     = os.environ.get("PTAB_TRACES", os.path.join(RESULTS, "traces"))  # solver transcripts
SOLVE_RESULTS = os.environ.get("PTAB_SOLVE_RESULTS", os.path.join(RESULTS, "solve_results.jsonl"))

# --- workspace (gitignored, rebuildable; see setup_workspace.sh) ---
WORKSPACE  = os.environ.get("PTAB_WORKSPACE", os.path.join(REPO, "workspace"))
SRC        = os.path.join(WORKSPACE, "src")            # main pytorch/pytorch clone
BASE_ENV   = os.path.join(WORKSPACE, "conda-env")      # base conda env (cloned per worker)
WORKTREES  = os.path.join(WORKSPACE, "worktrees")
ENVS       = os.path.join(WORKSPACE, "envs")
GITBAK     = os.path.join(WORKSPACE, ".gitbak")        # airtight git-history strip backups
LOGS       = os.path.join(WORKSPACE, "logs")           # worker + per-task claude logs
# collection runtime state
CANDIDATES = os.path.join(WORKSPACE, "candidates.jsonl")
SEEN_ISSUES = os.path.join(WORKSPACE, "seen_issues.txt")
COLLECT_RESULTS = os.path.join(WORKSPACE, "collect_results.jsonl")
COLLECT_CLAIMS  = os.path.join(WORKSPACE, "collect_claims")
# solve runtime state
SOLVE_CLAIMS = os.environ.get("PTAB_SOLVE_CLAIMS", os.path.join(WORKSPACE, "solve_claims"))

# --- external tools (overridable) ---
GH     = os.environ.get("PTAB_GH_BIN", "gh")
CLAUDE = os.environ.get("PTAB_CLAUDE_BIN", "claude")

def wt(i):  return os.path.join(WORKTREES, f"wt{i}")
def env(i): return os.path.join(ENVS, f"env{i}")

def _ensure_dirs():
    for d in (INSTANCES, RESULTS, TRACES, WORKSPACE, WORKTREES, ENVS, LOGS,
              COLLECT_CLAIMS, SOLVE_CLAIMS):
        os.makedirs(d, exist_ok=True)
