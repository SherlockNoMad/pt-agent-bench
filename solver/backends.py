"""Pluggable solver backends. A backend runs a coding agent in a prepared worktree
(BLIND: already reset to base_commit, PROBLEM.txt written, git history stripped) — it edits
files in place and returns a metadata dict. Patch capture, grading, and the anti-reward-hacking
git-strip are backend-agnostic and handled by solve.py.

Public backends: claude-code, codex. Additional/private backends can live in a gitignored
`backends_internal.py` overlay that registers itself on import — kept out of this repo.

Backend contract:  run(wt, env, prompt, timeout, trace_path) -> dict(meta)
  wt          worktree dir (cwd for the agent; files edited here become the model_patch)
  env         the per-worker conda env prefix (agent uses <env>/bin/python to test its fix)
  prompt      the task instruction (references PROBLEM.txt in wt)
  timeout     seconds
  trace_path  where to save the agent transcript (for audit); backend writes if it can
  returns     {backend, model, cost_usd, wall_ms, num_turns, in_tok, out_tok, is_error, ...}
Run agents with a CLEAN env (no /usr/lib64 on LD_LIBRARY_PATH — vendored CLIs crash otherwise).
"""
import os, json, subprocess, shutil

GENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
REGISTRY = {}
def register(name):
    def deco(fn): REGISTRY[name] = fn; return fn
    return deco
def get_backend(name):
    if name not in REGISTRY:
        raise SystemExit(f"unknown backend '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name]

# ---------------- Claude Code ----------------
@register("claude-code")
def claude_code(wt, env, prompt, timeout, trace_path):
    """Anthropic Claude Code CLI. Locked down: --bare (no plugins/MCP/skills) + a tool
    allowlist under default permission mode (NOT bypass, which ignores allow/deny)."""
    cli   = os.environ.get("PTAB_CLAUDE_BIN", "claude")
    model = os.environ.get("PTAB_CLAUDE_MODEL", "claude-opus-4-8")
    effort= os.environ.get("PTAB_CLAUDE_EFFORT", "xhigh")
    allowed = ["Read", "Edit", "Write", "Grep", "Glob", "Bash"]
    cmd = [cli, "-p", prompt, "--bare", "--permission-mode", "default",
           "--model", model, "--effort", effort, "--output-format", "json",
           "--allowedTools", *allowed]
    meta = {"backend": "claude-code", "model": f"{model}/{effort}"}
    try:
        r = subprocess.run(cmd, cwd=wt, env=GENV, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)
        j = json.loads(r.stdout); u = j.get("usage", {}) or {}
        sid = j.get("session_id")
        meta.update(cost_usd=j.get("total_cost_usd"), wall_ms=j.get("duration_ms"),
                    num_turns=j.get("num_turns"), in_tok=u.get("input_tokens"),
                    out_tok=u.get("output_tokens"),
                    cache_read_tok=u.get("cache_read_input_tokens"),
                    cache_create_tok=u.get("cache_creation_input_tokens"),
                    is_error=j.get("is_error"), stop_reason=j.get("stop_reason"), session_id=sid)
        if sid and trace_path:
            proj = os.path.expanduser("~/.claude/projects/" + os.path.abspath(wt).replace("/", "-"))
            src = os.path.join(proj, f"{sid}.jsonl")
            if os.path.exists(src): shutil.copy(src, trace_path)
    except subprocess.TimeoutExpired:
        meta.update(is_error=True, timeout=True)
    except Exception as e:
        meta.update(is_error=True, error=f"{type(e).__name__}:{str(e)[:150]}")
    return meta

# ---------------- Codex ----------------
@register("codex")
def codex(wt, env, prompt, timeout, trace_path):
    """OpenAI Codex CLI (`codex exec`). Sandboxed `workspace-write` = edits allowed, network
    off (anti-reward-hacking). Provider/model via env (PTAB_CODEX_MODEL); any gateway/config
    comes from the user's own codex config, not hardcoded here."""
    cli   = os.environ.get("PTAB_CODEX_BIN", "codex")
    model = os.environ.get("PTAB_CODEX_MODEL")  # optional; None -> codex default
    cmd = [cli, "exec", prompt, "-C", wt, "-s", "workspace-write",
           "--skip-git-repo-check", "--json", "--color", "never"]
    if model: cmd[3:3] = ["-m", model]
    meta = {"backend": "codex", "model": model or "codex-default"}
    try:
        r = subprocess.run(cmd, cwd=wt, env=GENV, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)
        if trace_path:
            open(trace_path, "w").write(r.stdout)
        # best-effort usage/cost from the JSONL event stream
        intok = outtok = turns = 0; cost = None; err = r.returncode != 0
        for line in r.stdout.splitlines():
            try: ev = json.loads(line)
            except Exception: continue
            u = ev.get("usage") or (ev.get("info") or {}).get("usage") or {}
            if u:
                intok = u.get("input_tokens", intok); outtok = u.get("output_tokens", outtok)
            if isinstance(ev.get("total_cost_usd"), (int, float)): cost = ev["total_cost_usd"]
            if ev.get("type") in ("turn.completed", "response.completed"): turns += 1
        meta.update(cost_usd=cost, in_tok=intok or None, out_tok=outtok or None,
                    num_turns=turns or None, is_error=err)
    except subprocess.TimeoutExpired:
        meta.update(is_error=True, timeout=True)
    except Exception as e:
        meta.update(is_error=True, error=f"{type(e).__name__}:{str(e)[:150]}")
    return meta

# ---------------- optional private overlay (gitignored) ----------------
try:
    import backends_internal  # noqa: F401  (registers extra backends when present)
except Exception:
    pass
