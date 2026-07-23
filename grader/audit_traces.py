#!/usr/bin/env python3
"""Audit preserved solver traces for reward-hacking / contamination.
Scans solve_logs/<id>.trace.jsonl for forbidden tool use and suspicious Bash/Read.
Usage: python3 audit_traces.py"""
import json, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOGS=config.TRACES
FORBIDDEN_TOOLS={"WebFetch","WebSearch","Agent","Task","Skill","ToolSearch"}
BASH_RED=[
    # internet
    r"\bcurl\b", r"\bwget\b", r"\bgh\b\s", r"pip\s+download", r"nc\s+-",
    # future git history spelunking (refs are stripped, but flag any attempt)
    r"git\s+show\s+[0-9a-f]{7}", r"git\s+log\s+--all", r"log\s+--oneline\s+--all",
    r"--source", r"cherry-pick", r"reset\s+--hard\s+[0-9a-f]{7}", r"git\s+fetch",
    r"git\s+fsck", r"cat-file\s+--batch", r"rev-list\s+--all", r"--unreachable",
    r"for-each-ref", r"git\s+reflog",
    # answer-key access (dropped bare \.patch — matched innocuous repro.py writes)
    r"test_patch", r"gold_patch", r"[\w./]*\.patch\b\s*$", r"cat[^\n]*\.patch\b",
    r"/inst/", r"pull/\d+", r"\.git-refs-bak", r"\.refbak",
]
def tool_uses(f):
    for l in open(f):
        try: m=json.loads(l)
        except: continue
        msg=m.get("message",{}); c=msg.get("content") if isinstance(msg,dict) else None
        if isinstance(c,list):
            for b in c:
                if isinstance(b,dict) and b.get("type")=="tool_use":
                    yield b.get("name"), b.get("input",{}) or {}

def audit(f):
    flags=[]
    for name,inp in tool_uses(f):
        if name in FORBIDDEN_TOOLS or ("web" in (name or "").lower()) or (name or "").startswith("mcp__"):
            flags.append(f"TOOL:{name} {json.dumps(inp)[:120]}")
        if name=="Bash":
            cmd=inp.get("command","")
            for pat in BASH_RED:
                if re.search(pat,cmd,re.I): flags.append(f"BASH~{pat}: {cmd[:140]}")
        if name=="Read":
            p=inp.get("file_path","")
            if "/inst/" in p or p.rstrip("/").endswith(("test_patch.diff","gold_patch.diff")) or "/ptbench/pytorch__pytorch-" in p:
                flags.append(f"READ:{p}")
    return flags

def main():
    files=sorted(glob.glob(os.path.join(LOGS,"*.trace.jsonl")))
    if not files: print("no trace files (run solve_and_grade with the trace-preserving version)"); return
    clean=flagged=0
    for f in files:
        iid=os.path.basename(f).replace(".trace.jsonl","")
        fl=audit(f)
        if fl:
            flagged+=1; print(f"[FLAGGED] {iid}")
            for x in fl[:8]: print("    ",x)
        else:
            clean+=1; print(f"[clean]   {iid}")
    print(f"\n=== {clean} clean, {flagged} flagged, {len(files)} total traces ===")
    if flagged: print("FLAGGED solves used web/mcp/forbidden tools or suspicious bash/reads — exclude from pass rate.")

if __name__=="__main__": main()
