#!/usr/bin/env python3
"""Enrich each problem instance with issue labels, timestamps, pre-fix conversation, and
provenance. Fetches from GitHub via `gh`. Rewrites problems/instances/<id>/instance.json and
rebuilds problems/pt-agent-bench.jsonl uniformly. Re-runnable (skips already-enriched).

hints_text = issue comments strictly BEFORE the fix-commit time (later comments can leak the
fix, so they are excluded). Adding fields is SWE-bench-compatible (extra keys are ignored)."""
import subprocess, json, os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REPO_GH="pytorch/pytorch"
CANON=["instance_id","repo","base_commit","problem_statement","patch","test_patch",
       "FAIL_TO_PASS","PASS_TO_PASS","version"]
NEW_ORDER=CANON+["issue_labels","issue_created_at","fix_commit_at","resolution_days",
    "hints_text","issue_url","pr_url","issue_numbers","pull_number",
    "fix_files","test_files","f2p_count","p2p_count","patch_size_loc"]

def gh(*a,paginate=False):
    cmd=[config.GH,"api"]+(["--paginate"] if paginate else [])+list(a)
    for att in range(4):
        try: return json.loads(subprocess.check_output(cmd,text=True))
        except Exception: time.sleep(4*(att+1))
    return None

def dt(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception: return None

def enrich(inst):
    issues=[str(x) for x in (inst.get("issue_numbers") or [])]
    pr=inst.get("pull_number")
    fix_at=dt(inst.get("created_at",""))            # fix-commit author date = leakage cutoff
    labels=set(); icreated=[]; iurl=None; comments=[]
    for n in issues:
        iss=gh(f"repos/{REPO_GH}/issues/{n}")
        if not iss: continue
        # keep real categorization labels; drop internal repro-pipeline artifact labels
        labels|={l["name"] for l in iss.get("labels",[]) if not l["name"].startswith("repro:")}
        if iss.get("created_at"): icreated.append(iss["created_at"])
        iurl=iurl or iss.get("html_url")
        cs=gh(f"repos/{REPO_GH}/issues/{n}/comments",paginate=True) or []
        for c in cs:
            cat=dt(c.get("created_at",""))
            if cat and fix_at and cat< fix_at:          # pre-fix only
                comments.append((c["created_at"],c.get("user",{}).get("login","?"),c.get("body","") or ""))
    comments.sort()
    hints="\n\n".join(f"[{u} · {t[:10]}]\n{b}".strip() for t,u,b in comments)
    prd=gh(f"repos/{REPO_GH}/pulls/{pr}") if pr else None
    icreated_min=min(icreated) if icreated else None
    # (no pr_merged_at: PyTorch lands via ghstack, GitHub merged_at is always null)
    # resolution = fix-commit landing time - issue open time (ghstack-safe)
    res_days=None
    if icreated_min and fix_at:
        res_days=round((fix_at-dt(icreated_min)).total_seconds()/86400,1)
    inst.update(
        issue_labels=sorted(labels),
        issue_created_at=icreated_min,
        fix_commit_at=inst.get("created_at"),
        resolution_days=res_days,
        hints_text=hints,
        issue_url=iurl,
        pr_url=(prd.get("html_url") if prd else None),
        f2p_count=len(inst.get("FAIL_TO_PASS",[])),
        p2p_count=len(inst.get("PASS_TO_PASS",[])),
        patch_size_loc=sum(1 for ln in (inst.get("patch","") or "").splitlines()
                           if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++","---"))),
    )
    return inst

def main():
    ids=sorted(os.listdir(config.INSTANCES))
    done=0
    for i,iid in enumerate(ids):
        p=os.path.join(config.INSTANCES,iid,"instance.json")
        if not os.path.exists(p): continue
        inst=json.load(open(p))
        if "issue_labels" in inst and "--force" not in sys.argv:
            done+=1; continue
        inst=enrich(inst)
        json.dump(inst,open(p,"w"),indent=2)
        done+=1
        print(f"[{done}/{len(ids)}] {iid}  labels={len(inst['issue_labels'])} "
              f"hints={len(inst['hints_text'])}c res={inst['resolution_days']}d",flush=True)
    # rebuild dataset jsonl uniformly (canonical + new fields, consistent key order)
    rows=[]
    for iid in ids:
        p=os.path.join(config.INSTANCES,iid,"instance.json")
        if not os.path.exists(p): continue
        inst=json.load(open(p))
        rows.append({k:inst.get(k) for k in NEW_ORDER})
    open(config.DATASET,"w").write("".join(json.dumps(r)+"\n" for r in rows))
    print(f"rebuilt {config.DATASET} with {len(rows)} rows, {len(NEW_ORDER)} fields")

if __name__=="__main__": main()
