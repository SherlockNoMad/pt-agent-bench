#!/usr/bin/env python3
"""validate_one.py <outdir>
Assumes the workspace pytorch tree (config.SRC) is checked out at the
instance's base_commit AND built. Runs the validation protocol on
TARGETED tests (fast): reproduce (fail) -> fix (pass) -> P2P -> determinism.
Updates instance.json with FAIL_TO_PASS / PASS_TO_PASS and prints a verdict."""
import subprocess, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SRC=config.SRC
PY=os.path.join(config.BASE_ENV,"bin","python")
RUNENV={**os.environ,"LD_LIBRARY_PATH":f"/usr/lib64:{config.BASE_ENV}/lib"}
GENV={k:v for k,v in os.environ.items() if k!="LD_LIBRARY_PATH"}
P2P_CAP=20

def git(*a): return subprocess.run(["git",*a],cwd=SRC,env=GENV,capture_output=True,text=True)
def apply(p):
    r=git("apply",p)
    return r.returncode==0
def pytest(nodeids, timeout=600):
    r=subprocess.run([PY,"-m","pytest","-q","-p","no:cacheprovider","--tb=no","-rA",*nodeids],
                     cwd=SRC,env=RUNENV,capture_output=True,text=True,timeout=timeout)
    st={}
    for line in r.stdout.splitlines():
        m=re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[[^\]]*\]\s+)?(test/\S+)",line)
        if m: st[m.group(2)]=m.group(1)
    return st, r.stdout

def collect(testfile, k=None):
    args=[PY,"-m","pytest","--collect-only","-q","-p","no:cacheprovider",testfile]
    if k: args+=["-k",k]
    r=subprocess.run(args,cwd=SRC,env=RUNENV,capture_output=True,text=True,timeout=600)
    return [l for l in r.stdout.splitlines() if l.startswith(testfile+"::")]

def main(outdir):
    inst=json.load(open(os.path.join(outdir,"instance.json")))
    base=inst["base_commit"]; tfile=inst["test_files"][0]
    tp=os.path.join(outdir,"patches/test_patch.diff")
    gp=os.path.join(outdir,"patches/gold_patch.diff")
    verdict={"instance_id":inst["instance_id"],"base_commit":base[:10]}

    # ensure clean base tree
    git("checkout","-q","-f",base); git("clean","-qfd","test/")
    # names of NEW test functions from the test patch
    new_names=sorted(set(re.findall(r'^\+\s*(?:async\s+)?def (test_\w+)',open(tp).read(),re.M)))
    if not new_names:
        verdict.update(valid=False,reason="no new test_ functions in test_patch"); return out(verdict,outdir,inst)

    # PRE state: base + test_patch
    if not apply(tp): verdict.update(valid=False,reason="test_patch failed to apply"); return out(verdict,outdir,inst)
    k=" or ".join(new_names)
    new_nodes=collect(tfile,k)
    if not new_nodes: verdict.update(valid=False,reason="new tests not collected"); return out(verdict,outdir,inst)
    pre,_=pytest(new_nodes)
    f2p=sorted(t for t in new_nodes if pre.get(t)=="FAILED")
    if not f2p:
        verdict.update(valid=False,reason=f"no new test fails pre-fix (states={ {t:pre.get(t) for t in new_nodes} })")
        return out(verdict,outdir,inst)
    # P2P candidates: other tests in same file (cap)
    all_nodes=collect(tfile)
    p2p_cand=[t for t in all_nodes if t not in new_nodes][:P2P_CAP]
    pre_p2p,_=pytest(p2p_cand) if p2p_cand else ({},"")

    # POST state: + gold
    if not apply(gp): verdict.update(valid=False,reason="gold_patch failed to apply"); return out(verdict,outdir,inst)
    post,_=pytest(f2p)
    if not all(post.get(t)=="PASSED" for t in f2p):
        verdict.update(valid=False,reason=f"F2P not all passing post-fix: { {t:post.get(t) for t in f2p} }")
        return out(verdict,outdir,inst)
    post_p2p,_=pytest(p2p_cand) if p2p_cand else ({},"")
    p2p=sorted(t for t in p2p_cand if pre_p2p.get(t)=="PASSED" and post_p2p.get(t)=="PASSED")
    regressions=[t for t in p2p_cand if pre_p2p.get(t)=="PASSED" and post_p2p.get(t)!="PASSED"]
    if regressions:
        verdict.update(valid=False,reason=f"regressions: {regressions}"); return out(verdict,outdir,inst)
    # determinism: rerun F2P 2x
    det=[all(pytest(f2p)[0].get(t)=="PASSED" for t in f2p) for _ in range(2)]
    if not all(det):
        verdict.update(valid=False,reason="flaky: F2P not deterministic"); return out(verdict,outdir,inst)

    inst["FAIL_TO_PASS"]=f2p; inst["PASS_TO_PASS"]=p2p
    verdict.update(valid=True,FAIL_TO_PASS=f2p,PASS_TO_PASS_count=len(p2p),
                   quality=inst.get("quality"))
    # reset tree
    git("checkout","-q","-f",base); git("clean","-qfd","test/")
    return out(verdict,outdir,inst)

def out(v,outdir,inst):
    json.dump(inst,open(os.path.join(outdir,"instance.json"),"w"),indent=2)
    json.dump(v,open(os.path.join(outdir,"validation.json"),"w"),indent=2)
    print(json.dumps(v,indent=2))
    print(">>> VALID:",v.get("valid"))

if __name__=="__main__":
    main(sys.argv[1])
