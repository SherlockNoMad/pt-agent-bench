#!/usr/bin/env python3
"""Self-driving parallel collector+validator for PyTorch SWE-bench tasks.

N workers, each with its own git worktree + cloned conda env + build. Each worker
claims candidates from a shared pool (atomic mkdir), collects the tuple, checks out
base_commit, (re)builds, and runs the validation protocol. Valid instances are written
and appended to results.jsonl. Resumable: candidates already in results are skipped.

Run: nohup python3 run_overnight.py 6 100 > overnight.log 2>&1 &
"""
import subprocess, json, re, os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ROOT=config.WORKSPACE
SRC=config.SRC                 # main clone (worktree source)
BASE_ENV=config.BASE_ENV       # env to clone (has all build deps)
CAND=config.CANDIDATES
CLAIMS=config.COLLECT_CLAIMS
RESULTS=config.COLLECT_RESULTS
INST=config.INSTANCES          # validated instances land in the committed problems/instances/
LOGS=os.path.join(config.LOGS,"collect")
COLLECT_ONE=os.path.join(config.REPO,"collect","collect_one.py")
BUILD_TIMEOUT=2400
GENV={k:v for k,v in os.environ.items() if k!="LD_LIBRARY_PATH"}   # for git

for d in (CLAIMS,INST,LOGS,config.WORKTREES,config.ENVS): os.makedirs(d,exist_ok=True)

def log(wid,msg):
    line=f"[{time.strftime('%H:%M:%S')}][w{wid}] {msg}"
    print(line,flush=True)
    with open(os.path.join(LOGS,f"w{wid}.log"),"a") as f: f.write(line+"\n")

def git(worktree,*a,timeout=1200):
    return subprocess.run(["git","-C",worktree,*a],env=GENV,capture_output=True,text=True,timeout=timeout)

def done_issues():
    s=set()
    if os.path.exists(RESULTS):
        for l in open(RESULTS):
            try: s.add(str(json.loads(l)["issue"]))
            except Exception: pass
    return s

def record(rec):
    with open(RESULTS,"a") as f:   # append is atomic for short lines
        f.write(json.dumps(rec)+"\n")

def claim(issue):
    try:
        os.mkdir(os.path.join(CLAIMS,str(issue))); return True
    except FileExistsError:
        return False

def load_pool():
    pool=[]
    if os.path.exists(CAND):
        for l in open(CAND):
            try: pool.append(json.loads(l))
            except Exception: pass
    return pool

# ---------- environment ----------
def ensure_worker_env(wid):
    wt=config.wt(wid)
    env=config.env(wid)
    if not os.path.exists(os.path.join(wt,"setup.py")):
        # independent full copy of the (submoduled) main clone, minus build/ so
        # cmake's cached absolute paths don't point at src. Worktrees break submodules.
        subprocess.run(["rsync","-a","--exclude","/build/","--exclude","/dist/",
                        SRC+"/", wt+"/"],check=True,capture_output=True,text=True,timeout=3600)
    if not os.path.isdir(env):
        subprocess.run(["conda","create","-y","-p",env,"--clone",BASE_ENV],check=True,
                       capture_output=True,text=True,timeout=1200)
    return wt,env

def build_env_vars(env,worktree,jobs):
    return {**os.environ,
            "USE_CUDA":"0","USE_DISTRIBUTED":"0","USE_MKLDNN":"1","USE_FBGEMM":"1",
            "BUILD_TEST":"1","MAX_JOBS":str(jobs),
            "CMAKE_POLICY_VERSION_MINIMUM":"3.5",
            "PATH":f"{env}/bin:"+os.environ["PATH"],
            "LD_LIBRARY_PATH":f"/usr/lib64:{env}/lib"}

def build(wid,wt,env,jobs):
    # force BUILD_TEST reconfigure if cache says False
    cache=os.path.join(wt,"build","CMakeCache.txt")
    if os.path.exists(cache):
        try:
            txt=open(cache).read()
            if "BUILD_TEST:BOOL=False" in txt or "USE_KINETO:BOOL=OFF" in txt:
                os.remove(cache)   # force reconfigure when a build option changed
        except Exception: pass
    ev=build_env_vars(env,wt,jobs)
    py=os.path.join(env,"bin","python")
    r=subprocess.run([py,"setup.py","develop"],cwd=wt,env=ev,
                     capture_output=True,text=True,timeout=BUILD_TIMEOUT)
    if r.returncode!=0:
        # stale build-dir/codegen mismatch across commit jumps -> clean rebuild
        import shutil as _sh
        _sh.rmtree(os.path.join(wt,"build"),ignore_errors=True)
        r=subprocess.run([py,"setup.py","develop"],cwd=wt,env=ev,
                         capture_output=True,text=True,timeout=BUILD_TIMEOUT)
    return r.returncode==0, r.stdout[-3000:]+r.stderr[-3000:]

# ---------- validation (parameterized by worktree+env) ----------
def pytest_run(py,ev,wt,nodeids,timeout=900):
    r=subprocess.run([py,"-m","pytest","-q","-p","no:cacheprovider","--tb=no","-rA",*nodeids],
                     cwd=wt,env=ev,capture_output=True,text=True,timeout=timeout)
    st={}
    for line in r.stdout.splitlines():
        m=re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[[^\]]*\]\s+)?(test/\S+)",line)
        if m: st[m.group(2)]=m.group(1)
    return st

def collect_nodes(py,ev,wt,testfile,k=None,timeout=600):
    args=[py,"-m","pytest","--collect-only","-q","-p","no:cacheprovider",testfile]
    if k: args+=["-k",k]
    r=subprocess.run(args,cwd=wt,env=ev,capture_output=True,text=True,timeout=timeout)
    return [l for l in r.stdout.splitlines() if l.startswith(testfile+"::")]

def validate(wid,wt,env,jobs,outdir):
    inst=json.load(open(os.path.join(outdir,"instance.json")))
    base=inst["base_commit"]; tfile=inst["test_files"][0]
    py=os.path.join(env,"bin","python"); ev=build_env_vars(env,wt,jobs)
    tp=os.path.join(outdir,"patches/test_patch.diff"); gp=os.path.join(outdir,"patches/gold_patch.diff")
    def gapply(p):
        return git(wt,"apply",p).returncode==0
    git(wt,"checkout","-q","-f",base); git(wt,"clean","-qfd","test/","torch/","tools/")
    # all test functions touched by the patch (added OR modified — def appears in +/context/@@)
    new_names=sorted(set(re.findall(r'def (test_\w+)',open(tp).read())))
    if not new_names: return dict(valid=False,reason="no test fns")
    if not gapply(tp): return dict(valid=False,reason="test_patch apply failed")
    new_nodes=collect_nodes(py,ev,wt,tfile," or ".join(new_names))
    if not new_nodes: return dict(valid=False,reason="new tests not collected")
    pre=pytest_run(py,ev,wt,new_nodes)
    f2p=sorted(t for t in new_nodes if pre.get(t)=="FAILED")
    if not f2p: return dict(valid=False,reason=f"no new test fails pre-fix")
    all_nodes=collect_nodes(py,ev,wt,tfile)
    p2p_cand=[t for t in all_nodes if t not in new_nodes][:20]
    pre_p2p=pytest_run(py,ev,wt,p2p_cand) if p2p_cand else {}
    if not gapply(gp): return dict(valid=False,reason="gold apply failed")
    post=pytest_run(py,ev,wt,f2p)
    if not all(post.get(t)=="PASSED" for t in f2p):
        return dict(valid=False,reason="F2P not all pass post-fix")
    post_p2p=pytest_run(py,ev,wt,p2p_cand) if p2p_cand else {}
    p2p=sorted(t for t in p2p_cand if pre_p2p.get(t)=="PASSED" and post_p2p.get(t)=="PASSED")
    if [t for t in p2p_cand if pre_p2p.get(t)=="PASSED" and post_p2p.get(t)!="PASSED"]:
        return dict(valid=False,reason="regression")
    if not all(all(pytest_run(py,ev,wt,f2p).get(t)=="PASSED" for t in f2p) for _ in range(2)):
        return dict(valid=False,reason="flaky")
    inst["FAIL_TO_PASS"]=f2p; inst["PASS_TO_PASS"]=p2p
    json.dump(inst,open(os.path.join(outdir,"instance.json"),"w"),indent=2)
    git(wt,"checkout","-q","-f",base); git(wt,"clean","-qfd","test/","torch/","tools/")
    return dict(valid=True,FAIL_TO_PASS=f2p,PASS_TO_PASS_count=len(p2p))

# ---------- worker ----------
def worker(wid,jobs,target,stop_flag):
    try:
        wt,env=ensure_worker_env(wid)
        log(wid,f"env ready wt={wt}")
    except Exception as e:
        log(wid,f"FATAL env setup: {e}"); return
    last_created=None
    while not stop_flag.value:
        pool=load_pool(); done=done_issues()
        avail=[c for c in pool if str(c["issue"]) not in done
               and not os.path.exists(os.path.join(CLAIMS,str(c["issue"])))]
        if not avail:
            # maybe discovery still running; wait
            if os.path.exists(os.path.join(ROOT,"discover_done")): break
            time.sleep(30); continue
        # locality: pick candidate closest in created-time to last built (fewer rebuilds)
        if last_created:
            avail.sort(key=lambda c: _dist(c.get("created",""),last_created))
        cand=avail[0]; issue=cand["issue"]
        if not claim(issue): continue
        try:
            outdir=os.path.join(INST,cand.get("instance_id",f"pytorch__pytorch-{cand['pr']}"))
            # collect
            r=subprocess.run(["python3",COLLECT_ONE,str(issue),outdir],
                             capture_output=True,text=True,timeout=300)
            if not os.path.exists(os.path.join(outdir,"instance.json")):
                record(dict(issue=str(issue),valid=False,reason="collect_reject",out=r.stdout.strip()[:200]))
                log(wid,f"#{issue} collect REJECT"); continue
            inst=json.load(open(os.path.join(outdir,"instance.json")))
            # STATIC pre-filter (no build wasted): test_patch must touch some test function
            tp=os.path.join(outdir,"patches/test_patch.diff")
            tptext=open(tp).read()
            if not re.search(r'def (test_\w+)',tptext):
                record(dict(issue=str(issue),instance_id=inst["instance_id"],valid=False,reason="no_test_fns_static"))
                log(wid,f"#{issue} static-reject: no test fns"); continue
            # skip GPU/triton-gated tests pre-build (won't run on CPU-only) — saves a wasted build
            added="\n".join(l for l in tptext.splitlines() if l.startswith('+'))
            if any(m in added for m in ("requires_cuda","requires_gpu","requires_triton","onlyCUDA","requires_cuda_and_triton","skipCUDAIf")):
                record(dict(issue=str(issue),instance_id=inst["instance_id"],valid=False,reason="gpu_gated_static"))
                log(wid,f"#{issue} static-reject: gpu-gated test"); continue
            base=inst["base_commit"]
            # checkout + submodules + build
            git(wt,"checkout","-q","-f",base)
            git(wt,"submodule","update","--init","--recursive",timeout=1800)
            ok,blog=build(wid,wt,env,jobs)
            if not ok:
                record(dict(issue=str(issue),instance_id=inst["instance_id"],valid=False,reason="build_failed"))
                log(wid,f"#{issue} BUILD FAILED"); open(os.path.join(outdir,"build_err.log"),"w").write(blog); continue
            last_created=cand.get("created")
            # validate
            v=validate(wid,wt,env,jobs,outdir)
            v.update(issue=str(issue),instance_id=inst["instance_id"])
            record(v)
            n=sum(1 for l in open(RESULTS) if json.loads(l).get("valid"))
            log(wid,f"#{issue} -> valid={v['valid']} reason={v.get('reason','')} [total valid={n}]")
            if n>=target: stop_flag.value=1
        except Exception as e:
            record(dict(issue=str(issue),valid=False,reason=f"exception:{type(e).__name__}:{str(e)[:150]}"))
            log(wid,f"#{issue} EXCEPTION {e}")
    log(wid,"exiting")

def _dist(a,b):
    # ISO timestamp proximity in seconds (for base_commit locality -> smaller rebuilds)
    from datetime import datetime
    try:
        fa=datetime.fromisoformat((a or "").replace("Z","+00:00"))
        fb=datetime.fromisoformat((b or "").replace("Z","+00:00"))
        return abs((fa-fb).total_seconds())
    except Exception:
        return 1e18

def main(nworkers=6, target=100):
    jobs=min(40, max(8, 360//nworkers))   # cap per-worker jobs: avoid OOM-killing memory-heavy compiles (Functions.cpp etc.)
    stop=mp.Value('i',0)
    procs=[mp.Process(target=worker,args=(i,jobs,target,stop)) for i in range(nworkers)]
    for p in procs: p.start(); time.sleep(20)   # stagger env/build startup
    while any(p.is_alive() for p in procs):
        time.sleep(60)
        n=sum(1 for l in open(RESULTS) if json.loads(l).get("valid")) if os.path.exists(RESULTS) else 0
        tot=sum(1 for _ in open(RESULTS)) if os.path.exists(RESULTS) else 0
        print(f"== progress: {n} valid / {tot} attempted (target {target}) ==",flush=True)
        if n>=target: stop.value=1
    for p in procs: p.join()
    print("ALL WORKERS DONE")

if __name__=="__main__":
    nw=int(sys.argv[1]) if len(sys.argv)>1 else 6
    tg=int(sys.argv[2]) if len(sys.argv)>2 else 100
    main(nw,tg)
