#!/usr/bin/env python3
"""Solve each collected task with a blind, pluggable agent backend, then grade it.

Per task: checkout base_commit -> build -> reset blind -> strip git history -> BACKEND solves
(sees only problem_statement + repo) -> capture model_patch -> apply model_patch+gold test_patch
-> run FAIL_TO_PASS/PASS_TO_PASS -> resolved. Appends to solve_results.jsonl.

Backend is pluggable (see solver/backends.py): claude-code (default), codex, + optional private overlay.
Usage: nohup python3 solve.py <nworkers> [limit] [--backend=NAME] > solve.log 2>&1 &
       (or PTAB_BACKEND=codex ...)
"""
import subprocess, json, re, os, sys, time, shutil, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `import backends` works
import config, backends

ROOT=config.WORKSPACE
SRC=config.SRC; BASE_ENV=config.BASE_ENV
INST=config.INSTANCES
DATASET=config.DATASET
SOLVE_RESULTS=config.SOLVE_RESULTS
SCLAIMS=config.SOLVE_CLAIMS
LOGS=config.LOGS            # worker logs (workspace, gitignored)
TRACES=config.TRACES        # full solver transcripts (committed for audit)
BACKEND=os.environ.get("PTAB_BACKEND","claude-code")   # pluggable: claude-code | codex | ... (see backends.py)
for _d in (SCLAIMS,LOGS,TRACES,config.WORKSPACE): os.makedirs(_d,exist_ok=True)
BUILD_TIMEOUT=2400; SOLVE_TIMEOUT=1800
GENV={k:v for k,v in os.environ.items() if k!="LD_LIBRARY_PATH"}

def log(wid,m):
    line=f"[{time.strftime('%H:%M:%S')}][s{wid}] {m}"; print(line,flush=True)
    open(os.path.join(LOGS,f"s{wid}.log"),"a").write(line+"\n")
def git(wt,*a,timeout=1200): return subprocess.run(["git","-C",wt,*a],env=GENV,capture_output=True,text=True,timeout=timeout)
def record(r):
    with open(SOLVE_RESULTS,"a") as f: f.write(json.dumps(r)+"\n")
def claim(iid):
    try: os.mkdir(os.path.join(SCLAIMS,iid)); return True
    except FileExistsError: return False
def done_ids():
    s=set()
    if os.path.exists(SOLVE_RESULTS):
        for l in open(SOLVE_RESULTS):
            try: s.add(json.loads(l)["instance_id"])
            except: pass
    return s

def ensure(wid):
    wt=config.wt(wid); env=config.env(wid)
    os.makedirs(config.WORKTREES,exist_ok=True); os.makedirs(config.ENVS,exist_ok=True)
    if not os.path.exists(os.path.join(wt,"setup.py")):
        subprocess.run(["rsync","-a","--exclude","/build/",SRC+"/",wt+"/"],check=True,capture_output=True,timeout=3600)
    _restore_history(wt)   # recover refs if a previous solve was killed mid-strip
    if not os.path.isdir(env):
        subprocess.run(["conda","create","-y","-p",env,"--clone",BASE_ENV],check=True,capture_output=True,timeout=1200)
    return wt,env

def bvars(env,jobs):
    return {**os.environ,"USE_CUDA":"0","USE_DISTRIBUTED":"0","USE_MKLDNN":"1","USE_FBGEMM":"1",
            "BUILD_TEST":"1","MAX_JOBS":str(jobs),"CMAKE_POLICY_VERSION_MINIMUM":"3.5",
            "PATH":f"{env}/bin:"+os.environ["PATH"],"LD_LIBRARY_PATH":f"/usr/lib64:{env}/lib"}

def build(wt,env,jobs):
    cache=os.path.join(wt,"build","CMakeCache.txt")
    if os.path.exists(cache):
        try:
            t=open(cache).read()
            if "BUILD_TEST:BOOL=False" in t or "USE_KINETO:BOOL=OFF" in t: os.remove(cache)
        except: pass
    ev=bvars(env,jobs); py=os.path.join(env,"bin","python")
    r=subprocess.run([py,"setup.py","develop"],cwd=wt,env=ev,capture_output=True,text=True,timeout=BUILD_TIMEOUT)
    if r.returncode!=0:
        import shutil; shutil.rmtree(os.path.join(wt,"build"),ignore_errors=True)
        r=subprocess.run([py,"setup.py","develop"],cwd=wt,env=ev,capture_output=True,text=True,timeout=BUILD_TIMEOUT)
    return r.returncode==0

def pytest_run(env,wt,nodeids,timeout=900):
    ev=bvars(env,1); py=os.path.join(env,"bin","python")
    r=subprocess.run([py,"-m","pytest","-q","-p","no:cacheprovider","--tb=no","-rA",*nodeids],
                     cwd=wt,env=ev,capture_output=True,text=True,timeout=timeout)
    st={}
    for line in r.stdout.splitlines():
        m=re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[[^\]]*\]\s+)?(test/\S+)",line)
        if m: st[m.group(2)]=m.group(1)
    return st

SOLVE_PROMPT="""Fix the bug described in ./PROBLEM.txt (a real PyTorch GitHub issue).

Rules:
- Edit ONLY non-test source under torch/ (and tools/ if needed). Do NOT modify or add anything under test/.
- Keep the fix minimal and correct.
- torch is already built; python-only edits take effect immediately.
- To run python: `export LD_LIBRARY_PATH=/usr/lib64:{env}/lib` then use `{env}/bin/python`.
- Run any git command with prefix `env -u LD_LIBRARY_PATH`.
You may write a throwaway repro script (not under test/) to check your fix, then delete it.
When done, just leave the edited files in place."""

GITBAK=config.GITBAK   # OUTSIDE any worktree — agents (cwd=wt) can't see it

def _bakdir(wt): return os.path.join(GITBAK, os.path.basename(wt.rstrip("/")))

def _strip_history(wt):
    """AIRTIGHT: move the real .git OUTSIDE the worktree and replace it with a fresh
    single-commit repo of the base tree. The future fix commit's OBJECTS are then genuinely
    absent, so `git show <fix_sha>` -> 'bad object' and `git log --all` shows one commit.
    (Ref/remote removal was NOT enough: the fix's objects are cached locally from the
    collection era, so `git show <sha>` still rendered the diff — opus exploited exactly this.)
    Interrupt-safe: the real .git is moved (atomic rename) to ROOT/.gitbak/<wt>, always
    restorable by _restore_history (called in `finally` and at worker startup)."""
    gd=os.path.join(wt,".git"); bak=_bakdir(wt)
    shutil.rmtree(bak,ignore_errors=True); os.makedirs(GITBAK,exist_ok=True)
    os.rename(gd,bak)                       # real .git safely OUTSIDE the worktree (atomic)
    git(wt,"init","-q","-b","base")
    git(wt,"add","-A")
    git(wt,"-c","user.email=b@b","-c","user.name=b","commit","-q","-m","base","--no-verify")

def _restore_history(wt):
    gd=os.path.join(wt,".git"); bak=_bakdir(wt)
    if not os.path.isdir(bak):
        return
    shutil.rmtree(gd,ignore_errors=True)   # drop the minimal repo
    os.rename(bak,gd)                        # restore real history for grading / next task

def solve(wid,wt,env,outdir,inst):
    # blind reset
    git(wt,"checkout","-q","-f",inst["base_commit"]); git(wt,"clean","-qfd","test/","torch/","tools/")
    pf=os.path.join(wt,"PROBLEM.txt")
    with open(pf,"w") as f: f.write(inst["problem_statement"])
    assert os.path.getsize(pf)>0, "empty problem_statement"
    prompt=SOLVE_PROMPT.format(env=env)
    trace_path=os.path.join(TRACES,f"{inst['instance_id']}.trace.jsonl")
    backend_fn=backends.get_backend(BACKEND)
    _strip_history(wt)   # hide the future fix commit from the solver (backend-agnostic)
    try:
        # backend runs the agent in wt (blind), edits files, writes its trace, returns meta
        meta=backend_fn(wt, env, prompt, SOLVE_TIMEOUT, trace_path)
    finally:
        _restore_history(wt)   # bring back real .git (HEAD detached at base_commit) for grading
    os.remove(pf)
    git(wt,"add","-N",".")
    d=git(wt,"diff","--","." ,":(exclude)test/*",":(exclude)PROBLEM.txt",":(exclude)agent_space/*")
    git(wt,"reset","-q")
    return d.stdout, meta

def grade(wt,env,inst,model_patch):
    base=inst["base_commit"]; f2p=inst["FAIL_TO_PASS"]; p2p=inst.get("PASS_TO_PASS",[])
    git(wt,"checkout","-q","-f",base); git(wt,"clean","-qfd","test/","torch/","tools/")
    # remove untracked files the patch would create
    for t in re.findall(r"^\+\+\+ b/(.+)$",model_patch,re.M):
        fp=os.path.join(wt,t)
        # only if new file (preceded by 'new file mode') — cheap: try remove if exists and patch has 'new file'
    open("/tmp/mp_%d.diff"%os.getpid(),"w").write(model_patch)
    mp_path="/tmp/mp_%d.diff"%os.getpid()
    # delete new-file targets
    for blk in re.split(r'(?m)(?=^diff --git )',model_patch):
        if "new file mode" in blk:
            m=re.search(r"^\+\+\+ b/(.+)$",blk,re.M)
            if m:
                fp=os.path.join(wt,m.group(1))
                if os.path.exists(fp): os.remove(fp)
    okm=git(wt,"apply",mp_path).returncode==0 if model_patch.strip() else True
    open("/tmp/tp_%d.diff"%os.getpid(),"w").write(inst["test_patch"])
    okt=git(wt,"apply","/tmp/tp_%d.diff"%os.getpid()).returncode==0
    if not (okm and okt): return {"resolved":False,"reason":"patch_apply_failed","patch_applied":okm}
    post=pytest_run(env,wt,f2p+p2p)
    f2p_ok=all(post.get(t)=="PASSED" for t in f2p)
    p2p_ok=all(post.get(t)=="PASSED" for t in p2p)
    git(wt,"checkout","-q","-f",base); git(wt,"clean","-qfd","test/","torch/","tools/")
    return {"resolved":bool(f2p_ok and p2p_ok),"f2p_pass":sum(post.get(t)=="PASSED" for t in f2p),
            "f2p_total":len(f2p),"p2p_ok":p2p_ok}

def worker(wid,tasks,jobs):
    try: wt,env=ensure(wid); log(wid,f"ready {wt}")
    except Exception as e: log(wid,f"FATAL {e}"); return
    for inst in tasks:
        iid=inst["instance_id"]
        if not claim(iid): continue
        try:
            t0=time.monotonic()
            git(wt,"checkout","-q","-f",inst["base_commit"])
            git(wt,"submodule","update","--init","--recursive",timeout=1800)
            tb=time.monotonic()
            ok=build(wt,env,jobs); t_build=round(time.monotonic()-tb,1)
            if not ok:
                record({"instance_id":iid,"resolved":False,"reason":"build_failed","t_build_s":t_build}); log(wid,f"{iid} build_failed ({t_build}s)"); continue
            ts=time.monotonic()
            mp,meta=solve(wid,wt,env,None,inst); t_solve=round(time.monotonic()-ts,1)
            tg=time.monotonic()
            g=grade(wt,env,inst,mp); t_grade=round(time.monotonic()-tg,1)
            g.update(instance_id=iid,patch_bytes=len(mp),
                     t_build_s=t_build,t_solve_s=t_solve,t_grade_s=t_grade,
                     t_e2e_s=round(time.monotonic()-t0,1),**meta)   # meta carries backend+model
            record(g)
            n=sum(1 for l in open(SOLVE_RESULTS) if json.loads(l).get("resolved"))
            log(wid,f"{iid} resolved={g['resolved']} ({g.get('f2p_pass')}/{g.get('f2p_total')} F2P) "
                    f"build={t_build}s solve={t_solve}s cost=${meta.get('cost_usd')} [total resolved={n}]")
        except Exception as e:
            record({"instance_id":iid,"resolved":False,"reason":f"exc:{type(e).__name__}:{str(e)[:120]}"}); log(wid,f"{iid} EXC {e}")

def main(nworkers=4, limit=None):
    backends.get_backend(BACKEND)   # validate early
    print(f"== backend: {BACKEND}  |  available: {sorted(backends.REGISTRY)} ==",flush=True)
    tasks=[json.loads(l) for l in open(DATASET) if l.strip()]
    # need F2P from inst/<id>/instance.json (dataset has it too)
    tasks=[t for t in tasks if t.get("FAIL_TO_PASS")]
    done=done_ids()
    # clear stale claims (claimed but no result) so interrupted tasks get reprocessed
    for c in os.listdir(SCLAIMS):
        if c not in done: shutil.rmtree(os.path.join(SCLAIMS,c),ignore_errors=True)
    tasks=[t for t in tasks if t["instance_id"] not in done]
    if limit: tasks=tasks[:limit]
    jobs=min(40,max(8,360//nworkers))
    shards=[tasks[i::nworkers] for i in range(nworkers)]
    procs=[mp.Process(target=worker,args=(i,shards[i],jobs)) for i in range(nworkers)]
    for p in procs: p.start(); time.sleep(15)
    while any(p.is_alive() for p in procs):
        time.sleep(60)
        if os.path.exists(SOLVE_RESULTS):
            r=[json.loads(l) for l in open(SOLVE_RESULTS)]
            print(f"== solved {sum(x.get('resolved') for x in r)} / graded {len(r)} ==",flush=True)
    for p in procs: p.join()
    print("DONE")

if __name__=="__main__":
    # usage: solve.py [nworkers] [limit] [--backend=NAME]   (or PTAB_BACKEND env)
    pos=[a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--backend="): BACKEND=a.split("=",1)[1]
    nw=int(pos[0]) if len(pos)>0 else 4
    lim=int(pos[1]) if len(pos)>1 else None
    main(nw,lim)
