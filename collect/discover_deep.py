#!/usr/bin/env python3
"""Deep, resumable discovery of python-only CPU-friendly PyTorch candidates.
Windows by created-date to get past GitHub search's 1000-result cap.
Appends unique candidates to candidates.jsonl; tracks seen issues in seen_issues.txt."""
import subprocess, json, re, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GH=config.GH
BAD_PRE=("aten/","c10/","torch/csrc/","third_party/","caffe2/","cmake")
AVOID_TEST=("test/inductor","test/distributed","test/cpp","test/cuda","test/mobile","test/onnx")
CAND=config.CANDIDATES
SEEN=config.SEEN_ISSUES
os.makedirs(config.WORKSPACE,exist_ok=True)
Q='''query($q:String!,$after:String){ search(query:$q,type:ISSUE,first:50,after:$after){
 issueCount pageInfo{endCursor hasNextPage}
 nodes{... on Issue{ number title stateReason
   timelineItems(itemTypes:[CLOSED_EVENT],last:1){nodes{... on ClosedEvent{
     closer{__typename ... on Commit{oid message}}}}}}}}}'''

def graphql(q,after):
    a=[GH,"api","graphql","-f","query="+Q,"-f","q="+q]
    if after:a+=["-f","after="+after]
    for attempt in range(5):
        try: return json.loads(subprocess.check_output(a,text=True))["data"]["search"]
        except Exception as e:
            time.sleep(10*(attempt+1))
    return {"nodes":[],"pageInfo":{"hasNextPage":False,"endCursor":None}}

def gh_commit(oid):
    for attempt in range(4):
        try: return json.loads(subprocess.check_output([GH,"api",f"repos/pytorch/pytorch/commits/{oid}"],text=True))
        except Exception: time.sleep(5*(attempt+1))
    return None

def load_seen():
    return set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()

def month_windows(start="2026-07-31", stop_year=2022):
    y,m=int(start[:4]),int(start[5:7])
    while y>stop_year or (y==stop_year and m>=1):
        lo=f"{y:04d}-{m:02d}-01"
        # end of month via next month minus a day (approx: use 28 to be safe is wrong; use 01 of next as exclusive)
        ny,nm=(y,m+1) if m<12 else (y+1,1)
        hi=f"{ny:04d}-{nm:02d}-01"
        yield lo,hi
        y,m=(y,m-1) if m>1 else (y-1,12)

def main(target=200):
    seen=load_seen(); prre=re.compile(r"pull/(\d+)")
    have=sum(1 for _ in open(CAND)) if os.path.exists(CAND) else 0
    cf=open(CAND,"a"); sf=open(SEEN,"a")
    for lo,hi in month_windows():
        if have>=target: break
        q=f"repo:pytorch/pytorch is:issue is:closed created:{lo}..{hi} sort:created-desc"
        after=None; pages=0
        while True:
            d=graphql(q,after); pages+=1
            for i in d["nodes"]:
                if not i: continue
                num=i["number"]
                if str(num) in seen: continue
                seen.add(str(num)); sf.write(f"{num}\n"); sf.flush()
                if i.get("stateReason")!="COMPLETED": continue
                node=i["timelineItems"]["nodes"]
                if not node or (node[0].get("closer") or {}).get("__typename")!="Commit": continue
                c=node[0]["closer"]; m=prre.search(c.get("message") or "")
                if not m: continue
                cm=gh_commit(c["oid"])
                if not cm: continue
                files=[f["filename"] for f in cm["files"]]
                tfiles=[f for f in files if f.startswith("test/") and f.endswith(".py")]
                fix=[f for f in files if not f.startswith("test/")]
                if not tfiles or not fix: continue
                if any(f.startswith(BAD_PRE) for f in fix): continue
                if not all(f.endswith((".py",".pyi")) for f in fix): continue
                if any(t.startswith(AVOID_TEST) for t in tfiles): continue
                churn=sum(f["additions"]+f["deletions"] for f in cm["files"])
                rec={"issue":num,"pr":int(m.group(1)),"oid":c["oid"],
                     "base_commit":cm["parents"][0]["sha"],"title":i["title"],
                     "test_files":tfiles,"fix_files":fix,"n_files":len(files),"churn":churn,
                     "created":cm["commit"]["author"]["date"]}
                cf.write(json.dumps(rec)+"\n"); cf.flush(); have+=1
                print(f"[{have}] #{num} PR{rec['pr']} {i['title'][:55]}",flush=True)
                if have>=target: break
            if have>=target or not d["pageInfo"]["hasNextPage"]: break
            after=d["pageInfo"]["endCursor"]
        print(f"== window {lo}..{hi}: total candidates now {have} ==",flush=True)
    print(f"DONE total candidates: {have}")

if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
