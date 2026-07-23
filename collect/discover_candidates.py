#!/usr/bin/env python3
"""Discover python-only, CPU-friendly PyTorch task candidates.
issue(closed,COMPLETED) -> closing commit -> pull/N ; split files; filter scope.
Also flags message-pinned F2P tests (quality signal, runbook 8a)."""
import subprocess, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
GH=config.GH
BAD_PRE=("aten/","c10/","torch/csrc/","third_party/","caffe2/","cmake")
AVOID_TEST=("test/inductor","test/distributed","test/cpp","test/cuda","test/mobile","test/onnx")
Q='''query($q:String!,$after:String){ search(query:$q,type:ISSUE,first:40,after:$after){
 pageInfo{endCursor hasNextPage}
 nodes{... on Issue{ number title stateReason
   timelineItems(itemTypes:[CLOSED_EVENT],last:1){nodes{... on ClosedEvent{
     closer{__typename ... on Commit{oid message}}}}}}}}}'''

def gh_json(*args):
    return json.loads(subprocess.check_output([GH,*args],text=True))
def graphql(q,after):
    a=[GH,"api","graphql","-f","query="+Q,"-f","q="+q]
    if after:a+=["-f","after="+after]
    return json.loads(subprocess.check_output(a,text=True))["data"]["search"]

def main(target=20):
    q="repo:pytorch/pytorch is:issue is:closed sort:created-desc"
    issues=[];after=None
    while len(issues)<400:
        d=graphql(q,after);issues+=[n for n in d["nodes"] if n]
        if not d["pageInfo"]["hasNextPage"]:break
        after=d["pageInfo"]["endCursor"]
    prre=re.compile(r"pull/(\d+)")
    cands=[]
    for i in issues:
        if i.get("stateReason")!="COMPLETED": continue
        node=i["timelineItems"]["nodes"]
        if not node: continue
        c=node[0].get("closer") or {}
        if c.get("__typename")!="Commit": continue
        m=prre.search(c.get("message") or "")
        if not m: continue
        pr=int(m.group(1)); oid=c["oid"]
        try: cm=gh_json("api",f"repos/pytorch/pytorch/commits/{oid}")
        except Exception: continue
        files=cm["files"]; paths=[f["filename"] for f in files]
        tfiles=[p for p in paths if p.startswith("test/") and p.endswith(".py")]
        fix=[p for p in paths if not p.startswith("test/")]
        if not tfiles or not fix: continue
        if any(p.startswith(BAD_PRE) for p in fix): continue
        if not all(p.endswith((".py",".pyi")) for p in fix): continue
        if any(t.startswith(AVOID_TEST) for t in tfiles): continue
        churn=sum(f["additions"]+f["deletions"] for f in files)
        cands.append({"issue":i["number"],"pr":pr,"oid":oid,
                      "base_commit":cm["parents"][0]["sha"],
                      "title":i["title"],"test_files":tfiles,"fix_files":fix,
                      "n_files":len(paths),"churn":churn})
    cands.sort(key=lambda x:(x["n_files"],x["churn"]))
    out=cands[:target]
    with open(config.CANDIDATES,"w") as f:
        for c in out: f.write(json.dumps(c)+"\n")
    for c in out:
        print(f"#{c['issue']} PR{c['pr']} files={c['n_files']} churn={c['churn']}  {c['title'][:60]}")
        print(f"    test={c['test_files']} fix={c['fix_files']}")
    print(f"\n{len(out)} candidates -> candidates.jsonl")

if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 20)
