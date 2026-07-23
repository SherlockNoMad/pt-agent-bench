#!/usr/bin/env python3
"""collect_one.py <issue_number> <outdir>
Derive a task tuple from a closed issue: issue -> closing commit -> pull/N.
Writes <outdir>/instance.json + patches/. Prints a summary or REJECT reason."""
import subprocess, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GH=config.GH
BAD_PRE=("aten/","c10/","torch/csrc/","third_party/","caffe2/","cmake")
def ghj(*a): return json.loads(subprocess.check_output([GH,*a],text=True))

def main(issue_num, outdir):
    os.makedirs(os.path.join(outdir,"patches"),exist_ok=True)
    Q='''query($n:Int!){repository(owner:"pytorch",name:"pytorch"){issue(number:$n){
      number title body stateReason
      timelineItems(itemTypes:[CLOSED_EVENT],last:1){nodes{... on ClosedEvent{
        closer{__typename ... on Commit{oid message}}}}}}}}'''
    d=json.loads(subprocess.check_output([GH,"api","graphql","-f","query="+Q,"-F","n=%d"%issue_num],text=True))
    iss=d["data"]["repository"]["issue"]
    if iss.get("stateReason")!="COMPLETED": return print("REJECT: issue not COMPLETED")
    nodes=iss["timelineItems"]["nodes"]
    if not nodes or (nodes[0].get("closer") or {}).get("__typename")!="Commit":
        return print("REJECT: not closed by a commit")
    c=nodes[0]["closer"]; oid=c["oid"]
    m=re.search(r"pull/(\d+)",c.get("message") or "")
    if not m: return print("REJECT: closing commit has no pull/N")
    pr=int(m.group(1))
    cm=ghj("api",f"repos/pytorch/pytorch/commits/{oid}")
    base=cm["parents"][0]["sha"]
    files=[f["filename"] for f in cm["files"]]
    tfiles=[f for f in files if f.startswith("test/") and f.endswith(".py")]
    fix=[f for f in files if not f.startswith("test/")]
    if not tfiles or not fix: return print("REJECT: missing test or fix files")
    if any(f.startswith(BAD_PRE) for f in fix): return print(f"REJECT: non-python fix {fix}")
    if not all(f.endswith((".py",".pyi")) for f in fix): return print(f"REJECT: non-.py fix {fix}")
    # full diff, split
    full=subprocess.check_output([GH,"api",f"repos/pytorch/pytorch/commits/{oid}",
                                  "-H","Accept: application/vnd.github.diff"],text=True)
    parts=[p for p in re.split(r'(?m)(?=^diff --git )',full) if p.strip()]
    gold="".join(p for p in parts if not re.search(r'^diff --git a/(test/\S+)',p,re.M))
    test="".join(p for p in parts if re.search(r'^diff --git a/(test/\S+)',p,re.M))
    open(os.path.join(outdir,"patches/full.diff"),"w").write(full)
    open(os.path.join(outdir,"patches/gold_patch.diff"),"w").write(gold)
    open(os.path.join(outdir,"patches/test_patch.diff"),"w").write(test)
    # quality: message-pinned F2P?
    msg_pin=bool(re.search(r'assertRaisesRegex|assertRegex|str\(\s*\w*exc?\w*\s*\)',test))
    inst={"instance_id":f"pytorch__pytorch-{pr}","repo":"pytorch/pytorch",
          "issue_numbers":[str(issue_num)],"pull_number":pr,"base_commit":base,
          "problem_statement":(iss["title"]+"\n\n"+(iss.get("body") or "")).strip(),
          "patch":gold,"test_patch":test,"FAIL_TO_PASS":[],"PASS_TO_PASS":[],
          "test_files":tfiles,"fix_files":fix,"created_at":cm["commit"]["author"]["date"],
          "version":"era-8fabe0e","quality":{"f2p_asserts_message_string":msg_pin}}
    json.dump(inst,open(os.path.join(outdir,"instance.json"),"w"),indent=2)
    print(f"OK instance_id={inst['instance_id']} base={base[:10]} test={tfiles} fix={fix} msg_pin={msg_pin}")

if __name__=="__main__":
    main(int(sys.argv[1]), sys.argv[2])
