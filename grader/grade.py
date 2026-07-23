#!/usr/bin/env python3
"""Minimal Docker-free grader: mirrors swebench run_evaluation's core loop.
Applies model_patch + gold test_patch to repo@base_commit, runs FAIL_TO_PASS /
PASS_TO_PASS, and reports `resolved`."""
import json, re, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REPO = config.SRC                                  # the built pytorch worktree to grade in
PY = os.path.join(config.BASE_ENV, "bin", "python")
RUNENV = {**os.environ, "LD_LIBRARY_PATH": f"/usr/lib64:{config.BASE_ENV}/lib"}

def git(*args):
    # host git must run WITHOUT LD_LIBRARY_PATH
    e = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    return subprocess.run(["git", *args], cwd=REPO, env=e, capture_output=True, text=True)

def _new_file_targets(diff_text):
    targets, lines = [], diff_text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("new file mode"):
            for j in range(i, min(i + 6, len(lines))):
                m = re.match(r"\+\+\+ b/(.+)", lines[j])
                if m:
                    targets.append(m.group(1)); break
    return targets

def apply_patch(path):
    txt = open(path).read()
    # remove any untracked files the patch would create (git apply refuses to overwrite)
    for t in _new_file_targets(txt):
        fp = os.path.join(REPO, t)
        if os.path.exists(fp):
            os.remove(fp)
    r = git("apply", path)
    if r.returncode != 0:
        r = git("apply", "--reject", "--whitespace=fix", path)
    return r.returncode == 0

def run_tests(nodeids):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--tb=no", "-rA", *nodeids],
                       cwd=REPO, env=RUNENV, capture_output=True, text=True)
    status = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[[^\]]*\]\s+)?(test/\S+)", line)
        if m:
            status[m.group(2)] = m.group(1)
    return status, r.stdout

def main(instance_json, predictions_jsonl):
    inst = json.load(open(instance_json))
    preds = {json.loads(l)["instance_id"]: json.loads(l)
             for l in open(predictions_jsonl) if l.strip()}
    iid = inst["instance_id"]
    pred = preds[iid]
    f2p, p2p = inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"]

    # 1. clean tree @ base_commit
    git("checkout", "-q", "-f", inst["base_commit"])
    git("clean", "-qfd", "test/")
    # 2. apply model prediction
    open("/tmp/mp.diff", "w").write(pred["model_patch"])
    ok_model = apply_patch("/tmp/mp.diff") if pred["model_patch"].strip() else True
    # 3. apply gold test patch
    open("/tmp/tp.diff", "w").write(inst["test_patch"])
    ok_test = apply_patch("/tmp/tp.diff")

    report = {"instance_id": iid, "model": pred["model_name_or_path"],
              "patch_applied": ok_model, "test_patch_applied": ok_test}
    if not (ok_model and ok_test):
        report["resolved"] = False
        report["reason"] = "patch did not apply"
    else:
        status, _ = run_tests(f2p + p2p)
        f2p_res = {t: status.get(t, "MISSING") for t in f2p}
        p2p_res = {t: status.get(t, "MISSING") for t in p2p}
        f2p_ok = all(v == "PASSED" for v in f2p_res.values())
        p2p_ok = all(v == "PASSED" for v in p2p_res.values())
        report["FAIL_TO_PASS"] = {"success": [t for t, v in f2p_res.items() if v == "PASSED"],
                                  "failure": [t for t, v in f2p_res.items() if v != "PASSED"]}
        report["PASS_TO_PASS_ok"] = p2p_ok
        report["PASS_TO_PASS_failures"] = [t for t, v in p2p_res.items() if v != "PASSED"]
        report["resolved"] = bool(f2p_ok and p2p_ok)

    # cleanup
    git("checkout", "-q", "-f", inst["base_commit"]); git("clean", "-qfd", "test/")
    out = os.path.join(config.RESULTS, f"report_{pred['model_name_or_path']}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("\n>>> RESOLVED:", report["resolved"])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
