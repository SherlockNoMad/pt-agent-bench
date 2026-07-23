#!/usr/bin/env python3
"""Render the results deck as 1920x1080 PNGs (one per slide) for Google Slides.
Palette = dataviz reference (validated). Run: python make_slides.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"slides")
os.makedirs(OUT,exist_ok=True)

SURF="#fcfcfb"; PLANE="#f2f1ee"; INK="#0b0b0b"; INK2="#52514e"; MUT="#898781"
S1="#2a78d6"; S2="#eb6834"; S3="#1baf7a"; GOOD="#0ca30c"; RING="#deddd6"
F="DejaVu Sans"

def fig():
    f=plt.figure(figsize=(12.8,7.2),dpi=150); f.patch.set_facecolor(SURF)
    ax=f.add_axes([0,0,1,1]); ax.set_xlim(0,1280); ax.set_ylim(0,720); ax.axis("off"); ax.invert_yaxis()
    return f,ax
def T(ax,x,y,s,size,color=INK,w="normal",style="normal",font=F,ha="left"):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=w,style=style,family=font,ha=ha,va="baseline")
def kicker(ax,s,color=S1): T(ax,72,88,s.upper(),15,color,"bold")
def tile(ax,x,y,w,h,val,key,vcolor=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=14",
        fc=PLANE,ec=RING,lw=1))
    T(ax,x+22,y+62,val,40,vcolor,"bold"); T(ax,x+22,y+h-22,key,15,INK2)
def bullet(ax,x,y,s,color=S1,size=21):
    ax.add_patch(Rectangle((x,y-13),12,12,fc=color,ec="none"))
    T(ax,x+26,y,s,size,INK)
def pg(ax,n): T(ax,1208,700,str(n),14,MUT)
def save(f,name): f.savefig(os.path.join(OUT,name),facecolor=SURF); plt.close(f)

# 1 — title
f,ax=fig()
kicker(ax,"Benchmark · Results")
T(ax,72,220,"pt-agent-bench",64,INK,"bold")
T(ax,72,275,"An execution-verified, agentic SWE benchmark built from real",24,INK2)
T(ax,72,312,"pytorch/pytorch issues — extending SWE-bench to PyTorch.",24,INK2)
for i,(c,s) in enumerate([(S1,"99 validated tasks"),(GOOD,"opus-4.8 / xhigh blind baseline"),(S2,"audited · no reward hacking")]):
    y=420+i*44; ax.add_patch(Rectangle((72,y-14),14,14,fc=c,ec="none")); T(ax,96,y,s,21,INK)
T(ax,72,660,"2026-07-23",14,MUT)
save(f,"01_title.png")

# 2 — what it is
f,ax=fig(); kicker(ax,"What it is")
T(ax,72,150,"Real issue + fix, graded by tests",34,INK,"bold")
rows=[(S1,"Input: repo at the pre-fix commit + the GitHub issue text"),
      (S1,"The agent must produce a patch"),
      (S1,"Graded by running tests — no diff matching"),
      (GOOD,"FAIL_TO_PASS: the PR's tests go failing → passing"),
      (S2,"PASS_TO_PASS: existing tests stay green")]
for i,(c,s) in enumerate(rows): bullet(ax,72,230+i*52,s,c)
T(ax,72,545,"Why PyTorch is harder than SWE-bench's Python repos:",18,INK2)
for i,s in enumerate(["From-source C++/CUDA build per task (~9 min)",
    "ghstack merges → link issues via the closing commit, not PR refs",
    "Solver runs a real coding agent → must be sandboxed & audited"]):
    bullet(ax,72,585+i*40,s,S3,18)
pg(ax,2); save(f,"02_what.png")

# 3 — how built
f,ax=fig(); kicker(ax,"How it's built")
T(ax,72,150,"Collection → 99 validated tasks",34,INK,"bold")
for i,(c,s) in enumerate([(S1,"Closed issue → closing commit → 'Pull Request resolved: /pull/N'"),
    (S1,"Split squashed commit into gold patch + test patch; parent = base_commit"),
    (S1,"Filter: Python-only fix, CPU-runnable test, has a repro test"),
    (GOOD,"Validate: build → fail pre-fix → pass post-fix → no regressions → deterministic")]):
    bullet(ax,72,235+i*54,s,c,19)
for i,(v,k) in enumerate([("99","validated tasks"),("6×","parallel build workers"),
    ("~41%","closed issues w/ fix commit"),("CPU","Python-only · v1 scope")]):
    tile(ax,72+i*288,520,268,120,v,k)
pg(ax,3); save(f,"03_pipeline.png")

# 4 — headline
f,ax=fig(); kicker(ax,"Headline result")
T(ax,72,150,"Blind opus-4.8 / xhigh solves 61.6%",34,INK,"bold")
T(ax,72,340,"61.6%",104,INK,"bold")
T(ax,72,430,"61 of 99 tasks resolved on a single blind attempt.",22,INK2)
# proportion bar
bx,by,bw=680,250,520
rw=bw*61/99-3
ax.add_patch(FancyBboxPatch((bx,by),rw,54,boxstyle="round,pad=0,rounding_size=7",fc=S1,ec="none"))
ax.add_patch(FancyBboxPatch((bx+rw+6,by),bw-rw-6,54,boxstyle="round,pad=0,rounding_size=7",fc=S2,ec="none"))
T(ax,bx+16,by+34,"61 resolved",19,"#ffffff","bold")
T(ax,bx+rw+22,by+34,"38 not",19,"#ffffff","bold")
for i,(c,s) in enumerate([(S1,"Resolved  61"),(S2,"Unresolved  38  (11 partial)")]):
    ax.add_patch(Rectangle((bx+i*230,by+92),13,13,fc=c,ec="none")); T(ax,bx+i*230+22,by+104,s,16,INK2)
T(ax,72,640,"Single blind, single-shot attempt · one model — a capability snapshot, not a leaderboard number.",15,MUT)
pg(ax,4); save(f,"04_result.png")

# 5 — cost & speed
f,ax=fig(); kicker(ax,"Cost & speed")
T(ax,72,150,"$139 to run the whole benchmark",34,INK,"bold")
for i,(v,k) in enumerate([("$139","total cost · 99 tasks"),("$1.51","mean / task · med $0.95"),
    ("$0.45","cost / min · 6 workers"),("~5.2h","wall-clock · ~19/hr")]):
    tile(ax,72+i*288,205,268,120,v,k)
kicker2_y=410; T(ax,72,kicker2_y,"WHERE THE TIME GOES — MEAN MINUTES / TASK",15,S1,"bold")
bx,by,bw=72,445,1136; tot=18.7; sc=bw/tot
segs=[("Build",8.8,S1),("Solve",9.2,S2),("Grade",0.6,S3)]; x=bx
for name,val,c in segs:
    w=val*sc-3
    ax.add_patch(FancyBboxPatch((x,by),w,48,boxstyle="round,pad=0,rounding_size=6",fc=c,ec="none"))
    if w>90: T(ax,x+14,by+31,f"{name} {val}m",17,"#ffffff","bold")
    x+=val*sc
T(ax,bx,by+82,"0",14,MUT); T(ax,bx+bw,by+82,"18.7 min end-to-end",14,MUT,ha="right")
for i,(name,val,c) in enumerate(segs):
    ax.add_patch(Rectangle((72+i*230,by+108),13,13,fc=c,ec="none")); T(ax,72+i*230+22,by+120,f"{name} {val}m",16,INK2)
pg(ax,5); save(f,"05_cost.png")

# 6 — validity
f,ax=fig(); kicker(ax,"Validity")
T(ax,72,150,"Survives a reward-hacking audit",34,INK,"bold")
for i,s in enumerate(["Tool lockdown (--bare): no web, MCP, sub-agents, or skills",
    "Airtight git-history strip → git show <fix> returns 'bad object'",
    "Every solver trace saved & audited; flagged solves excluded"]):
    bullet(ax,72,230+i*50,s,S1,19)
# callout box (full width, below bullets)
bx,by,bw,bh=72,410,1136,190
ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle="round,pad=0,rounding_size=14",fc=PLANE,ec=RING,lw=1))
T(ax,bx+28,by+48,"opus actively tried to cheat — and was caught & blocked:",19,INK,"bold")
for i,(c,s) in enumerate([(S2,"web search for the PR"),
    (S2,"git log --all / git show <sha>"),(GOOD,"audit: 90 clean + 2 blocked")]):
    x=bx+28+i*372
    ax.add_patch(Rectangle((x,by+108),13,13,fc=c,ec="none")); T(ax,x+24,by+120,s,17,INK)
T(ax,72,660,"Sandbox the task, not the agent — it keeps full repo tools; it just can't reach the internet or the answer.",15,MUT)
pg(ax,6); save(f,"06_validity.png")

# 7 — status
f,ax=fig(); kicker(ax,"Status & roadmap")
T(ax,72,150,"Public-release ready",34,INK,"bold")
T(ax,72,240,"DONE",16,GOOD,"bold")
for i,s in enumerate(["99-task benchmark + collect/solve/grade/audit harness",
    "opus-4.8/xhigh baseline: 61.6%, audited","clean repo: docs, README, MIT, rebuildable workspace"]):
    bullet(ax,72,290+i*48,s,GOOD,19)
T(ax,72,480,"NEXT",16,S2,"bold")
for i,s in enumerate(["push to GitHub + publish dataset (HuggingFace)",
    "pass@k + a second model for comparison","expand: C++/CUDA-source & GPU tasks"]):
    bullet(ax,72,530+i*48,s,S2,19)
T(ax,72,690,"github.com/SherlockNoMad/pt-agent-bench  ·  MIT",14,MUT)
save(f,"07_status.png")

print("wrote", len(os.listdir(OUT)), "PNGs to", OUT)
