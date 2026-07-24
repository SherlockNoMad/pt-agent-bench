#!/usr/bin/env python3
"""Render the pt-agent-bench corpus distribution as PNG charts into results/figures/.

Uses matplotlib with the data-viz reference palette (light mode, validated). Produces one PNG per
chart plus a combined dashboard.png. Needs matplotlib (present in the workspace conda-env):

    python results/plot_png.py     # use the workspace conda-env's python
"""
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

FIGDIR = os.path.join(config.RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --- palette (data-viz reference, light) ---
SURF, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
S1, S2, REM = "#2a78d6", "#eb6834", "#dcdbd4"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUT, "ytick.color": MUT, "axes.titlecolor": INK,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "svg.fonttype": "none",
})

# ---------------------------------------------------------------- data --------
rows = [json.loads(l) for l in open(config.DATASET)]
N = len(rows)
by_id = {r["instance_id"]: r for r in rows}
res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(config.SOLVE_RESULTS)}
graded = [(iid, d) for iid, d in res.items() if iid in by_id]


def area_of(f):
    p = f.split("/")
    return "/".join(p[:2]) if len(p) > 1 else p[0]


def short(a):
    return a[len("torch/"):] if a.startswith("torch/") else a


def size_bucket(s):
    return ("1–5" if s <= 5 else "6–15" if s <= 15 else
            "16–30" if s <= 30 else "31–60" if s <= 60 else "60+")


SIZE_ORDER = ["1–5", "6–15", "16–30", "31–60", "60+"]

fix_area = Counter(area_of(f) for r in rows for f in (r["fix_files"] or []))
mod_label = Counter(l for r in rows for l in (r["issue_labels"] or []) if l.startswith("module:"))
size_hist = Counter(size_bucket(r["patch_size_loc"]) for r in rows)
f2p_hist = Counter(r["f2p_count"] for r in rows)


def rate_table(keyfn, order=None, min_n=1):
    b = defaultdict(lambda: [0, 0])
    for iid, d in graded:
        for k in keyfn(by_id[iid]):
            b[k][1] += 1
            b[k][0] += 1 if d["resolved"] else 0
    items = [(k, p, t) for k, (p, t) in b.items() if t >= min_n]
    items.sort(key=(lambda x: order.index(x[0])) if order else (lambda x: -x[2]))
    return items


rate_size = rate_table(lambda r: [size_bucket(r["patch_size_loc"])], SIZE_ORDER)
rate_f2p = rate_table(lambda r: ["1 test" if r["f2p_count"] == 1 else
                                 "2 tests" if r["f2p_count"] == 2 else "3+ tests"],
                      ["1 test", "2 tests", "3+ tests"])
rate_area = rate_table(lambda r: {area_of(f) for f in (r["fix_files"] or [])}, min_n=3)

# ---------------------------------------------------------------- helpers -----
def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
    ax.tick_params(length=0)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def titles(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


# ---------------------------------------------------------------- charts ------
def draw_hbar(ax, data, title, sub):
    labels = [k for k, _ in data][::-1]
    vals = [v for _, v in data][::-1]
    y = range(len(labels))
    ax.barh(y, vals, color=S1, height=0.66, zorder=3)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10, color=INK2)
    ax.set_xlim(0, max(vals) * 1.12)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, str(v), va="center", ha="left",
                fontsize=9.5, color=INK2)
    style(ax, "x"); ax.set_xlabel("tasks", fontsize=9, color=MUT)
    titles(ax, title, sub)


def draw_vbar(ax, data, title, sub, xlabel=""):
    labels = [str(k) for k, _ in data]
    vals = [v for _, v in data]
    x = range(len(labels))
    ax.bar(x, vals, color=S1, width=0.62, zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10, color=INK2)
    ax.set_ylim(0, max(vals) * 1.15)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, str(v), ha="center", va="bottom",
                fontsize=9.5, color=INK2)
    style(ax, "y"); ax.set_ylabel("tasks", fontsize=9, color=MUT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUT)
    titles(ax, title, sub)


def draw_rate(ax, data, title, sub):
    labels = [k for k, _, _ in data][::-1]
    fr = [(p / t if t else 0, p, t) for _, p, t in data][::-1]
    y = range(len(labels))
    for yi, (f, p, t) in zip(y, fr):
        ax.barh(yi, 1.0, color=REM, height=0.6, zorder=2)
        ax.barh(yi, f, color=S1, height=0.6, zorder=3,
                edgecolor=SURF, linewidth=1.5 if 0 < f < 1 else 0)
        ax.text(1.03, yi, f"{f*100:.0f}%", va="center", ha="left",
                fontsize=10, color=INK, fontweight="bold")
        ax.text(1.16, yi, f"{p}/{t}", va="center", ha="left", fontsize=9, color=MUT)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10, color=INK2)
    # xlim runs past 100% so the % / count labels sit INSIDE the axes box (no bleed
    # into a neighbouring subplot); the axis line itself stops at 100%.
    ax.set_xlim(0, 1.34); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.spines["bottom"].set_bounds(0, 1.0)
    ax.tick_params(length=0)
    ax.grid(True, axis="x", color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    titles(ax, title, sub)


CHARTS = [
    ("fix_location", lambda ax: draw_hbar(ax, [(short(k), v) for k, v in fix_area.most_common(11)],
        "Where the fix lands", "Fix files by area (under torch/ unless noted)")),
    ("module_labels", lambda ax: draw_hbar(ax, [(k.replace("module: ", ""), v) for k, v in mod_label.most_common(12)],
        "Module labels", "GitHub 'module:' labels on the source issue")),
    ("patch_size", lambda ax: draw_vbar(ax, [(b, size_hist[b]) for b in SIZE_ORDER],
        "Patch size distribution", "Lines changed in the gold patch", "LOC changed")),
    ("f2p_count", lambda ax: draw_vbar(ax, [(k, f2p_hist[k]) for k in sorted(f2p_hist)],
        "FAIL_TO_PASS test count", "Tests each task must flip fail→pass", "# FAIL_TO_PASS tests")),
    ("rate_by_size", lambda ax: draw_rate(ax, rate_size,
        "Pass rate by patch size", "Smaller patches solved far more often (blind opus-4.8)")),
    ("rate_by_f2p", lambda ax: draw_rate(ax, rate_f2p,
        "Pass rate by FAIL_TO_PASS count", "3+ tests to flip is markedly harder")),
    ("rate_by_subsystem", lambda ax: draw_rate(ax, [(short(k), p, t) for k, p, t in rate_area],
        "Pass rate by subsystem", "Solved / total per fix area (n≥3, under torch/)")),
]

# ---------------------------------------------------------------- render ------
for name, fn in CHARTS:
    h = {"fix_location": 4.3, "module_labels": 4.6, "rate_by_subsystem": 4.6}.get(name, 3.6)
    fig, ax = plt.subplots(figsize=(7.6, h))
    fn(ax)
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, config.REPO))

# combined dashboard
fig = plt.figure(figsize=(15.5, 17.5), facecolor=SURF)
gs = fig.add_gridspec(4, 2, hspace=0.72, wspace=0.34,
                      left=0.11, right=0.965, top=0.90, bottom=0.035)
fig.suptitle("pt-agent-bench · corpus distribution", x=0.11, y=0.972, ha="left",
             fontsize=20, fontweight="bold")
solved = sum(1 for _, d in graded if d["resolved"])
fig.text(0.11, 0.948, f"{N} execution-verified pytorch/pytorch tasks   ·   blind pass "
         f"{solved}/{len(graded)} = {100*solved/len(graded):.0f}%   ·   opus-4.8 (xhigh)",
         ha="left", fontsize=12, color=INK2)
# first 6 charts in a 3×2 grid; the wide subsystem chart spans the full bottom row
for i, (name, fn) in enumerate(CHARTS[:6]):
    fn(fig.add_subplot(gs[i // 2, i % 2]))
CHARTS[6][1](fig.add_subplot(gs[3, :]))
p = os.path.join(FIGDIR, "dashboard.png")
fig.savefig(p, dpi=150, facecolor=SURF)
plt.close(fig)
print("wrote", os.path.relpath(p, config.REPO))
