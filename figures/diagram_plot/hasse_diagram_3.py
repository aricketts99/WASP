#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:36:45 2026

@author: andrew

b4_matrix_alpha_compare.py

2^[4] by inclusion, matrix (B2 x B2) layout only.  Five panels:

    symmetric   alpha = 1.0   undirected, rung 1.0
    alpha = 0.6  up (rung 0.6)   and  down (rung 1.4)
    alpha = 1.4  up (rung 1.4)   and  down (rung 0.6)

H_alpha(g1, g2) = sum_i [ alpha 1(g1_i=0, g2_i=1) + (2-alpha) 1(g1_i=1, g2_i=0) ]

All panels share xlim, ylim and aspect, so heights are directly comparable.
Edges curve only where a straight line would foul a third node; arrows and
lines terminate exactly on the node discs.
"""
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

NL, ARROW, ALPHA, EMPTY = chr(10), chr(8594), chr(945), chr(8709)

P = 4
NODES = [tuple(b) for b in product((0, 1), repeat=P)]
COVERS = [(u, v) for u in NODES for v in NODES
          if sum(v) == sum(u) + 1 and all(a <= b for a, b in zip(u, v))]
NBRS = {v: [] for v in NODES}
for _u, _v in COVERS:
    NBRS[_u].append(_v)
    NBRS[_v].append(_u)

lab = lambda v: "".join(str(b) for b in v)
rank = lambda v: sum(v)

UP_C, DN_C, SY_C = "#1a5fa8", "#b23026", "#333333"
NODE_R, LW, HEAD, FS = 3.0, 0.9, 7.5, 7.5


def H(g1, g2, alpha):
    """Generalised Hamming loss H_alpha(gamma_1, gamma_2)."""
    return sum(alpha if (g1[i] == 0 and g2[i] == 1) else
               (2 - alpha) if (g1[i] == 1 and g2[i] == 0) else 0.0
               for i in range(P))


# ------------------------------------------------------- matrix layout, x only
MATRIX_OFF = (-1.0, 1.0, -3.0, 3.0)
# o1+o2 = o3+o4 = 0 puts 1100 and 0011 on top of each other at level 2;
# separate just that pair and leave the rest on the regular grid
TWEAK = {(1, 1, 0, 0): -0.9, (0, 0, 1, 1): 0.9}


def x_matrix():
    xd = {v: float(sum(v[i] * MATRIX_OFF[i] for i in range(P))) for v in NODES}
    for v, dx in TWEAK.items():
        xd[v] += dx
    for k in range(P + 1):                      # assert distinctness per level
        xs = sorted(xd[v] for v in NODES if rank(v) == k)
        assert all(b - a > 1e-6 for a, b in zip(xs, xs[1:])), "collision"
    return xd


# -------------------------------------------------- curvature only where needed
def _bezier(p0, p2, rad, n=64):
    p0, p2 = np.asarray(p0, float), np.asarray(p2, float)
    d = p2 - p0
    ctrl = 0.5 * (p0 + p2) + rad * np.array([d[1], -d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2


def _clear(curve, others):
    return float(np.linalg.norm(curve[:, None, :] - others[None, :, :],
                                axis=-1).min())


def solve_curvatures(pos, margin=0.28, rad_max=0.45, steps=9):
    pts = {v: np.asarray(pos[v], float) for v in NODES}
    rads, mags = {}, np.linspace(0.08, rad_max, steps)
    for u, v in COVERS:
        others = np.array([pts[w] for w in NODES if w not in (u, v)])
        if _clear(_bezier(pts[u], pts[v], 0.0), others) >= margin:
            rads[(u, v)] = 0.0
            continue
        best = (rad_max, -np.inf)
        for m in mags:
            trials = sorted(((r, _clear(_bezier(pts[u], pts[v], r), others))
                             for r in (m, -m)), key=lambda t: -t[1])
            if trials[0][1] >= margin:
                best = trials[0]
                break
            if trials[0][1] > best[1]:
                best = trials[0]
        rads[(u, v)] = float(best[0])
    return rads


def _label_angle(v, pos):
    a = sorted(float(np.degrees(np.arctan2(pos[w][1] - pos[v][1],
                                           pos[w][0] - pos[v][0]))) % 360
               for w in NBRS[v])
    gaps = [(a[(i + 1) % len(a)] - a[i]) % 360 for i in range(len(a))]
    i = int(np.argmax(gaps))
    return (a[i] + gaps[i] / 2.0) % 360


# ----------------------------------------------------------------- one panel
def panel(ax, xd, rung, mode, xlim, ylim, margin=0.28, label_pad=11.0,
          ruler=True, title=None, colour=None):
    """mode: 'up' (arrows u -> v), 'down' (arrows v -> u), 'sym' (plain lines)."""
    col = colour or {"up": UP_C, "down": DN_C, "sym": SY_C}[mode]
    pos = {v: (xd[v], rung * rank(v)) for v in NODES}
    rads = solve_curvatures(pos, margin=margin)

    for u, v in COVERS:
        r = rads[(u, v)]
        if mode == "down":
            src, dst, rr = pos[v], pos[u], -r    # arc3 rad flips with direction
        else:
            src, dst, rr = pos[u], pos[v], r
        ax.add_patch(FancyArrowPatch(
            src, dst, connectionstyle="arc3,rad={:.4f}".format(rr),
            arrowstyle=("-" if mode == "sym" else "-|>"),
            mutation_scale=HEAD, linewidth=LW, color=col,
            joinstyle="round", capstyle="round",
            shrinkA=NODE_R, shrinkB=NODE_R, zorder=2))

    ax.plot([pos[v][0] for v in NODES], [pos[v][1] for v in NODES],
            linestyle="none", marker="o", markersize=2 * NODE_R,
            markerfacecolor="white", markeredgecolor="#222222",
            markeredgewidth=0.9, zorder=3)

    for v in NODES:
        th = np.radians(_label_angle(v, pos))
        ax.annotate(lab(v), pos[v], textcoords="offset points",
                    xytext=(label_pad * np.cos(th), label_pad * np.sin(th)),
                    ha="center", va="center", fontsize=FS, family="monospace",
                    color="#111111", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])

    if ruler:
        x0 = xlim[0] + 0.3
        ax.plot([x0, x0], [0.0, rung * P], color="0.65", lw=0.7, zorder=0)
        for k in range(P + 1):
            val = rung * (P - k) if mode == "down" else rung * k
            ax.plot([x0 - 0.12, x0 + 0.12], [rung * k] * 2,
                    color="0.65", lw=0.7, zorder=0)
            ax.text(x0 - 0.24, rung * k, "{:g}".format(val), fontsize=FS - 1,
                    ha="right", va="center", color="0.4")

    if title:
        ax.set_title(title, fontsize=9.5, color=col, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return pos


def _titles(alpha, mode):
    rung = alpha if mode == "up" else 2 - alpha
    if mode == "sym":
        return ("symmetric, {} = 1:  undirected".format(ALPHA) + NL
                + "rung = 1,  full chain costs 4 either way")
    if mode == "up":
        return ("{} = {:g},  all arrows UP  (0 {} 1, add a 1)".format(
                    ALPHA, alpha, ARROW) + NL
                + "rung = {} = {:g},  {} to 1111 costs {:g}".format(
                    ALPHA, rung, EMPTY, rung * P))
    return ("{} = {:g},  all arrows DOWN  (1 {} 0, drop a 1)".format(
                ALPHA, alpha, ARROW) + NL
            + "rung = 2 - {} = {:g},  1111 to {} costs {:g}".format(
                ALPHA, rung, EMPTY, rung * P))


# ------------------------------------------------------------------- figures
def _frame(xd, rung_max=1.4, pad_lo=2.0, pad_hi=1.1):
    xs = list(xd.values())
    return ((min(xs) - pad_lo, max(xs) + pad_hi),
            (-0.9, rung_max * P + 0.9))


def figure_overview(alphas=(0.6, 1.4), margin=0.28, save=None):
    """All five panels on one canvas, common scale."""
    xd = x_matrix()
    rung_max = max([1.0] + [max(a, 2 - a) for a in alphas])
    xlim, ylim = _frame(xd, rung_max)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    panel(axes[0, 0], xd, 1.0, "sym", xlim, ylim, margin,
          title=_titles(1.0, "sym"))
    for row, a in zip((0, 1), alphas):
        panel(axes[row, 1], xd, a, "up", xlim, ylim, margin,
              title=_titles(a, "up"))
        panel(axes[row, 2], xd, 2 - a, "down", xlim, ylim, margin,
              title=_titles(a, "down"))
    # note: panel(..., 2 - a, "down") draws rung 2 - a, matching H(v, u)

    axes[1, 0].axis("off")
    note = (
        "H_{}(g1, g2) = {} x (0{}1 flips) + (2-{}) x (1{}0 flips)".format(
            ALPHA, ALPHA, ARROW, ALPHA, ARROW) + NL + NL
        + "Vertical distance = accumulated loss." + NL
        + "The loss is graded on comparable pairs," + NL
        + "so this is exact, not a fit." + NL + NL
        + "Because 2 - 0.6 = 1.4, the two asymmetric" + NL
        + "cases are mirror images: the {} = 1.4 UP".format(ALPHA) + NL
        + "panel is congruent to the {} = 0.6 DOWN".format(ALPHA) + NL
        + "panel.  Only three geometries exist here" + NL
        + "(rungs 0.6, 1.0, 1.4); what changes is" + NL
        + "which direction is the expensive one.")
    axes[1, 0].text(0.02, 0.96, note, transform=axes[1, 0].transAxes,
                    fontsize=9, va="top", ha="left", family="monospace",
                    color="#333333")

    fig.suptitle("2^[4] ordered by inclusion, matrix (B2 x B2) layout:  "
                 "symmetric versus {} = {:g} and {} = {:g}".format(
                     ALPHA, alphas[0], ALPHA, alphas[1]), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
    return fig


def figure_symmetric(margin=0.28, save=None):
    xd = x_matrix()
    xlim, ylim = _frame(xd)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    panel(ax, xd, 1.0, "sym", xlim, ylim, margin, title=_titles(1.0, "sym"))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
    return fig


def figure_pair(alpha, margin=0.28, save=None):
    xd = x_matrix()
    xlim, ylim = _frame(xd)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    panel(axes[0], xd, alpha, "up", xlim, ylim, margin,
          title=_titles(alpha, "up"))
    panel(axes[1], xd, 2 - alpha, "down", xlim, ylim, margin,
          title=_titles(alpha, "down"))
    fig.suptitle("matrix layout, {} = {:g}   (up rung {:g}, down rung {:g}, "
                 "ratio {:.2f} to 1)".format(ALPHA, alpha, alpha, 2 - alpha,
                                             max(alpha, 2 - alpha)
                                             / min(alpha, 2 - alpha)),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
    return fig


def report(alphas=(0.6, 1.0, 1.4)):
    print("nodes {}, covers {}".format(len(NODES), len(COVERS)))
    lo, hi = (0, 0, 0, 0), (1, 1, 1, 1)
    for a in alphas:
        print("  alpha {:.1f}:  up rung {:.1f}  down rung {:.1f}   "
              "H(0000,1111) = {:.1f}   H(1111,0000) = {:.1f}".format(
                  a, a, 2 - a, H(lo, hi, a), H(hi, lo, a)))


if __name__ == "__main__":
    report()
    figure_overview()          # the five-panel comparison
    # figure_symmetric()
    # figure_pair(0.6)
    # figure_pair(1.4)
    # figure_overview(save="matrix_alpha_compare.png")
    plt.show()
