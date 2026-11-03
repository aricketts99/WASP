#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b4_updown_hasse.py

2^[4] ordered by inclusion.  For each of four layouts, TWO Hasse diagrams:

    left   all arrows UP     u -> v, u covered by v,  H_alpha(u, v) = alpha
    right  all arrows DOWN   v -> u,                  H_alpha(v, u) = 2 - alpha

Edges curve only where a straight line would collide with a third node
(as in the Wikipedia 4x4-matrix tesseract).  Arrows terminate exactly on the
node discs.
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

UP_C, DN_C = "#1a5fa8", "#b23026"
NODE_FC, NODE_EC = "#ffffff", "#222222"
NODE_R = 3.0          # node disc radius, points
LW = 0.9              # edge line width
HEAD = 7.5            # arrow head scale
FS = 7.5              # label font size


def H(g1, g2, alpha):
    """Generalised Hamming loss H_alpha(gamma_1, gamma_2)."""
    return sum(alpha if (g1[i] == 0 and g2[i] == 1) else
               (2 - alpha) if (g1[i] == 1 and g2[i] == 0) else 0.0
               for i in range(P))


# ------------------------------------------------------------------ layouts
def _separate(xd, gap=1.25):
    out = dict(xd)
    for k in range(P + 1):
        row = sorted((v for v in NODES if rank(v) == k),
                     key=lambda v: (xd[v], lab(v)))
        xs = [xd[v] for v in row]
        for i in range(1, len(xs)):
            xs[i] = max(xs[i], xs[i - 1] + gap)
        shift = 0.5 * (xs[0] + xs[-1]) - 0.5 * (xd[row[0]] + xd[row[-1]])
        for v, x in zip(row, xs):
            out[v] = x - shift
    return out


def x_levels(dx=1.7, sweeps=24):
    order = {k: sorted((v for v in NODES if rank(v) == k), key=lab)
             for k in range(P + 1)}
    x = {}

    def place():
        for k, row in order.items():
            for i, v in enumerate(row):
                x[v] = dx * (i - (len(row) - 1) / 2)

    place()
    for s in range(sweeps):
        for k in (range(P + 1) if s % 2 == 0 else range(P, -1, -1)):
            key = {v: float(np.mean([x[w] for w in NBRS[v]])) for v in order[k]}
            order[k] = sorted(order[k], key=lambda v: (key[v], lab(v)))
            place()
    return x


_off = lambda o: {v: float(sum(v[i] * o[i] for i in range(P))) for v in NODES}
x_matrix = lambda: _off((-1.0, 1.0, -3.0, 3.0))      # B2 x B2, the 4x4 matrix
x_projection = lambda: _off((-3.5, -1.5, 0.5, 4.5))  # skew tesseract
x_nested = lambda s=2.6: {v: s ** v[3] * (-1.15 * v[0] + 0.15 * v[1]
                                          + 1.45 * v[2]) for v in NODES}

LAYOUTS = {"levels (graded)": x_levels,
           "matrix (B2 x B2)": x_matrix,
           "skew projection": x_projection,
           "nested (cube in cube)": x_nested}

# the matrix layout needs no x-separation pass: its grid is already regular
NO_SEPARATE = {"matrix (B2 x B2)"}


# -------------------------------------------------- curvature only where needed
def _bezier(p0, p2, rad, n=64):
    p0, p2 = np.asarray(p0, float), np.asarray(p2, float)
    d = p2 - p0
    ctrl = 0.5 * (p0 + p2) + rad * np.array([d[1], -d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p2


def _clearance(curve, others):
    if len(others) == 0:
        return np.inf
    return float(np.linalg.norm(curve[:, None, :] - others[None, :, :],
                                axis=-1).min())


def solve_curvatures(pos, margin=0.30, rad_max=0.45, steps=9):
    """rad for each cover (u, v): 0 if the straight line is clear, else the
    smallest bow that keeps every other node at distance >= margin."""
    pts = {v: np.asarray(pos[v], float) for v in NODES}
    rads, mags = {}, np.linspace(0.08, rad_max, steps)
    for u, v in COVERS:
        others = np.array([pts[w] for w in NODES if w not in (u, v)])
        if _clearance(_bezier(pts[u], pts[v], 0.0), others) >= margin:
            rads[(u, v)] = 0.0
            continue
        best = (rad_max, -np.inf)
        for m in mags:
            trials = [(r, _clearance(_bezier(pts[u], pts[v], r), others))
                      for r in (m, -m)]
            trials.sort(key=lambda t: -t[1])
            if trials[0][1] >= margin:
                best = trials[0]
                break
            if trials[0][1] > best[1]:
                best = trials[0]
        rads[(u, v)] = float(best[0])
    return rads


def _label_angle(v, pos):
    """Widest angular gap between incident edges: where the label goes."""
    a = sorted(float(np.degrees(np.arctan2(pos[w][1] - pos[v][1],
                                           pos[w][0] - pos[v][0]))) % 360
               for w in NBRS[v])
    if not a:
        return 270.0
    gaps = [(a[(i + 1) % len(a)] - a[i]) % 360 for i in range(len(a))]
    i = int(np.argmax(gaps))
    return (a[i] + gaps[i] / 2.0) % 360


# ------------------------------------------------------------------ one panel
def panel(ax, xd, alpha, direction, unit=1.0, xlim=None, ylim=None,
          margin=0.30, ruler=True, label_pad=11.0):
    cost = alpha if direction == "up" else 2 - alpha
    h = cost * unit
    col = UP_C if direction == "up" else DN_C
    pos = {v: (xd[v], h * rank(v)) for v in NODES}
    rads = solve_curvatures(pos, margin=margin)

    for u, v in COVERS:
        r = rads[(u, v)]
        # arc3 rad is orientation dependent: negate it when the arrow reverses
        src, dst, rr = ((pos[u], pos[v], r) if direction == "up"
                        else (pos[v], pos[u], -r))
        ax.add_patch(FancyArrowPatch(
            src, dst,
            connectionstyle="arc3,rad={:.4f}".format(rr),
            arrowstyle="-|>", mutation_scale=HEAD, linewidth=LW,
            color=col, joinstyle="round", capstyle="round",
            shrinkA=NODE_R, shrinkB=NODE_R, zorder=2))

    xs = [pos[v][0] for v in NODES]
    ys = [pos[v][1] for v in NODES]
    ax.plot(xs, ys, linestyle="none", marker="o", markersize=2 * NODE_R,
            markerfacecolor=NODE_FC, markeredgecolor=NODE_EC,
            markeredgewidth=0.9, zorder=3)

    for v in NODES:
        th = np.radians(_label_angle(v, pos))
        ax.annotate(lab(v), pos[v],
                    textcoords="offset points",
                    xytext=(label_pad * np.cos(th), label_pad * np.sin(th)),
                    ha="center", va="center", fontsize=FS, family="monospace",
                    color="#111111", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])

    if ruler:
        x0 = (xlim[0] if xlim else min(xd.values()) - 1.8) + 0.3
        ax.plot([x0, x0], [0.0, h * P], color="0.65", lw=0.7, zorder=0)
        for k in range(P + 1):
            val = cost * k if direction == "up" else cost * (P - k)
            ax.plot([x0 - 0.12, x0 + 0.12], [h * k, h * k],
                    color="0.65", lw=0.7, zorder=0)
            ax.text(x0 - 0.24, h * k, "{:g}".format(val), fontsize=FS - 1,
                    ha="right", va="center", color="0.4")

    head = ("all arrows UP:  0 " + ARROW + " 1, add a 1"
            if direction == "up" else
            "all arrows DOWN:  1 " + ARROW + " 0, drop a 1")
    rung = ("rung = {} = {:g}".format(ALPHA, cost) if direction == "up"
            else "rung = 2 - {} = {:g}".format(ALPHA, cost))
    span = ("{} to 1111 costs {:g}".format(EMPTY, cost * P)
            if direction == "up" else
            "1111 to {} costs {:g}".format(EMPTY, cost * P))
    ax.set_title(head + NL + rung + ",   " + span, fontsize=9.5, color=col,
                 pad=8)

    ax.set_aspect("equal")
    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return pos, rads


# --------------------------------------------- one figure = one layout, 2 panels
def figure_for_layout(name, alpha=0.6, unit=1.0, margin=0.30, save=None):
    xd = LAYOUTS[name]()
    if name not in NO_SEPARATE:
        xd = _separate(xd)
    xs = list(xd.values())
    xlim = (min(xs) - 2.0, max(xs) + 1.1)
    ylim = (-1.0, max(alpha, 2 - alpha) * unit * P + 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    _, r_up = panel(axes[0], xd, alpha, "up", unit, xlim, ylim, margin)
    _, r_dn = panel(axes[1], xd, alpha, "down", unit, xlim, ylim, margin)
    curved = sum(abs(r) > 1e-9 for r in r_up.values())
    fig.suptitle(
        "2^[4] ordered by inclusion, layout: " + name
        + "     {} = {:g},  2 - {} = {:g},  ratio {:.2f} to 1".format(
            ALPHA, alpha, ALPHA, 2 - alpha,
            max(alpha, 2 - alpha) / min(alpha, 2 - alpha))
        + NL + "vertical distance = accumulated loss;  "
        + "{} of {} edges curved to clear intervening nodes".format(
            curved, len(COVERS)),
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    if save:
        fig.savefig(save, dpi=220, bbox_inches="tight")
    return fig


def make_all(alpha=0.6, unit=1.0, margin=0.30, save_prefix=None):
    figs = []
    for name in LAYOUTS:
        fn = None
        if save_prefix:
            fn = "{}_{}_alpha{:g}.png".format(save_prefix, name.split()[0],
                                              alpha)
        figs.append(figure_for_layout(name, alpha, unit, margin, save=fn))
    return figs


if __name__ == "__main__":
    make_all(alpha=0.6)
    # make_all(alpha=1.0)                        # symmetric check
    # make_all(alpha=0.6, save_prefix="hasse")   # write 4 PNGs
    plt.show()
