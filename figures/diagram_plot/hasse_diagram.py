#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:05:10 2026

@author: andrew

b4_bidirectional_hasse.py
---------------------------------------------------------------------------
The Boolean lattice 2^{[4]} (16 binary vectors, ordered by inclusion), drawn
in four styles, with every cover rendered as TWO arcs carrying the asymmetric
Hamming loss

     0 -> 1  (add a 1, false positive):   cost  alpha
     1 -> 0  (drop a 1, false negative):  cost  2 - alpha

alpha = 1 recovers ordinary Hamming distance (and the classical pictures).
Requires numpy + matplotlib only.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

P = 4
NODES  = [tuple(b) for b in product((0, 1), repeat=P)]
COVERS = [(u, v) for u in NODES for v in NODES
         if sum(v) == sum(u) + 1 and all(a <= b for a, b in zip(u, v))]  # 32

lab     = lambda v: "".join(str(b) for b in v)
rank    = lambda v: sum(v)
flipped = lambda u, v: next(i for i in range(P) if u[i] != v[i])


# ---------------------------------------------------------------------------
# 1. The loss
# ---------------------------------------------------------------------------
def alpha_vec(alpha) -> np.ndarray:
   """Scalar alpha, or one alpha per coordinate."""
   a = np.atleast_1d(np.asarray(alpha, dtype=float))
   a = np.repeat(a, P) if a.size == 1 else a
   if a.size != P:
       raise ValueError(f"alpha must be scalar or length {P}")
   if np.any(a <= 0) or np.any(a >= 2):
       raise ValueError("need 0 < alpha < 2 so both directions cost > 0")
   return a


def loss(u, v, alpha) -> float:
   """Generalised Hamming loss of predicting v when the truth is u."""
   a = alpha_vec(alpha)
   u, v = np.asarray(u), np.asarray(v)
   return float(a[(u == 0) & (v == 1)].sum() + (2 - a)[(u == 1) & (v == 0)].sum())


def coord_weights(alpha, sym: str = "max") -> np.ndarray:
   """
   Per-coordinate geometric edge length, normalised to mean 1.
   With a SCALAR alpha this is all-ones for every choice of `sym`
   (because alpha + (2-alpha) = 2); it only bites for a vector alpha.
   """
   a = alpha_vec(alpha)
   g = {"mean": 0.5 * (a + (2 - a)),
        "max":  np.maximum(a, 2 - a),
        "geom": np.sqrt(a * (2 - a)),
        "up":   a,
        "down": 2 - a}[sym]
   return g / g.mean()


# ---------------------------------------------------------------------------
# 2. The four layouts  (y is strictly increasing along every cover)
# ---------------------------------------------------------------------------
def _linear(xoff, alpha=1.0, h=1.6, sym="max", spread=1.0) -> dict:
   """pos(v) = sum_i v_i e_i  with e_i = g_i * (xoff_i, h)."""
   g = coord_weights(alpha, sym)
   e = [g[i] * np.array([xoff[i], h]) for i in range(P)]
   return {v: tuple(spread * sum((v[i] * e[i] for i in range(P)), np.zeros(2)))
           for v in NODES}


def layout_levels(alpha=1.0, h=1.6, dx=1.75, spread=1.0, key=lab, **kw) -> dict:
   """Graded drawing: one row per cardinality (the crossing-heavy classic)."""
   pos = {}
   for k in range(P + 1):
       row = sorted((v for v in NODES if rank(v) == k), key=key)
       for i, v in enumerate(row):
           pos[v] = (spread * dx * (i - (len(row) - 1) / 2), spread * h * k)
   return pos


def layout_matrix(alpha=1.0, s=3.0, **kw) -> dict:
   """B4 = B2 x B2: two nested diamonds -> the 4x4 'matrix' picture."""
   return _linear((-1.0, 1.0, -s, s), alpha=alpha, **kw)


def layout_projection(alpha=1.0, **kw) -> dict:
   """Skew linear projection of the tesseract (Sidon offsets: no collisions)."""
   return _linear((-3.5, -1.5, 0.5, 4.5), alpha=alpha, **kw)


def layout_nested(alpha=1.0, h=1.6, scale=2.0, lift=1.3, spread=1.0,
                 sym="max", **kw) -> dict:
   """Cube-in-cube: the b4=0 cube small, the b4=1 copy enlarged and lifted."""
   g = coord_weights(alpha, sym)
   e = [g[i] * np.array([x, h]) for i, x in enumerate((-1.2, 0.2, 1.4))]
   pos = {}
   for v in NODES:
       p = sum((v[i] * e[i] for i in range(3)), np.zeros(2))
       if v[3]:
           p = scale * p + np.array([0.0, g[3] * lift * h])
       pos[v] = tuple(spread * p)
   return pos


LAYOUTS = {"levels (graded)":      layout_levels,
          "matrix (B2 x B2)":     layout_matrix,
          "skew projection":      layout_projection,
          "nested (cube-in-cube)": layout_nested}


# ---------------------------------------------------------------------------
# 3. Arc geometry: choose curvature so that arc length is proportional to cost
# ---------------------------------------------------------------------------
def _bezier(p0, p2, rad, n=80):
   p0, p2 = np.asarray(p0, float), np.asarray(p2, float)
   d = p2 - p0
   c = 0.5 * (p0 + p2) + rad * np.array([d[1], -d[0]])   # matplotlib's arc3
   t = np.linspace(0.0, 1.0, n)[:, None]
   return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2


def _arclen(p0, p2, rad):
   q = _bezier(p0, p2, rad)
   return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def rad_for_length(p0, p2, target, rad_max=3.0):
   """Curvature giving a Bezier of the requested length (monotone -> bisect)."""
   chord = float(np.linalg.norm(np.asarray(p2) - np.asarray(p0)))
   if target <= chord:
       return 0.0
   if _arclen(p0, p2, rad_max) < target:
       return rad_max                       # clipped: ratio too extreme to draw
   lo, hi = 0.0, rad_max
   for _ in range(50):
       mid = 0.5 * (lo + hi)
       lo, hi = (mid, hi) if _arclen(p0, p2, mid) < target else (lo, mid)
   return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 4. Drawing
# ---------------------------------------------------------------------------
UP_C, DN_C = "#1f6fb4", "#c0392b"


def draw_hasse(pos, alpha, ax=None, title="", eps=0.07, gamma=1.0,
              arc_mode="length", rad_fixed=0.14, rad_max=3.0,
              edge_labels="auto", node_fs=9, legend=True):
   """
   arc_mode = "length": arc lengths in the ratio  alpha : (2 - alpha)
            = "fixed" : equal gentle bows; cost shown by width/colour only
   gamma    : temper the length ratio, (c_hi/c_lo)**gamma  (1 = faithful)
   """
   a = alpha_vec(alpha)
   if ax is None:
       _, ax = plt.subplots(figsize=(8.5, 8))
   varies = not np.allclose(a, a[0])
   show_lbl = varies if edge_labels == "auto" else bool(edge_labels)
   cmax = max(a.max(), (2 - a).max())

   for u, v in COVERS:
       i = flipped(u, v)
       pu, pv = np.asarray(pos[u], float), np.asarray(pos[v], float)
       d = float(np.linalg.norm(pv - pu))
       c_up, c_dn = a[i], 2 - a[i]
       c_lo = min(c_up, c_dn)

       for src, dst, c, col in ((pu, pv, c_up, UP_C), (pv, pu, c_dn, DN_C)):
           if arc_mode == "length":
               r = rad_for_length(src, dst, d * (1 + eps) * (c / c_lo) ** gamma,
                                  rad_max)
               r = max(r, rad_fixed * 0.45)         # keep the two lanes apart
           else:
               r = rad_fixed
           ax.add_patch(FancyArrowPatch(
               src, dst, connectionstyle=f"arc3,rad={r}",
               arrowstyle="-|>", mutation_scale=12,
               linewidth=0.7 + 1.9 * c / cmax, color=col, alpha=0.9,
               shrinkA=17, shrinkB=17, zorder=1))
           if show_lbl:
               m = 0.5 * (src + dst)
               e = dst - src
               q = m + 0.5 * r * np.array([e[1], -e[0]])
               ax.text(*q, f"{c:.2f}", color=col, fontsize=node_fs - 2,
                       ha="center", va="center", zorder=3,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                 ec="none", alpha=0.85))

   for v in NODES:
       ax.text(*pos[v], lab(v), fontsize=node_fs, ha="center", va="center",
               zorder=4, family="monospace",
               bbox=dict(boxstyle="round,pad=0.28", fc="white",
                         ec="0.25", lw=0.9))

   if legend and not varies:
       ax.legend(handles=[
           Line2D([], [], color=UP_C, lw=2.4,
                  label=rf"add a 1   $0\to1$:  $\alpha$ = {a[0]:.2f}"),
           Line2D([], [], color=DN_C, lw=2.4,
                  label=rf"drop a 1  $1\to0$:  $2-\alpha$ = {2 - a[0]:.2f}")],
           loc="upper left", fontsize=8, framealpha=0.9)

   ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.14)
   if title:
       ax.set_title(title, fontsize=11)
   return ax


def min_node_gap(pos) -> float:
   q = np.array([pos[v] for v in NODES])
   dd = np.linalg.norm(q[:, None] - q[None], axis=-1) + np.eye(len(q)) * 1e9
   return float(dd.min())


# ---------------------------------------------------------------------------
# 5. All four diagrams at once
# ---------------------------------------------------------------------------
def figure_all(alpha=0.6, sym="max", arc_mode="length", gamma=1.0, save=None):
   a = alpha_vec(alpha)
   spread = 1.0 + 0.7 * float(np.abs(a - 1).max())      # room for the bows
   fig, axes = plt.subplots(2, 2, figsize=(16, 15))
   for ax, (name, fn) in zip(axes.ravel(), LAYOUTS.items()):
       pos = fn(alpha=alpha, sym=sym, spread=spread)
       draw_hasse(pos, alpha, ax=ax, title=name, arc_mode=arc_mode, gamma=gamma)
   txt = (rf"$2^{{[4]}}$ ordered by inclusion — bidirectional Hasse diagrams,"
          rf"  $\alpha$ = {a[0]:.2f},  $2-\alpha$ = {2 - a[0]:.2f}"
          if np.allclose(a, a[0]) else
          rf"$2^{{[4]}}$, per-coordinate $\alpha$ = {np.round(a, 2)}")
   fig.suptitle(txt + "   (arc length $\\propto$ directional cost)", fontsize=13)
   fig.tight_layout(rect=(0, 0, 1, 0.97))
   if save:
       fig.savefig(save, dpi=200, bbox_inches="tight")
   return fig


if __name__ == "__main__":
   figure_all(alpha=1.0)    # symmetric: the classical Wikipedia pictures
   figure_all(alpha=0.6)    # dropping a 1 is ~2.3x dearer than adding one
   figure_all(alpha=1.5)    # the other way round
   # per-coordinate weights: now the *node positions* move as well
   figure_all(alpha=[0.5, 0.9, 1.2, 1.6])
   plt.show()
