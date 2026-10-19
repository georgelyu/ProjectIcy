#!/usr/bin/env python3
"""Reproduce the vector method figures and their analytic stencil data.

The two accuracy controls isolate missing-direction normalization and
cross-solid contamination. They are manufactured local fields, not solver
benchmarks. Run verify_gradient_figure.py to check the actual Taichi kernels.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures"
PREVIEW = PAPER / "build" / "figure-preview"
# Author at Nature's 89 mm single-column width; embed at ACM column width.
# 7 pt ordinary text, 8 pt panel letters, editable embedded TrueType text.
WIDTH = 89 / 25.4
INK = "#171717"
TEAL = "#0072B2"
ORANGE = "#D55E00"
ICE = "#E6E9EB"
ICE_EDGE = "#9CA4AA"
GRID = "#E3E7E9"
PHI0 = 0.4
EXACT = np.array([0.12, 0.08])
DIRECTIONS = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
])
WEIGHTS = np.array([1 / 9] * 4 + [1 / 36] * 4)
HALF_MASK = DIRECTIONS[:, 0] <= 0
CORNER_MASK = DIRECTIONS.sum(axis=1) <= 0
ALL_FLUID_MASK = CORNER_MASK | np.all(DIRECTIONS == [1, 1], axis=1)


def normal_equations(mask, values):
    c, w = DIRECTIONS[mask], WEIGHTS[mask]
    return (np.einsum("i,ij,ik->jk", w, c, c),
            np.einsum("i,ij,i->j", w, c, values[mask] - PHI0))


def analytic_cases():
    values = PHI0 + DIRECTIONS @ EXACT
    a, b = normal_equations(HALF_MASK, values)
    contaminated = values.copy()
    contaminated[4] = 0.9
    ac, bc = normal_equations(CORNER_MASK, contaminated)
    au, bu = normal_equations(ALL_FLUID_MASK, contaminated)
    records = []
    for name, mask, matrix, rhs, baseline, control in [
        ("truncated", HALF_MASK, a, b, 3 * b, "3 b on accepted donors"),
        ("separated", CORNER_MASK, ac, bc, np.linalg.solve(au, bu),
         "weighted linear fit to all fluid neighbors, without connectivity"),
    ]:
        fitted = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(fitted, EXACT, atol=1e-15, rtol=0)
        records.append({
            "case": name,
            "accepted_offsets": DIRECTIONS[mask].tolist(),
            "normal_matrix": matrix.tolist(),
            "right_hand_side": rhs.tolist(),
            "connected_gradient": fitted.tolist(),
            "control": control,
            "control_gradient": baseline.tolist(),
            "control_relative_error_percent":
                float(100 * np.linalg.norm(baseline - EXACT) / np.linalg.norm(EXACT)),
            "connected_relative_error_percent_exact_arithmetic": 0,
        })
    return {
        "description": "Analytic local stencil examples; no simulation accuracy claim.",
        "lattice_spacing": 1,
        "center_phase": PHI0,
        "exact_gradient_on_connected_side": EXACT.tolist(),
        "separated_neighbor_offset": [1, 1],
        "separated_neighbor_phase": 0.9,
        "ice_slab_coverage": "max(0, 1 - abs((xi_1 + xi_2 - 1) / sqrt(2)))",
        "solid_threshold": 0.5,
        "cases": records,
    }


def setup(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=7, **kwargs):
    return ax.text(x, y, label, fontsize=size, color=INK, va="center", **kwargs)


def arrow(ax, start, end, color=INK, lw=.65, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=5, lw=lw, color=color, **kwargs))


def dot(ax, x, y, radius=.045, fc="white", ec=TEAL, lw=.8, **kwargs):
    ax.add_patch(Circle((x, y), radius, fc=fc, ec=ec, lw=lw, **kwargs))


def panel(ax, x, y, letter, title):
    text(ax, x, y, letter, size=8, weight="bold")
    text(ax, x + .18, y, title, size=7)


def band_polygon(lo, hi, extent):
    """Clip the square to lo <= x+y <= hi, retaining vector geometry."""
    poly = [(-extent, -extent), (extent, -extent),
            (extent, extent), (-extent, extent)]
    for normal, bound in [(np.array([-1., -1.]), -lo), (np.ones(2), hi)]:
        result = []
        for k, p in enumerate(poly):
            p, q = np.array(p), np.array(poly[(k + 1) % len(poly)])
            fp, fq = normal @ p - bound, normal @ q - bound
            if fp <= 0:
                result.append(tuple(p))
            if (fp <= 0) != (fq <= 0):
                result.append(tuple(p + fp / (fp - fq) * (q - p)))
        poly = result
    return np.array(poly)


def stencil(ax, center, step, kind, numbers=False, links=True):
    cx, cy = center
    extent = 1.43
    for q in [-1, 0, 1]:
        ax.plot([cx - extent * step, cx + extent * step], [cy + q * step] * 2,
                color=GRID, lw=.4, zorder=1)
        ax.plot([cx + q * step] * 2, [cy - extent * step, cy + extent * step],
                color=GRID, lw=.4, zorder=1)
    if kind == "half":
        ax.add_patch(Rectangle((cx + .5 * step, cy - extent * step),
                              (extent - .5) * step, 2 * extent * step,
                              fc=ICE, ec="none", zorder=2))
        ax.plot([cx + .5 * step] * 2, [cy - extent * step, cy + extent * step],
                color=ICE_EDGE, lw=.55, zorder=3)
        mask = HALF_MASK
    else:
        poly = band_polygon(1 - 1 / np.sqrt(2), 1 + 1 / np.sqrt(2), extent)
        ax.add_patch(Polygon(np.array(center) + step * poly, closed=True,
                             fc=ICE, ec=ICE_EDGE, lw=.55, zorder=2))
        mask = CORNER_MASK
    for k, (dx, dy) in enumerate(DIRECTIONS):
        x, y = cx + step * dx, cy + step * dy
        is_remote = kind == "corner" and (dx, dy) == (1, 1)
        if links and mask[k]:
            ax.plot([cx, x], [cy, y], color=TEAL, lw=.7, zorder=3)
        if links and is_remote:
            ax.plot([cx, x], [cy, y], color=ORANGE, lw=.7,
                    ls=(0, (3, 2)), zorder=3)
            ax.plot(cx + .5 * step, cy + .5 * step, marker="x", ms=4.2,
                    mew=.85, color=ORANGE, zorder=5)
        if mask[k] or is_remote:
            dot(ax, x, y, radius=.104 if numbers else .048,
                ec=ORANGE if is_remote else TEAL, zorder=5)
            if numbers:
                value = .9 if is_remote else PHI0 + EXACT @ [dx, dy]
                text(ax, x, y, f"{value:.2f}", size=6.3, ha="center", zorder=6)
        else:
            ax.plot(x, y, marker="x", ms=3.5, mew=.6, color=ICE_EDGE, zorder=5)
    dot(ax, cx, cy, radius=.104 if numbers else .055,
        ec=INK, fc="white" if numbers else INK, lw=.95, zorder=6)
    if numbers:
        text(ax, cx, cy, "0.40", size=6.3, ha="center", zorder=7)
    else:
        text(ax, cx - .115, cy - .11, r"$x$", ha="center", zorder=7)


def rank_glyph(ax, x, y, rank):
    if rank:
        ax.plot([x - .085, x + .085], [y, y], color=TEAL, lw=.6)
        for xx in [x - .085, x + .085]:
            dot(ax, xx, y, radius=.023, lw=.6)
    if rank == 2:
        ax.plot([x, x], [y, y + .105], color=TEAL, lw=.6)
        dot(ax, x, y + .105, radius=.023, lw=.6)
    dot(ax, x, y, radius=.026, ec=INK, fc=INK, lw=.5)


def method_figure():
    fig, ax = setup(WIDTH, 3.12)
    panel(ax, .06, 2.99, "a", "Connected stencil")
    panel(ax, 1.86, 2.99, "b", "Gradient fit")
    stencil(ax, (.77, 2.36), .315, "corner")
    text(ax, 1.28, 2.73, "Far-side\nfluid", size=6.5, linespacing=1.2)
    ax.plot([1.12, 1.22, 1.26], [2.68, 2.73, 2.73], color=INK, lw=.45)
    text(ax, 1.28, 2.10, "Ice", size=6.5)
    ax.plot([1.12, 1.22, 1.26], [2.20, 2.10, 2.10], color=INK, lw=.45)
    dot(ax, .16, 1.78, radius=.033)
    text(ax, .25, 1.78, "Accepted", size=6.5)
    ax.plot([.93, 1.10], [1.78, 1.78], color=ORANGE, lw=.7, ls=(0, (3, 2)))
    ax.plot(1.015, 1.78, marker="x", color=ORANGE, ms=3.3, mew=.7)
    text(ax, 1.19, 1.78, "Blocked", size=6.5)

    for y, rank, label, formula in [
        (2.62, 2, "Rank 2", r"$\boldsymbol{D}_x=\boldsymbol{A}_x^{-1}\boldsymbol{b}_x$"),
        (2.17, 1, "Rank 1", r"$\boldsymbol{D}_x=\boldsymbol{b}_x/\mathrm{tr}\,\boldsymbol{A}_x$"),
        (1.72, 0, "No donors", r"$\boldsymbol{D}_x=\boldsymbol{0}$"),
    ]:
        rank_glyph(ax, 2.00, y, rank)
        text(ax, 2.21, y + .09, label, size=6.5)
        text(ax, 2.21, y - .10, formula)

    panel(ax, .06, 1.37, "c", "Refill exposed fluid")
    xs = [.47, 1.25, 2.03, 2.81]
    ys = [.84, .50]
    for col, x in enumerate(xs):
        for y in ys:
            ax.add_patch(Rectangle((x - .28, y - .15), .56, .30,
                                   fc=ICE if col == 3 else "white",
                                   ec=GRID, lw=.4))
    for y in ys:
        arrow(ax, (xs[0], y), (xs[1] - .07, y), color=TEAL)
        arrow(ax, (xs[1], y), (xs[2] - .07, y), color=TEAL)
        for col in range(3):
            x = xs[col]
            dot(ax, x, y, radius=.062,
                fc=TEAL if col == 0 else "white", ec=TEAL, lw=.7)
            if col:
                text(ax, x, y, str(col), size=6.5, ha="center")
    ax.plot([.86, .86], [.29, 1.03], color=INK, lw=.55, ls=(0, (2.5, 2)))
    for x, label in zip(xs, ["Known", "Pass 1", "Pass 2", "Ice"]):
        text(ax, x, 1.11, label, size=6.5, ha="center")
    text(ax, WIDTH / 2, .13, r"Connected averages of $(\phi,p',\boldsymbol{u})$",
         ha="center")
    return fig


def accuracy_figure(data):
    height = 3.12
    fig, ax = setup(WIDTH, height)
    panel(ax, .06, 2.99, "a", "Missing directions")
    panel(ax, 1.86, 2.99, "b", "Separated fluid")
    stencil(ax, (.82, 2.36), .315, "half", numbers=True, links=False)
    stencil(ax, (2.63, 2.36), .315, "corner", numbers=True, links=True)
    text(ax, .82, 1.78, r"Fixed scale: $3\boldsymbol{b}_x$", ha="center")
    text(ax, 2.63, 1.78, "Fit without connectivity", ha="center")
    panel(ax, .06, 1.37, "c", "Relative gradient error (%)")

    chart = fig.add_axes([1.27 / WIDTH, .40 / height, 1.95 / WIDTH, .65 / height])
    chart.set_xlim(0, 52)
    chart.set_ylim(-.50, 2.50)
    chart.set_yticks([])
    chart.set_xticks([0, 20, 40])
    chart.tick_params(axis="x", labelsize=6.5, colors=INK, length=2, width=.45, pad=2)
    chart.set_axisbelow(True)
    chart.grid(axis="x", color=GRID, lw=.4)
    chart.spines[["top", "right", "left"]].set_visible(False)
    chart.spines["bottom"].set_color(INK)
    chart.spines["bottom"].set_linewidth(.45)
    for y, value, label in [
        (2, data["cases"][0]["control_relative_error_percent"], "a  Fixed scale"),
        (1, data["cases"][1]["control_relative_error_percent"], "b  Ungated fit"),
    ]:
        chart.barh(y, value, height=.33, color="#919AA1", zorder=3)
        chart.text(-3.1, y, label, fontsize=6.5, color=INK, ha="right", va="center")
        chart.text(value + 1.5, y, f"{value:.1f}", fontsize=6.5, color=INK, va="center")
    chart.plot(0, 0, "D", color=TEAL, ms=3.3, clip_on=False, zorder=4)
    chart.text(-3.1, 0, "Connected fit (a, b)", fontsize=6.5,
               color=INK, ha="right", va="center")
    chart.text(2.6, 0, "0", fontsize=6.5, color=INK, va="center")
    text(ax, WIDTH / 2, .13,
         r"Target gradient: $\nabla\phi=(0.12,\,0.08)^{\mathsf{T}}$", ha="center")
    return fig


def main():
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.bfit": "Liberation Sans:bold:italic",
        "mathtext.sf": "Liberation Sans",
        "mathtext.cal": "Liberation Sans:italic",
        "mathtext.tt": "Liberation Mono",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "savefig.facecolor": "white",
    })
    OUT.mkdir(exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    data = analytic_cases()
    (OUT / "gradient_examples.json").write_text(json.dumps(data, indent=2) + "\n")
    for name, fig in [("connected_reconstruction", method_figure()),
                      ("gradient_accuracy", accuracy_figure(data))]:
        for ext in ["pdf", "svg"]:
            fig.savefig(OUT / f"{name}.{ext}")
        fig.savefig(PREVIEW / f"{name}.png", dpi=300)
        plt.close(fig)
    print("Generated two 89 mm single-column vector PDF/SVG figures.")


if __name__ == "__main__":
    main()
