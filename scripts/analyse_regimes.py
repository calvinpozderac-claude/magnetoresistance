#!/usr/bin/env python3
r"""All regimes of D(r_c, tau), for the raw and the orbit-averaged landscape.

Every cell is reduced to the landscape's own units, so the two models can be
compared directly.  With Gamma_eff, v_eff = rms|grad V'| and
xi_eff = sqrt(2) Gamma_eff / v_eff measured from the field itself,

    rho = r_c / xi_eff        T = v_eff tau / xi_eff      Dh = D / (v_eff xi_eff)

For the raw field xi_eff = xi_0 and v_eff = v_0 by construction.  Orbit
averaging leaves xi_eff at sqrt(2) xi_0 but drives v_eff ~ r_c^(-1/2), so the
averaged landscape is *weaker*, not coarser -- which is why the two models look
different in lab units and identical in these.

The plane needs three collapses, each of which is a single curve:

  A   Dh            vs  u = rho^2 / T       regimes 1 and 2, kink at u = 1
  B   Dh T^(3/7)    vs  X = rho T^(3/7)     regimes 2 and 3a, kink at X = 1
  C   Dh T^(3/7)    vs  rho                 regimes 3a and 3b, kink at rho = 1

Regime 3b -- the kick longer than the correlation length -- is not in the notes;
it is a function of rho rather than X, which is why it breaks collapse B.

    python scripts/analyse_regimes.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# measured for the bare field (scripts/measure_avg_landscape.py, r_c = 0)
RAW_V, RAW_XI = 0.99374, 1.0063

# predicted exponents (a, b) of Dh ~ T^a rho^b in each regime
PRED = {"1": (-1.0, 2.0), "2": (-3 / 13, 6 / 13), "3a": (-3 / 7, 0.0),
        "3b": (-3 / 7, 3 / 7)}
# 3b is not in the notes.  Empirically Dh depends only on rho/T = r_c/(v tau),
# i.e. it is regime 3 with the tube width taken at coarse-graining scale r_c
# instead of xi:  w = xi (L/xi)^(-3/7)  ->  xi (L/r_c)^(-3/7).
LABEL = {"1": "1  free walk", "2": "2  small kick, Levy contours",
         "3a": "3a  contour-limited, $r_c<\\xi$",
         "3b": "3b  contour-limited, $r_c>\\xi$"}
# the fourth collapse: regime 3b alone, against its single variable
WVAR = r"$w=\rho/T=r_c/v_{\rm eff}\tau$"


def load(paths, ll_tol=0.15):
    """Flatten a set of run_grid.py outputs into per-cell arrays."""
    out = {k: [] for k in ("r_c", "tau", "D", "E", "xi", "v", "avg")}
    for q in paths:
        z = np.load(q)
        rc, tau = z["r_c"], z["tau"]
        avg = bool(z["orbit_average"]) if "orbit_average" in z else False
        has = "v_eff" in z and np.isfinite(z["v_eff"]).any()
        for a, r in enumerate(rc):
            v = float(z["v_eff"][a]) if has else RAW_V
            xi = float(z["xi_eff"][a]) if has else RAW_XI
            if not np.isfinite(v):
                v, xi = RAW_V, RAW_XI
            for b, t in enumerate(tau):
                if not z["done"][a, b]:
                    continue
                if abs(z["loglog"][a, b] - 1.0) > ll_tol:
                    continue
                out["r_c"].append(r); out["tau"].append(t)
                out["D"].append(z["D"][a, b]); out["E"].append(z["D_err"][a, b])
                out["xi"].append(xi); out["v"].append(v); out["avg"].append(avg)
    d = {k: np.asarray(v) for k, v in out.items()}
    # later files win on duplicated (r_c, tau) cells
    key = d["r_c"] * 1e6 + d["tau"]
    _, last = np.unique(key[::-1], return_index=True)
    keep = d["r_c"].size - 1 - last
    d = {k: v[np.sort(keep)] for k, v in d.items()}
    d["rho"] = d["r_c"] / d["xi"]
    d["T"] = d["v"] * d["tau"] / d["xi"]
    d["Dh"] = d["D"] / (d["v"] * d["xi"])
    d["Eh"] = d["E"] / (d["v"] * d["xi"])
    d["u"] = d["rho"] ** 2 / d["T"]
    d["X"] = d["rho"] * d["T"] ** (3 / 7)
    d["Y"] = d["Dh"] * d["T"] ** (3 / 7)
    d["Ey"] = d["Eh"] * d["T"] ** (3 / 7)
    return d


def regime(d, margin=0.0):
    """Label every cell 1 / 2 / 3a / 3b, blank if within `margin` of a line."""
    m1 = np.log(np.sqrt(d["T"]) / d["rho"])     # >0 below the free-walk line
    m2 = np.log(d["X"])                         # >0 above the 2/3 line
    m3 = np.log(d["rho"])                       # >0 above the 3a/3b line
    lab = np.full(d["rho"].size, "", dtype="<U2")
    lab[(-m1 > margin)] = "1"
    lab[(m1 > margin) & (-m2 > margin)] = "2"
    lab[(m1 > margin) & (m2 > margin) & (-m3 > margin)] = "3a"
    lab[(m1 > margin) & (m2 > margin) & (m3 > margin)] = "3b"
    return lab


def jfit(d, sel):
    """ln Dh = a ln T + b ln rho + c, weighted by the measured errors."""
    n = int(sel.sum())
    if n < 5:
        return None
    w = (d["Dh"][sel] / np.maximum(d["Eh"][sel], 1e-12)) ** 2
    X = np.c_[np.log(d["T"][sel]), np.log(d["rho"][sel]), np.ones(n)]
    y = np.log(d["Dh"][sel])
    W = np.diag(w)
    cov0 = np.linalg.inv(X.T @ W @ X)
    beta = cov0 @ (X.T @ W @ y)
    res = y - X @ beta
    scale = max((w * res ** 2).sum() / w.sum() * n / (n - 3), 0.0)
    cov = cov0 * (w.sum() / n) * scale
    return n, beta[0], np.sqrt(cov[0, 0]), beta[1], np.sqrt(cov[1, 1]), \
        float(np.sqrt((res ** 2).mean()))


def powerfit(x, y, sy):
    lx, ly, w = np.log(x), np.log(y), (y / np.maximum(sy, 1e-12)) ** 2
    W = w.sum(); mx, my = (w * lx).sum() / W, (w * ly).sum() / W
    sxx = (w * (lx - mx) ** 2).sum()
    b = (w * (lx - mx) * (ly - my)).sum() / sxx
    res = ly - my - b * (lx - mx)
    chi2 = (w * res ** 2).sum() / max(lx.size - 2, 1)
    return b, np.sqrt(max(chi2, 1.0) / sxx), np.exp(my - b * mx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", nargs="+", default=sorted(glob.glob("data/grid*.npz")))
    p.add_argument("--avg", nargs="+", default=sorted(glob.glob("data/avg_[A-Z].npz")))
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()

    raw, avg = load(args.raw), load(args.avg)
    sets = [("raw potential", raw, "#20558a", "o"),
            ("orbit-averaged", avg, "#c2492f", "s")]
    for name, d, _, _ in sets:
        print(f"{name}: {d['rho'].size} cells   "
              f"rho {d['rho'].min():.3g}..{d['rho'].max():.3g}   "
              f"T {d['T'].min():.3g}..{d['T'].max():.3g}")

    # ---------------- exponents -------------------------------------------
    print(f"\njoint fits  Dh ~ T^a rho^b  (cells >= e^{args.margin:g} from every "
          f"regime line; errors weighted by the measured D_err)")
    print(f"{'regime':<32}{'n':>4}  {'a (T)':>20}  {'b (rho)':>20}   predicted")
    fits = {}
    for name, d, _, _ in sets:
        print(f"  -- {name} --")
        lab = regime(d, args.margin)
        for g in ("1", "2", "3a", "3b"):
            f = jfit(d, lab == g)
            fits[(name, g)] = f
            pa, pb = PRED[g]
            ps = f"({pa:+.3f}, " + ("--" if pb is None else f"{pb:+.3f}") + ")"
            if f is None:
                print(f"    {LABEL[g]:<28}{int((lab==g).sum()):>4}  "
                      f"{'too few cells':>20}   {'':>20}   {ps}")
                continue
            n, a, ea, b, eb, rms = f
            print(f"    {LABEL[g]:<28}{n:>4}  {a:+.3f} +- {ea:.3f}      "
                  f"{b:+.3f} +- {eb:.3f}        {ps}")

    # ---------------- the three collapses ---------------------------------
    def panel(ax, which, d, col, mk, lab_prefix=""):
        lab = regime(d, 0.0)
        if which == "A":                       # regimes 1 and 2
            sel = (lab == "1") | (lab == "2")
            x, y, e = d["u"][sel], d["Dh"][sel], d["Eh"][sel]
            groups = [("1", lab[sel] == "1"), ("2", lab[sel] == "2")]
        elif which == "B":                     # regimes 2 and 3a
            sel = ((lab == "2") | (lab == "3a")) & (d["rho"] <= 1.0)
            x, y, e = d["X"][sel], d["Y"][sel], d["Ey"][sel]
            groups = [("2", lab[sel] == "2"), ("3a", lab[sel] == "3a")]
        elif which == "C":                     # regimes 3a and 3b
            sel = ((lab == "3a") | (lab == "3b")) & (d["X"] >= 1.0)
            x, y, e = d["rho"][sel], d["Y"][sel], d["Ey"][sel]
            groups = [("3a", lab[sel] == "3a"), ("3b", lab[sel] == "3b")]
        else:                                  # regime 3b on its own variable
            sel = lab == "3b"
            x = d["rho"][sel] / d["T"][sel]
            y, e = d["Dh"][sel], d["Eh"][sel]
            groups = [("3b", np.ones(int(sel.sum()), dtype=bool))]
        ax.errorbar(x, y, yerr=e, fmt=mk, color=col, ms=5, lw=0, elinewidth=0.9,
                    capsize=1.6, alpha=0.85, zorder=3,
                    label=f"{lab_prefix}{int(sel.sum())} cells")
        res = {}
        for g, m in groups:
            if m.sum() >= 4:
                res[g] = powerfit(x[m], y[m], e[m])
        return x, y, res

    guide = {"A": [("1", 1.0), ("2", 3 / 13)],
             "B": [("2", 6 / 13), ("3a", 0.0)],
             "C": [("3a", 0.0), ("3b", 3 / 7)],
             "D": [("3b", 3 / 7)]}
    axis = {"A": (r"$u=\rho^2/T$", r"$\hat D=D/v_{\rm eff}\xi_{\rm eff}$",
                  "A  regimes 1 + 2"),
            "B": (r"$X=\rho\,T^{3/7}$", r"$\hat D\,T^{3/7}$",
                  r"B  regimes 2 + 3a   ($\rho\leq1$)"),
            "C": (r"$\rho=r_c/\xi_{\rm eff}$", r"$\hat D\,T^{3/7}$",
                  r"C  regimes 3a + 3b   ($X\geq1$)"),
            "D": (r"$\rho/T=r_c/v_{\rm eff}\tau$", r"$\hat D$",
                  r"D  regime 3b alone, one variable")}

    # figure 1: the two models overlaid on the same three collapses
    _nullf, _null = plt.subplots()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.4))
    for k, which in enumerate("ABCD"):
        ax = axes.flat[k]
        for name, d, col, mk in sets:
            x, y, res = panel(ax, which, d, col, mk, lab_prefix=f"{name}: ")
            for g, (b, eb, amp) in res.items():
                xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 12)
                ax.plot(xx, amp * xx ** b, "-", color=col, lw=1.0, alpha=0.35)
        ax.set_xscale("log"); ax.set_yscale("log")
        # predicted slopes, anchored on the raw data so they are guides not fits
        for g, sl in guide[which]:
            if sl is None:
                continue
            xr, yr, _ = panel(_null, which, raw, "k", "o")
            side = (xr <= 1.0) if g in ("2",) and which != "A" else (xr >= 1.0)
            if which == "A":
                side = xr >= 1.0 if g == "1" else xr <= 1.0
            if which == "D":
                side = np.ones(xr.size, dtype=bool)
            if side.sum() < 2:
                continue
            anchor = np.exp(np.median(np.log(yr[side]) - sl * np.log(xr[side])))
            xs = np.array([xr[side].min(), xr[side].max()])
            ax.plot(xs, anchor * xs ** sl, "k--", lw=1.2, zorder=5)
            ax.annotate(f"slope {sl:.3f}" if sl else "slope 0",
                        (xs[-1], anchor * xs[-1] ** sl), fontsize=8,
                        textcoords="offset points", xytext=(4, -2))
        if which != "D":
            ax.axvline(1.0, color="0.55", ls=":", lw=1.2)
        xl, yl, ti = axis[which]
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ti, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Raw and orbit-averaged landscapes in their own units "
                 r"($\rho=r_c/\xi_{\rm eff}$, $T=v_{\rm eff}\tau/\xi_{\rm eff}$)",
                 fontsize=12)
    plt.close(_nullf)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(args.outdir, exist_ok=True)
    q = os.path.join(args.outdir, "regimes_overlay.png")
    fig.savefig(q, dpi=165); print("\nwrote", q)

    # ---------------- report the collapse fits ----------------------------
    print("\ncollapse branch fits (weighted power laws)")
    for which in "ABCD":
        xl, _, ti = axis[which]
        print(f"  {ti}   x = {xl}")
        for name, d, col, mk in sets:
            f2, ax2 = plt.subplots()
            _, _, res = panel(ax2, which, d, col, mk)
            plt.close(f2)
            for g in ("1", "2", "3a", "3b"):
                if g in res:
                    b, eb, _ = res[g]
                    pb = {"A": {"1": 1.0, "2": 3 / 13},
                          "B": {"2": 6 / 13, "3a": 0.0},
                          "C": {"3a": 0.0, "3b": 3 / 7},
                          "D": {"3b": 3 / 7}}[which].get(g)
                    ps = "--" if pb is None else f"{pb:+.3f}"
                    print(f"     {name:<16} regime {g:<3} slope "
                          f"{b:+.3f} +- {eb:.3f}   predicted {ps}")
    print("\nregime 3b as a one-parameter law  Dh = C (r_c / v_eff tau)^p:")
    for name, d, _, _ in sets:
        lab = regime(d, args.margin)
        m = lab == "3b"
        if m.sum() < 4:
            print(f"  {name:<16} only {int(m.sum())} cells")
            continue
        w = d["rho"][m] / d["T"][m]
        pw, epw, amp = powerfit(w, d["Dh"][m], d["Eh"][m])
        C = d["Dh"][m] / w ** (3 / 7)
        print(f"  {name:<16} n={int(m.sum()):3d}   p = {pw:+.4f} +- {epw:.4f} "
              f"(3/7 = {3/7:.4f})   C = {C.mean():.3f} "
              f"(spread {100*C.std(ddof=1)/C.mean():.1f}%)")
    print("regime 3a amplitude  Dh T^(3/7) = C:")
    for name, d, _, _ in sets:
        lab = regime(d, args.margin)
        m = lab == "3a"
        if m.sum() < 3:
            print(f"  {name:<16} only {int(m.sum())} cells")
            continue
        C = d["Dh"][m] * d["T"][m] ** (3 / 7)
        print(f"  {name:<16} n={int(m.sum()):3d}   C = {C.mean():.3f} +- "
              f"{C.std(ddof=1)/np.sqrt(m.sum()):.3f}  "
              f"(spread {100*C.std(ddof=1)/C.mean():.1f}%)")

    json.dump({f"{k[0]}|{k[1]}": (None if v is None else list(map(float, v)))
               for k, v in fits.items()},
              open(os.path.join("data", "regime_fits.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
