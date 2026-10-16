"""Publication figures in a classic 1990s journal style: serif type, full black frames,
inward ticks on all four sides, thin lines, and only black, red and blue ink.
Models are told apart by colour AND marker shape/fill so the figures survive greyscale printing.

usage: python figures_retro.py --data <CWRU Data folder> [--res results] [--fig figures]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pacb.data import FS, fault_type, load_cwru, make_windows
from pacb.features import ORD, ORDER_GRID, PAF_NAMES

# ----------------------------------------------------------------------------- style
for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    font_manager.fontManager.addfont(f)
BLACK, RED, BLUE = "#000000", "#C8102E", "#0033A0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": BLACK, "axes.linewidth": 0.6, "axes.labelcolor": BLACK,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 0.8, "lines.markersize": 4, "lines.markeredgewidth": 0.7,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": BLACK, "legend.framealpha": 1.0,
    "legend.fontsize": 7.5, "legend.borderpad": 0.4, "legend.handlelength": 1.6,
    "hatch.linewidth": 0.4, "patch.linewidth": 0.6, "savefig.dpi": 300, "savefig.facecolor": "white",
})

MODELS = ["TS-RF", "FFT-LR", "EOS-LR", "PAF-LR", "CNN"]
# colour, marker, filled
STYLE = {"TS-RF": (BLACK, "^", False), "FFT-LR": (BLUE, "s", True), "EOS-LR": (BLUE, "D", False),
         "PAF-LR": (RED, "s", False), "CNN": (BLACK, "o", True), "PHY3-LR": (RED, "v", True)}
FAULT = {"OR": (BLUE, "o", True, "Outer race"), "IR": (RED, "s", False, "Inner race"), "B": (BLACK, "^", False, "Ball")}


def mk(model_or_style, ms=4.5):
    c, m, filled = model_or_style[:3]
    return dict(color=c, marker=m, markersize=ms, markerfacecolor=c if filled else "white",
                markeredgecolor=c, markeredgewidth=0.8)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def save(fig, path, **k):
    fig.savefig(path, **k)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def fig_acquisition(data, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, harm, ttl, letter in zip(axs, (False, True), ("As distributed (normal decimated 48 to 12 kHz)",
                                                          "Harmonised (common 4.8 kHz low-pass)"), "ab"):
        W = make_windows(load_cwru(data, harmonised=harm), stride=4096)
        X = W.X - W.X.mean(1, keepdims=True)
        S = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
        S /= S.sum(1, keepdims=True)
        f = np.fft.rfftfreq(X.shape[1], 1 / FS) / 1000
        n = W.label == "Normal"
        sm = lambda v: np.convolve(v, np.ones(16) / 16, mode="same")
        ax.plot(f, 10 * np.log10(sm(np.median(S[~n], 0)) + 1e-30), color=BLACK, lw=0.8, label="Fault records")
        ax.plot(f, 10 * np.log10(sm(np.median(S[n], 0)) + 1e-30), color=RED, lw=0.8, label="Normal records")
        ax.axvspan(5.5, 6.0, facecolor="none", edgecolor=BLACK, hatch="////", lw=0)
        ax.set_xlim(0, 6); ax.set_ylim(-140, -10)
        ax.minorticks_on()
        ax.set_xlabel("Frequency (kHz)"); ax.set_title(ttl, loc="left")
        panel(ax, letter)
    axs[0].set_ylabel("Relative PSD (dB)")
    axs[0].legend(loc="lower left", title="Median over windows", title_fontsize=7)
    axs[0].annotate("5.5 to 6 kHz", xy=(5.75, -100), xytext=(4.1, -122), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=BLACK))
    fig.tight_layout(w_pad=1.2)
    save(fig, os.path.join(out, "fig2_acquisition_audit.png"))


def fig_accuracy_coverage(m, out):
    prot = [("P1-leaky", "P1 random\n(leaky)"), ("P2-block", "P2 time block\n(in-domain)"),
            ("P3-cross-load", "P3 cross-load\n(12 pairs)"), ("P4-cross-size", "P4 cross-size\n(3 folds)")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, col, ttl, letter in zip(axs, ("acc", "coverage"), ("Accuracy", "Conformal coverage, target 0.90"), "ab"):
        for j, mod in enumerate(MODELS):
            xs, ys, lo, hi = [], [], [], []
            for i, (p, _) in enumerate(prot):
                s = m[(m.protocol == p) & (m.model == mod)]
                v = s.groupby("fold")[col].mean()
                xs.append(i + (j - 2) * 0.14); ys.append(s[col].mean()); lo.append(v.min()); hi.append(v.max())
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)], linestyle="none",
                        elinewidth=0.7, capsize=1.8, capthick=0.7, ecolor=STYLE[mod][0], label=mod, **mk(STYLE[mod]))
        if col == "coverage":
            ax.axhline(0.9, color=BLACK, lw=0.6, ls="--")
            ax.text(3.45, 0.91, "nominal", fontsize=7.5, ha="right", va="bottom")
        for x in (0.5, 1.5, 2.5):
            ax.axvline(x, color=BLACK, lw=0.4, ls=":")
        ax.set_xticks(range(4)); ax.set_xticklabels([b for _, b in prot], fontsize=7.5)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.2, 1.04); ax.set_yticks(np.arange(0.2, 1.01, 0.1), minor=True)
        ax.set_title(ttl, loc="left")
        panel(ax, letter)
    axs[0].set_ylabel("Fraction of target windows")
    axs[0].legend(loc="lower left", ncol=1, numpoints=1)
    fig.tight_layout(w_pad=1.0)
    fig.text(0.5, -0.01, "Marker: mean over folds and seeds. Bar: worst to best fold.", ha="center", fontsize=7.5)
    save(fig, os.path.join(out, "fig3_accuracy_coverage.png"), bbox_inches="tight")


def fig_silent_failure(m, out):
    s = m[m.protocol.isin(["P3-cross-load", "P4-cross-size"])]
    g = s.groupby(["model", "protocol", "fold"]).agg(cov=("coverage", "mean"), size=("set_size", "mean")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, p, ttl, letter in zip(axs, ("P3-cross-load", "P4-cross-size"),
                                  ("P3 cross-load, 12 pairs", "P4 unseen defect size, 3 folds"), "ab"):
        for mod in MODELS:
            e = g[(g.model == mod) & (g.protocol == p)]
            ax.plot(e["size"], e["cov"], linestyle="none", label=mod, **mk(STYLE[mod], 5))
        ax.axhline(0.9, color=BLACK, lw=0.6, ls="--")
        ax.text(0.98, 0.905, "nominal 0.90", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5)
        ax.set_xlim(0.9, 2.0); ax.set_ylim(0.25, 1.03); ax.minorticks_on()
        ax.set_xlabel("Mean prediction-set size (labels)"); ax.set_title(ttl, loc="left")
        panel(ax, letter)
    axs[0].set_ylabel("Empirical coverage on target")
    axs[1].legend(loc="lower right", numpoints=1)
    fig.tight_layout(w_pad=1.0)
    save(fig, os.path.join(out, "fig4_silent_failure.png"))


def fig_orders(data, res, out):
    F = np.load(os.path.join(res, "features.npz"))
    W = make_windows(load_cwru(data))
    eos = 10 ** F["eos"]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
    marks = {"OR": ("BPFO", ORD["BPFO"]), "IR": ("BPFI", ORD["BPFI"]), "B": ("2BSF", 2 * ORD["BSF"])}
    for ax, t, ttl, letter in zip(axs, ["N", "OR", "IR", "B"], ["Normal", "Outer race (all sizes, 6 o'clock)",
                                                                  "Inner race (all sizes)", "Ball (all sizes)"], "abcd"):
        sel = np.array([fault_type(l) == t for l in W.label])
        med = np.median(eos[sel], 0)
        for kk, (nm, o) in marks.items():
            for h in (1, 2, 3):
                if h * o <= 12:
                    if kk == t:
                        ax.axvline(h * o, color=RED, lw=0.7, ls="--")
                    else:
                        ax.axvline(h * o, color=BLACK, lw=0.4, ls=":")
        ax.plot(ORDER_GRID, 20 * np.log10(med), color=BLACK if t == "N" else BLUE, lw=0.8)
        ax.text(0.99, 0.88, ttl, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_ylabel("dB re median", fontsize=8)
        ax.minorticks_on()
        panel(ax, letter, x=-0.075, y=0.80)
    for kk, (nm, o) in marks.items():
        axs[0].text(o, axs[0].get_ylim()[1], nm if nm != "2BSF" else r"2$\times$BSF", ha="center", va="bottom", fontsize=7.5)
    axs[-1].set_xlabel("Envelope order (multiples of shaft speed)")
    fig.align_ylabels(axs)
    axs[-1].set_xlim(0.4, 12.1)
    fig.legend(handles=[Line2D([], [], color=RED, lw=0.7, ls="--", label="Harmonics of the expected order"),
                        Line2D([], [], color=BLACK, lw=0.4, ls=":", label="Harmonics of the other orders")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6)
    save(fig, os.path.join(out, "fig5_envelope_orders.png"))


def record_signatures(data, res):
    F = np.load(os.path.join(res, "features.npz"))
    recs = load_cwru(data)
    W = make_windows(recs)
    exp = {"OR": "BPFO_h1", "IR": "BPFI_h1", "B": "2BSF_h1"}
    rows = []
    for i, r in enumerate(recs):
        t = fault_type(r.label)
        if t == "N":
            continue
        c = PAF_NAMES.index(exp[t])
        rows.append(dict(rec=i, label=r.label, load=r.load, type=t, cwru_id=r.cwru_id,
                         signature_db=20 * np.median(F["paf"][W.rec == i, c])))
    return pd.DataFrame(rows)


def fig_signature(preds, sig, out):
    p4 = preds[preds.protocol == "P4-cross-size"]
    acc = p4.assign(ok=p4.y == p4.yhat).groupby(["model", "rec"]).ok.mean().unstack(0)
    d = sig.set_index("rec").join(acc)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, mod, letter in zip(axs, ("CNN", "FFT-LR", "PHY3-LR"), "abc"):
        ax.axvspan(10.2, 14.3, facecolor="none", edgecolor=BLACK, hatch="////", lw=0)
        for t in ("OR", "IR", "B"):
            e = d[d.type == t]
            ax.plot(e.signature_db, e[mod], linestyle="none", label=FAULT[t][3], **mk(FAULT[t], 4.5))
        rho = d[["signature_db", mod]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{mod}, Spearman $\\rho$ = " + f"{rho:.2f}".replace("-", "−"), loc="left", fontsize=8.5)
        ax.set_xlabel("Signature at expected order (dB)", fontsize=8.5)
        ax.set_xlim(5, 27); ax.set_ylim(-0.05, 1.05); ax.minorticks_on()
        panel(ax, letter)
    axs[0].set_ylabel("Record accuracy, unseen size")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, numpoints=1, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    save(fig, os.path.join(out, "fig6_signature_vs_accuracy.png"))


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, text, ec=BLACK, lw=0.8, fs=7.6, bold_first=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.2", fc="white", ec=ec, lw=lw))
        lines = text.split("\n")
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, color=BLACK,
                linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=8))

    box(0.5, 33, 14, 12, "CWRU record\n12 kHz drive end\n(record number\nverified)")
    box(18, 33, 14, 12, "Harmonise\nnormal: 48 to 12 kHz\n4.8 kHz low-pass\n(all records)", ec=RED, lw=1.1)
    box(35.5, 33, 13, 12, "Windows\nN = 4096\nstride 1024")
    arrow(15.0, 39, 17.6, 39); arrow(32.4, 39, 35.1, 39)
    reps = [("Time statistics (10)\nrandom forest", 41.5, 6), ("Log spectrum (256)\nlogistic regression", 34.5, 6),
            ("Kurtosis band, SES,\norder grid: EOS / PAF / PHY3\nlogistic regression", 25.0, 8.5),
            ("Raw window\n1D CNN", 17, 6)]
    for t, y, h in reps:
        env = "Kurtosis" in t
        box(53, y, 20, h, t, ec=BLUE if env else BLACK, lw=1.1 if env else 0.8, fs=7.2)
        arrow(48.9, 39, 52.6, y + h / 2)
        arrow(73.4, y + h / 2, 76.6, 37)
    box(77, 30, 22, 14, "Split conformal\n($\\alpha$ = 0.1)\nthreshold from source\ncalibration block\nprediction set")
    box(77, 9, 22, 15, "Label-free shift checks\n(unlabelled target pool):\ndomain AUC, weighted\nconformal, Mahalanobis", fs=7.2)
    arrow(88, 24.3, 88, 29.6)
    ax.text(0.5, 27.5, "Protocols", fontsize=8.5, color=BLACK, weight="bold")
    for i, t in enumerate(["P1  random windows (leaky)", "P2  time blocks (in-domain)",
                           "P3  load a to load b (12 pairs)", "P4  unseen defect size (3 folds)"]):
        ax.text(0.5, 23 - 4.3 * i, t, fontsize=7.8, color=BLACK)
    ax.plot([0.5, 46], [3.5, 3.5], color=BLACK, lw=0.5)
    ax.text(0.5, 0.6, "Physics audit: envelope signature at each record's expected order vs record accuracy",
            fontsize=7.8, color=BLACK)
    fig.tight_layout(pad=0.2)
    save(fig, os.path.join(out, "fig1_pipeline.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", default="results")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    m = pd.concat([pd.read_csv(os.path.join(a.res, f)) for f in ("metrics.csv", "metrics_phy3.csv")
                   if os.path.exists(os.path.join(a.res, f))], ignore_index=True)
    preds = pd.concat([pd.read_csv(os.path.join(a.res, f)) for f in ("predictions.csv.gz", "predictions_phy3.csv.gz")
                       if os.path.exists(os.path.join(a.res, f))], ignore_index=True)
    fig_pipeline(a.fig)
    fig_acquisition(a.data, a.fig)
    fig_accuracy_coverage(m, a.fig)
    fig_silent_failure(m, a.fig)
    fig_orders(a.data, a.res, a.fig)
    fig_signature(preds, record_signatures(a.data, a.res), a.fig)


if __name__ == "__main__":
    main()
