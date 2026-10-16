"""Aggregate metrics into paper tables and draw the figures.

usage: python analyse.py --data <CWRU Data folder> --res results --fig figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pacb.data import CLASSES10, FS, fault_type, load_cwru, make_windows
from pacb.features import ORD, ORDER_GRID, PAF_NAMES

MODELS = ["TS-RF", "FFT-LR", "EOS-LR", "PAF-LR", "CNN"]
# Reference categorical palette (dataviz skill, light mode), fixed order per model.
COL = {"TS-RF": "#eb6834", "FFT-LR": "#1baf7a", "EOS-LR": "#6250d6", "PAF-LR": "#2a78d6", "CNN": "#e34948",
       "PHY3-LR": "#008300"}
GREY, INK, INK2 = "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def mean_sd(d, col):
    g = d.groupby("model")[col]
    return g.mean(), g.std(ddof=0)


def tables(m, out):
    T = {}
    # Table: leakage (P1 vs P2), mean +/- sd over seeds
    rows = []
    for mod in MODELS:
        r = {"model": mod}
        for p in ("P1-leaky", "P2-block"):
            s = m[(m.protocol == p) & (m.model == mod)]
            r[f"{p} acc"] = f"{100 * s.acc.mean():.2f} ± {100 * s.acc.std(ddof=0):.2f}"
            r[f"{p} cov"] = f"{s.coverage.mean():.3f}"
            r[f"{p} |S|"] = f"{s.set_size.mean():.2f}"
        rows.append(r)
    T["leakage"] = pd.DataFrame(rows)
    # Table: transfer summary (fold-mean then seed-mean/sd; min over folds)
    rows = []
    for p in ("P2-block", "P3-cross-load", "P4-cross-size"):
        for mod in MODELS + ["PAF-LR (no order norm.)", "PHY3-LR"]:
            s = m[(m.protocol == p) & (m.model == mod)]
            if s.empty:
                continue
            per_seed = s.groupby("seed").agg(acc=("acc", "mean"), f1=("macro_f1", "mean"),
                                             cov=("coverage", "mean"), size=("set_size", "mean"))
            fold = s.groupby("fold").acc.mean()
            rows.append(dict(protocol=p, model=mod, acc_mean=per_seed.acc.mean(), acc_sd=per_seed.acc.std(ddof=0),
                             f1_mean=per_seed.f1.mean(), worst_fold_acc=fold.min(),
                             worst_fold=fold.idxmin(), coverage=per_seed["cov"].mean(),
                             min_fold_coverage=s.groupby("fold").coverage.mean().min(),
                             set_size=per_seed["size"].mean(), singleton_rate=s.singleton_rate.mean(),
                             selective_acc=s.selective_acc.mean()))
    T["transfer"] = pd.DataFrame(rows)
    # Table: shift-aware conformal and error detection (transfer protocols only)
    rows = []
    for p in ("P3-cross-load", "P4-cross-size"):
        for mod in MODELS + ["PHY3-LR"]:
            s = m[(m.protocol == p) & (m.model == mod)]
            if s.empty:
                continue
            rows.append(dict(protocol=p, model=mod, coverage=s.coverage.mean(), set_size=s.set_size.mean(),
                             cov_wfft=s.coverage_wfft.mean(), size_wfft=s.set_size_wfft.mean(),
                             full_wfft=s.full_set_rate_wfft.mean(),
                             cov_wpaf=s.coverage_wpaf.mean(), size_wpaf=s.set_size_wpaf.mean(),
                             full_wpaf=s.full_set_rate_wpaf.mean(),
                             auc_dom_fft=s.domain_auc_fft.mean(), auc_dom_paf=s.domain_auc_paf.mean(),
                             err_msp=s.errdet_auc_msp.mean(), err_md_paf=s.errdet_auc_md_paf.mean(),
                             err_setsize=s.errdet_auc_setsize.mean()))
    T["shift"] = pd.DataFrame(rows)
    # per-fold accuracy matrix for P3 (seed mean)
    T["p3_folds"] = m[m.protocol == "P3-cross-load"].pivot_table(index="fold", columns="model", values="acc")
    T["p4_folds"] = m[m.protocol == "P4-cross-size"].pivot_table(index="fold", columns="model", values="acc")
    T["p4_cov"] = m[m.protocol == "P4-cross-size"].pivot_table(index="fold", columns="model", values="coverage")
    for k, v in T.items():
        v.to_csv(os.path.join(out, f"table_{k}.csv"), index=k in ("p3_folds", "p4_folds", "p4_cov"))
    return T


def fig_acquisition(data, fig):
    fig_, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, harm, ttl in zip(axs, (False, True), ("As distributed (normal decimated 48→12 kHz)",
                                                  "Harmonised (common 4.8 kHz low-pass)")):
        W = make_windows(load_cwru(data, harmonised=harm), stride=4096)
        X = W.X - W.X.mean(1, keepdims=True)
        S = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
        S /= S.sum(1, keepdims=True)
        f = np.fft.rfftfreq(X.shape[1], 1 / FS) / 1000
        n = W.label == "Normal"
        k = 16
        sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
        ax.plot(f, 10 * np.log10(sm(np.median(S[~n], 0)) + 1e-30), color=GREY, lw=1.2, label="Fault records (median)")
        ax.plot(f, 10 * np.log10(sm(np.median(S[n], 0)) + 1e-30), color=COL["PAF-LR"], lw=1.4, label="Normal records (median)")
        ax.axvspan(5.5, 6.0, color=GREY, alpha=0.15, lw=0)
        ax.set_xlim(0, 6); ax.set_ylim(-140, -10)
        ax.set_xlabel("Frequency (kHz)"); ax.set_title(ttl, fontsize=9, color=INK, loc="left")
    axs[0].set_ylabel("Relative PSD (dB)")
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[0].text(5.75, -20, "5.5–6 kHz", ha="center", fontsize=7, color=INK2)
    fig_.tight_layout(); fig_.savefig(os.path.join(fig, "fig2_acquisition_audit.png")); plt.close(fig_)


def fig_orders(data, res, fig):
    F = np.load(os.path.join(res, "features.npz"))
    W = make_windows(load_cwru(data))
    eos = 10 ** F["eos"]
    fig_, axs = plt.subplots(4, 1, figsize=(7.2, 5.6), sharex=True)
    marks = {"OR": ("BPFO", ORD["BPFO"]), "IR": ("BPFI", ORD["BPFI"]), "B": ("2×BSF", 2 * ORD["BSF"])}
    for ax, t, ttl in zip(axs, ["N", "OR", "IR", "B"], ["Normal", "Outer race (all sizes, 6 o'clock)",
                                                         "Inner race (all sizes)", "Ball (all sizes)"]):
        sel = np.array([fault_type(l) == t for l in W.label])
        med = np.median(eos[sel], 0)
        ax.plot(ORDER_GRID, 20 * np.log10(med), color=INK if t == "N" else COL["PAF-LR"], lw=1.2)
        for kk, (nm, o) in marks.items():
            for h in (1, 2, 3):
                if h * o <= 12:
                    ax.axvline(h * o, color=GREY, lw=0.8, ls=":" if kk != t else "-", alpha=0.9)
            if kk == t:
                ax.text(o, ax.get_ylim()[1] if False else 0, "", fontsize=7)
        ax.set_title(ttl, fontsize=8, loc="right", color=INK, pad=2)
        ax.set_ylabel("dB re median", fontsize=8)
    for kk, (nm, o) in marks.items():
        axs[0].text(o, axs[0].get_ylim()[1], nm, ha="center", va="bottom", fontsize=7, color=INK2)
    axs[-1].set_xlabel("Envelope order (multiples of shaft speed)")
    fig_.tight_layout(); fig_.savefig(os.path.join(fig, "fig5_envelope_orders.png")); plt.close(fig_)


def fig_accuracy_coverage(m, fig):
    prot = [("P1-leaky", "P1 random\n(leaky)"), ("P2-block", "P2 time-block\n(in-domain)"),
            ("P3-cross-load", "P3 cross-load\n(12 pairs)"), ("P4-cross-size", "P4 cross-size\n(3 folds)")]
    fig_, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, col, ttl in zip(axs, ("acc", "coverage"), ("Accuracy", "Conformal coverage (target 0.90)")):
        for j, mod in enumerate(MODELS):
            xs, ys, lo, hi = [], [], [], []
            for i, (p, _) in enumerate(prot):
                s = m[(m.protocol == p) & (m.model == mod)]
                v = s.groupby("fold")[col].mean()
                xs.append(i + (j - 2) * 0.13); ys.append(s[col].mean()); lo.append(v.min()); hi.append(v.max())
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)], fmt="o", ms=4, color=COL[mod], lw=1, capsize=0, label=mod)
        if col == "coverage":
            ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
            ax.text(3.45, 0.905, "nominal", fontsize=7, color=INK2, ha="right", va="bottom")
        ax.set_xticks(range(4)); ax.set_xticklabels([b for _, b in prot], fontsize=7.5)
        ax.set_ylim(0.2, 1.02); ax.set_title(ttl, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
    fig_.text(0.5, -0.02, "Dot: mean over folds and seeds. Bar: range of fold means (worst to best transfer).",
              ha="center", fontsize=7, color=INK2)
    fig_.tight_layout(); fig_.savefig(os.path.join(fig, "fig3_accuracy_coverage.png"), bbox_inches="tight"); plt.close(fig_)


def fig_silent_failure(m, fig):
    s = m[m.protocol.isin(["P3-cross-load", "P4-cross-size"])]
    g = s.groupby(["model", "protocol", "fold"]).agg(cov=("coverage", "mean"), size=("set_size", "mean")).reset_index()
    fig_, ax = plt.subplots(figsize=(4.2, 3.2))
    for mod in MODELS:
        d = g[g.model == mod]
        for p, mk in (("P3-cross-load", "o"), ("P4-cross-size", "s")):
            e = d[d.protocol == p]
            ax.scatter(e["size"], e["cov"], s=22, marker=mk, color=COL[mod], edgecolor="white", lw=0.6,
                       label=mod if p == "P3-cross-load" else None)
    ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Mean prediction-set size (labels)"); ax.set_ylabel("Empirical coverage on target")
    ax.text(ax.get_xlim()[1], 0.905, "nominal 0.90", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc="lower right", title="circle: cross-load, square: cross-size",
              title_fontsize=6.5)
    ax.grid(color="#e6e5e1", lw=0.6)
    fig_.tight_layout(); fig_.savefig(os.path.join(fig, "fig4_silent_failure.png")); plt.close(fig_)


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


def fig_signature(m_pred, sig, fig, out):
    p4 = m_pred[m_pred.protocol == "P4-cross-size"]
    acc = p4.assign(ok=p4.y == p4.yhat).groupby(["model", "rec"]).ok.mean().unstack(0)
    d = sig.set_index("rec").join(acc)
    d.to_csv(os.path.join(out, "table_record_signature.csv"))
    fig_, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, mod in zip(axs, ("CNN", "FFT-LR", "PHY3-LR")):
        for t, mk in (("OR", "o"), ("IR", "s"), ("B", "^")):
            e = d[d.type == t]
            ax.scatter(e.signature_db, e[mod], marker=mk, s=22, color={"OR": "#2a78d6", "IR": "#eb6834", "B": GREY}[t],
                       edgecolor="white", lw=0.5, label={"OR": "Outer race", "IR": "Inner race", "B": "Ball"}[t])
        rho = d[["signature_db", mod]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{mod}  (Spearman ρ = {rho:.2f})", fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("Signature at expected order (dB)", fontsize=8)
        ax.axvspan(10.2, 14.3, color=GREY, alpha=0.12, lw=0)
        ax.grid(color="#e6e5e1", lw=0.6)
    axs[0].set_ylabel("P4 record accuracy (unseen size)")
    h, l = axs[0].get_legend_handles_labels()
    fig_.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig_.tight_layout(rect=(0, 0.07, 1, 1))
    fig_.savefig(os.path.join(fig, "fig6_signature_vs_accuracy.png")); plt.close(fig_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", default="results")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    m = pd.concat([pd.read_csv(os.path.join(a.res, f)) for f in ("metrics.csv", "metrics_phy3.csv")
                   if os.path.exists(os.path.join(a.res, f))], ignore_index=True)
    T = tables(m, a.res)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for k, v in T.items():
        print(f"\n== {k}\n", v.round(4).to_string())
    fig_acquisition(a.data, a.fig)
    fig_orders(a.data, a.res, a.fig)
    fig_accuracy_coverage(m, a.fig)
    fig_silent_failure(m, a.fig)
    preds = pd.concat([pd.read_csv(os.path.join(a.res, f)) for f in ("predictions.csv.gz", "predictions_phy3.csv.gz")
                       if os.path.exists(os.path.join(a.res, f))], ignore_index=True)
    sig = record_signatures(a.data, a.res)
    fig_signature(preds, sig, a.fig, a.res)
    print(pd.read_csv(os.path.join(a.res, "table_record_signature.csv")).round(3).to_string())


if __name__ == "__main__":
    main()
