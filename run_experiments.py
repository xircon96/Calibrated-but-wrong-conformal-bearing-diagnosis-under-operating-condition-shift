"""Run every protocol x model x seed and write results/metrics.csv.

usage: python run_experiments.py --data <path to CWRU mirror Data folder> [--seeds 3] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from pacb.conformal import (lac_sets, lac_threshold, selective_accuracy, set_metrics,
                            weighted_lac_sets)
from pacb.data import load_cwru, make_windows
from pacb.features import envelope_features, fft_spectrum, physics_scores, time_stats
from pacb.models import CNN, forest, logreg
from pacb.protocols import block_split, cross_load, cross_severity, leaky_random
from pacb.shift import mahalanobis_score, shift_weights

N, STRIDE, ALPHA = 4096, 1024, 0.10
MIN_STEPS = 1500  # CNN optimiser steps per run, independent of training-set size


def featurise(W, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t = time.time()
    env = envelope_features(W.X, W.rpm)
    env_fixed = envelope_features(W.X, W.rpm, order_normalise=False)
    F = dict(ts=time_stats(W.X), fft=fft_spectrum(W.X), eos=env["eos"], paf=env["paf"],
             paf_fixed=env_fixed["paf"], eos_fixed=env_fixed["eos"])
    print(f"features: {time.time() - t:.1f} s")
    np.savez_compressed(cache, **F)
    return F


MODELS = {
    # name: (input key, factory(n_cls, seed))
    "TS-RF": ("ts", lambda k, s: forest(s)),
    "FFT-LR": ("fft", lambda k, s: logreg(0.1, s)),
    "EOS-LR": ("eos", lambda k, s: logreg(0.1, s)),
    "PAF-LR": ("paf", lambda k, s: logreg(1.0, s)),
    "PAF-LR (no order norm.)": ("paf_fixed", lambda k, s: logreg(1.0, s)),
    "CNN": ("raw", None),
    # three physics scores only; fault TYPE task (P4) only, since they carry no size information
    "PHY3-LR": ("phy3", lambda k, s: logreg(1.0, s)),
}
ONLY = {"PAF-LR (no order norm.)": ("P3-cross-load", "P4-cross-size"), "PHY3-LR": ("P4-cross-size",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="1 seed, few CNN steps (smoke test)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--protocols", nargs="*", default=None, help="subset: P1 P2 P3 P4")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [0] if a.quick else list(range(a.seeds))
    steps = 100 if a.quick else MIN_STEPS

    recs = load_cwru(a.data)
    W = make_windows(recs, N, STRIDE)
    reclen = np.array([len(r.x) for r in recs])
    F = featurise(W, os.path.join(a.out, "features.npz"))
    F["raw"] = W.X
    F["phy3"] = physics_scores(F["paf"])
    pd.DataFrame([dict(label=r.label, load=r.load, cwru_id=r.cwru_id, rpm=r.rpm, rpm_source=r.rpm_source,
                       n_samples=len(r.x), n_windows=int((W.rec == i).sum())) for i, r in enumerate(recs)]
                 ).to_csv(os.path.join(a.out, "records.csv"), index=False)

    rows, cms, preds, fitted = [], [], [], {}
    protos = [leaky_random, block_split, cross_load, cross_severity]
    if a.protocols:
        protos = [p for p, k in zip(protos, ("P1", "P2", "P3", "P4")) if k in a.protocols]
    for seed in seeds:
        for proto in protos:
            for sp in proto(W, reclen, N, seed):
                classes = sp["classes"]
                y = np.array([classes.index(v) for v in sp["y"]])
                transfer = "adapt" in sp
                shifts = {k: shift_weights(F[k], sp) for k in ("fft", "paf")} if transfer else {}
                md_paf = mahalanobis_score(F["paf"][sp["train"]], y[sp["train"]], F["paf"][sp["test"]]) if transfer else None
                for name, (key, factory) in MODELS.items():
                    if name in ONLY and sp["protocol"] not in ONLY[name]:
                        continue
                    if a.models and name not in a.models:
                        continue
                    X = F[key]
                    tr, ca, te = sp["train"], sp["cal"], sp["test"]
                    h = hashlib.md5(tr.tobytes() + str(classes).encode()).hexdigest()
                    ck = (name, seed, h)
                    t0 = time.time()
                    if ck not in fitted:
                        if name == "CNN":
                            spe = max(1, tr.sum() // 64)
                            m = CNN(len(classes), seed=seed, epochs=int(np.ceil(steps / spe))).fit(X[tr], y[tr])
                        else:
                            m = factory(len(classes), seed).fit(X[tr], y[tr])
                        fitted[ck] = m
                    m = fitted[ck]
                    fit_s = time.time() - t0
                    p_cal, p_te = m.predict_proba(X[ca]), m.predict_proba(X[te])
                    yhat = p_te.argmax(1)
                    q = lac_threshold(p_cal, y[ca], ALPHA)
                    S = lac_sets(p_te, q)
                    sm = set_metrics(S, y[te])
                    sm["selective_acc"] = selective_accuracy(S, yhat, y[te])
                    for space, (wc, we, auc) in shifts.items():
                        Sw, qw = weighted_lac_sets(p_cal, y[ca], wc, p_te, we, ALPHA)
                        smw = set_metrics(Sw, y[te])
                        sm.update({f"{k}_w{space}": v for k, v in smw.items()})
                        sm[f"selective_acc_w{space}"] = selective_accuracy(Sw, yhat, y[te])
                        sm[f"full_set_rate_w{space}"] = float(np.isinf(qw).mean())
                        sm[f"domain_auc_{space}"] = auc
                    err = (yhat != y[te]).astype(int)
                    if transfer and 0 < err.sum() < len(err):
                        sm["errdet_auc_msp"] = roc_auc_score(err, 1 - p_te.max(1))
                        sm["errdet_auc_md_paf"] = roc_auc_score(err, md_paf)
                        sm["errdet_auc_setsize"] = roc_auc_score(err, S.sum(1))
                    if sp["protocol"] != "P1-leaky":
                        idx = np.where(te)[0]
                        preds.append(pd.DataFrame(dict(protocol=sp["protocol"], fold=sp["fold"], model=name, seed=seed,
                                                       win=idx, rec=W.rec[idx], y=y[te], yhat=yhat,
                                                       pmax=p_te.max(1).round(4), set_size=S.sum(1),
                                                       covered=S[np.arange(len(idx)), y[te]])))
                    row = dict(protocol=sp["protocol"], fold=sp["fold"], model=name, seed=seed,
                               n_train=int(tr.sum()), n_cal=int(ca.sum()), n_test=int(te.sum()),
                               acc=accuracy_score(y[te], yhat),
                               macro_f1=f1_score(y[te], yhat, average="macro", labels=range(len(classes))),
                               cal_acc=accuracy_score(y[ca], p_cal.argmax(1)), qhat=q, fit_s=fit_s, **sm)
                    rows.append(row)
                    cm = confusion_matrix(y[te], yhat, labels=range(len(classes)))
                    cms.append(dict(protocol=sp["protocol"], fold=sp["fold"], model=name, seed=seed,
                                    classes="|".join(classes), cm=" ".join(map(str, cm.ravel()))))
                    print(f"[{sp['protocol']:14s} {sp['fold']:10s} s{seed}] {name:24s} acc={row['acc']:.4f} "
                          f"F1={row['macro_f1']:.4f} cov={sm['coverage']:.3f} |S|={sm['set_size']:.2f} "
                          f"covWfft={sm.get('coverage_wfft', float('nan')):.3f} covWpaf={sm.get('coverage_wpaf', float('nan')):.3f} "
                          f"|Swpaf|={sm.get('set_size_wpaf', float('nan')):.2f} ({fit_s:.1f}s)",
                          flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(a.out, f"metrics{a.tag}.csv"), index=False)
                pd.DataFrame(cms).to_csv(os.path.join(a.out, f"confusion{a.tag}.csv"), index=False)
            if preds:
                pd.concat(preds).to_csv(os.path.join(a.out, f"predictions{a.tag}.csv.gz"), index=False)
            # free CNN models from the finished protocol to bound memory
            fitted = {k: v for k, v in fitted.items() if k[0] != "CNN"}


if __name__ == "__main__":
    main()
