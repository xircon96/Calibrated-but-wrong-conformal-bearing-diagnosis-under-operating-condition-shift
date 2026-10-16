"""Acquisition-chain audit: can the Normal class be recognised from spectral SHAPE in a band
that carries no bearing information? Leave-one-load-out logistic regression on the band's
log-power profile with its mean removed (amplitude-free). Run on raw and harmonised data.

usage: python audit_acquisition.py --data <CWRU Data folder>
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pacb.data import FS, load_cwru, make_windows

BANDS = [(5500, 6000), (4800, 5500), (4000, 4800), (2000, 4000), (100, 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/acquisition_audit.csv")
    a = ap.parse_args()
    rows = []
    for harmonised in (False, True):
        W = make_windows(load_cwru(a.data, harmonised=harmonised))
        X = W.X - W.X.mean(1, keepdims=True)
        S = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
        f = np.fft.rfftfreq(X.shape[1], 1 / FS)
        y = (W.label == "Normal").astype(int)
        hf = S[:, f >= 5500].sum(1) / S.sum(1)
        for lo, hi in BANDS:
            sel = (f >= lo) & (f < hi)
            P = np.log10(S[:, sel] + 1e-30)
            P -= P.mean(1, keepdims=True)
            rec, fa = [], []
            for L in range(4):
                tr, te = W.load != L, W.load == L
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(P[tr], y[tr])
                p = m.predict(P[te])
                rec.append(((p == 1) & (y[te] == 1)).sum() / (y[te] == 1).sum())
                fa.append(((p == 1) & (y[te] == 0)).sum() / (y[te] == 0).sum())
            rows.append(dict(harmonised=harmonised, band_hz=f"{lo}-{hi}", normal_recall=np.mean(rec),
                             false_alarm=np.mean(fa),
                             hf_ratio_normal=np.median(hf[y == 1]), hf_ratio_fault=np.median(hf[y == 0])))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
