"""Edge-cost profile: parameters, stored bytes, analytic multiply-accumulates and measured
single-window latency on one CPU thread. Run with no other jobs on the machine.

usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
       python edge_cost.py --data <CWRU Data folder>
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd

from pacb.data import load_cwru, make_windows
from pacb.features import candidate_bands, envelope_features, fft_spectrum, time_stats
from pacb.models import CNN, cnn_macs, forest, logreg, n_params


def bench(fn, reps=200, warm=10):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t)), 1e3 * float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/edge_cost.csv")
    a = ap.parse_args()
    W = make_windows(load_cwru(a.data))
    y = np.array([sorted(set(W.label)).index(v) for v in W.label])
    rng = np.random.default_rng(0)
    tr = rng.random(len(y)) < 0.5
    x1, r1 = W.X[:1], W.rpm[:1]
    env = envelope_features(W.X, W.rpm)
    feats = dict(ts=time_stats(W.X), fft=fft_spectrum(W.X), eos=env["eos"], paf=env["paf"])
    N, K = W.X.shape[1], len(candidate_bands())
    rows = []

    def lr_row(name, key, fe, C, fe_flops):
        m = logreg(C).fit(feats[key][tr], y[tr])
        lr = m[-1]
        p = lr.coef_.size + lr.intercept_.size + 2 * feats[key].shape[1]  # + scaler mean/scale
        xf = feats[key][:1]
        fe_ms = bench(fe)
        cl_ms = bench(lambda: m.predict_proba(xf))
        rows.append(dict(model=name, params=p, bytes_fp32=4 * p,
                         macs_classifier=lr.coef_.size, feature_flops_est=fe_flops,
                         feat_ms_median=fe_ms[0], feat_ms_p95=fe_ms[1],
                         clf_ms_median=cl_ms[0], clf_ms_p95=cl_ms[1]))

    fft_flops = lambda n: 5 * n * np.log2(n)  # radix-2 estimate per complex FFT
    lr_row("FFT-LR", "fft", lambda: fft_spectrum(x1), 0.1, fft_flops(N) / 2 + 2 * N)
    lr_row("EOS-LR", "eos", lambda: envelope_features(x1, r1), 0.1, (2 + K) * fft_flops(N))
    lr_row("PAF-LR", "paf", lambda: envelope_features(x1, r1), 1.0, (2 + K) * fft_flops(N))

    rf = forest(0).fit(feats["ts"][tr], y[tr])
    nodes = sum(e.tree_.node_count for e in rf.estimators_)
    fe_ms = bench(lambda: time_stats(x1))
    rf.set_params(n_jobs=1)
    cl_ms = bench(lambda: rf.predict_proba(feats["ts"][:1]), reps=50)
    depth = np.mean([e.tree_.max_depth for e in rf.estimators_])
    # a node stores feature id, threshold, two children and (leaves) class counts: ~4 values per node
    rows.append(dict(model="TS-RF", params=int(nodes), bytes_fp32=int(4 * 4 * nodes),
                     macs_classifier=int(len(rf.estimators_) * depth), feature_flops_est=12 * N,
                     feat_ms_median=fe_ms[0], feat_ms_p95=fe_ms[1], clf_ms_median=cl_ms[0], clf_ms_p95=cl_ms[1]))

    cnn = CNN(10, seed=0, epochs=1).fit(W.X[tr][:640], y[tr][:640])
    p = n_params(cnn.params)
    cl_ms = bench(lambda: cnn.predict_proba(x1))
    rows.append(dict(model="CNN", params=p, bytes_fp32=4 * p, macs_classifier=cnn_macs(N, 10),
                     feature_flops_est=0, feat_ms_median=0.0, feat_ms_p95=0.0,
                     clf_ms_median=cl_ms[0], clf_ms_p95=cl_ms[1]))
    df = pd.DataFrame(rows)
    df["total_ms_median"] = df.feat_ms_median + df.clf_ms_median
    df["window_ms"] = 1e3 * N / 12000
    df["host"] = f"{platform.machine()} {platform.processor() or ''} {os.cpu_count()} vCPU, 1 thread".strip()
    print(df.to_string())
    df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
