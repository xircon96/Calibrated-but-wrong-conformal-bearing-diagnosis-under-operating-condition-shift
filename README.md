# Calibrated but wrong: conformal bearing diagnosis under operating-condition shift

Code and results for the draft paper of the same name (A. Gad, 2026). The benchmark measures whether
split-conformal prediction sets for vibration-based bearing diagnosis keep their coverage when the
motor load or the defect size changes, on the CWRU 12 kHz drive-end data, with sample-disjoint
splits, an acquisition-chain audit and a per-record physics audit.

## Layout

| Path | What it does |
| --- | --- |
| `pacb/data.py` | Loads the 40 CWRU records, checks every file against its original record number, decimates the 48 kHz normal baseline, applies the common 4.8 kHz low-pass, cuts windows |
| `pacb/features.py` | Time statistics, log spectrum, kurtosis-selected squared-envelope spectrum, order-normalised EOS, 23 physics-anchored features (PAF), 3 physics scores (PHY3) |
| `pacb/models.py` | Random forest and logistic pipelines; WDCNN-style 1D CNN in JAX |
| `pacb/protocols.py` | P1 random (leaky), P2 time-block, P3 cross-load (12 pairs), P4 cross-size (3 folds) |
| `pacb/conformal.py` | Split conformal (LAC) and covariate-shift weighted conformal |
| `pacb/shift.py` | Domain-classifier likelihood ratios, Mahalanobis atypicality |
| `run_experiments.py` | Runs every protocol x model x seed; writes `results/metrics*.csv`, `confusion*.csv`, `predictions*.csv.gz` |
| `audit_acquisition.py` | Normal-vs-fault separability from band shape, raw and harmonised |
| `edge_cost.py` | Parameters, operations and single-thread latency |
| `analyse.py`, `fig1_pipeline.py` | Paper tables (`results/table_*.csv`) and figures (`figures/`) |
| `figures_retro.py` | All six paper figures in the journal style (serif, black/red/blue, full frames, inward ticks) |
| `export_figures.py` | Draws every figure via `figures_retro.py` and saves PNG, 300 dpi LZW TIFF and vector PDF |

## Reproduce

```bash
pip install numpy scipy scikit-learn pandas matplotlib "jax[cpu]" optax
git clone https://github.com/XiongMeijing/CWRU-1.git   # public mirror; files are verified on load
DATA=CWRU-1/Data
python audit_acquisition.py --data $DATA
python run_experiments.py --data $DATA --seeds 3                      # about 25 min on 2 CPU cores
python run_experiments.py --data $DATA --seeds 3 --models PHY3-LR --protocols P4 --tag _phy3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
python edge_cost.py --data $DATA
python analyse.py --data $DATA && python export_figures.py $DATA
```

Versions used: Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2,
JAX 0.11.2, Optax 0.2.8. Logistic models are deterministic; the random forest and CNN use seeds 0, 1, 2.
CNN results can differ slightly across JAX versions and CPUs.

## Headline numbers (from `results/`)

* In-domain (P2): all models reach at least 0.90 conformal coverage.
* Cross-load (P3) worst-pair coverage: 0.60 (TS-RF) to 0.80 (CNN).
* Unseen size (P4) worst-fold coverage: 0.28 (PHY3-LR) to 0.47 (EOS-LR).
* OR 0.021 in records (17 dB BPFO signature): CNN 0 % correct, PHY3-LR 88 % under unseen size.

Licence: to be chosen by the author before public release.
