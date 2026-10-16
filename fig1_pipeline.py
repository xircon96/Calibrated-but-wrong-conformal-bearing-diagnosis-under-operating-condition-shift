"""Figure 1: pipeline schematic."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, GREY, ACC, LIGHT = "#0b0b0b", "#52514e", "#9a9893", "#2a78d6", "#eef4fc"
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, text, accent=False, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=LIGHT if accent else "white", ec=ACC if accent else GREY, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))


box(0.5, 33, 14, 12, "CWRU record\n12 kHz drive end\n(record number\nverified)")
box(18, 33, 14, 12, "Harmonise\n48→12 kHz (normal)\n4.8 kHz low-pass\n(all records)", accent=True)
box(35.5, 33, 13, 12, "Windows\nN = 4096\nstride 1024")
arrow(15.2, 39, 17.6, 39); arrow(32.6, 39, 35.1, 39)
# representations
reps = [("Time statistics (10)\n→ random forest", 41.5), ("Log spectrum (256)\n→ logistic reg.", 34.5),
        ("Kurtosis band → SES →\norder grid: EOS / PAF / PHY3\n→ logistic regression", 25.5), ("Raw window\n→ 1D CNN", 17)]
for t, y in reps:
    box(53, y, 20, 6 if "Kurtosis" not in t else 8, t, accent="Kurtosis" in t, fs=6.8)
    arrow(48.8, 39, 52.6, y + (3 if "Kurtosis" not in t else 4))
box(77, 30, 22, 14, "Split conformal (α = 0.1)\nthreshold from source\ncalibration block\n→ prediction set", accent=False)
for _, y in reps:
    arrow(73.4, y + 3, 76.6, 37)
# shift diagnostics
box(77, 9, 22, 15, "Label-free shift checks\n(unlabelled target pool)\ndomain AUC, weighted\nconformal, Mahalanobis", fs=6.8)
arrow(88, 23.8, 88, 29.6)
# protocols strip
ax.text(0.5, 26, "Protocols", fontsize=7.5, color=INK, weight="bold")
for i, t in enumerate(["P1 random windows (leaky)", "P2 time blocks (in-domain)",
                       "P3 load a → load b (12 pairs)", "P4 unseen defect size (3 folds)"]):
    ax.text(0.5, 21.5 - 4.2 * i, t, fontsize=6.9, color=INK2)
ax.text(0.5, 2.5, "Physics audit: envelope signature at the expected order of each record vs record accuracy",
        fontsize=6.9, color=INK2)
fig.tight_layout(pad=0.2)
fig.savefig("figures/fig1_pipeline.png", dpi=300)
