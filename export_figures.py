"""Draw every figure (figures_retro.py style) and save PNG, 300 dpi LZW TIFF and vector PDF copies."""
import os
import sys

from matplotlib.figure import Figure

_orig = Figure.savefig


def savefig(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    root, _ = os.path.splitext(str(fname))
    kk = {kx: v for kx, v in k.items() if kx in ("bbox_inches",)}
    _orig(self, root + ".pdf", **kk)
    _orig(self, root + ".tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"}, **kk)


Figure.savefig = savefig
sys.argv = [sys.argv[0], "--data", sys.argv[1]]
import figures_retro
figures_retro.main()
