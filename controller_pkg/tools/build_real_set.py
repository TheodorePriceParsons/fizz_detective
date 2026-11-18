
#!/usr/bin/env python3
"""Turn the autosaved real passes into a labelled npz (the held-out test set).
 
Expected layout (made by `clue_overlay.py --live ... --autosave`):
  ~/clue_data/real/<camera>_<timestamp>/clues.csv
  ~/clue_data/real/<camera>_<timestamp>/pass_003_sign5/0000_w231.png ...
 
'sign5' is the location id 1-8 = row 5 of that launch's clues.csv = the text printed on that board.
Passes without a _signN tag are skipped. Each launch folder is one group, so you can hold out whole launches.
 
  python3 tools/build_real_set.py --out ~/clue_data/real_test.npz
"""
import argparse
import csv
import glob
import os
import re
import sys
 
import cv2
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from controller_pkg.clue import reader as r  # noqa: E402
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/clue_data/real")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
 
    X, Y, ROW, IDX, LAUNCH, WIDTH = [], [], [], [], [], []
    launches = sorted(d for d in glob.glob(os.path.join(os.path.expanduser(a.root), "*")) if os.path.isdir(d))
    for li, ldir in enumerate(launches):
        csv_path = os.path.join(ldir, "clues.csv")
        if not os.path.exists(csv_path):
            print(f"skip {ldir}: no clues.csv")
            continue
        with open(csv_path) as f:
            rows = [row for row in csv.reader(f) if len(row) >= 2][:8]
        n_img = 0
        for pdir in sorted(glob.glob(os.path.join(ldir, "pass_*_sign[1-8]"))):
            sign = int(re.search(r"_sign(\d)$", pdir).group(1))
            key, value = rows[sign - 1][0], rows[sign - 1][1]
            y = r.encode_clue(key, value)
            for fn in sorted(glob.glob(os.path.join(pdir, "*.png"))):
                tex = cv2.imread(fn)
                X.append(r.preprocess_face(tex))
                Y.append(y)
                ROW.append(np.repeat([0, 1], [6, 12]))
                IDX.append(np.r_[np.arange(6), np.arange(12)])
                m = re.search(r"_w(\d+)", fn)
                WIDTH.append(np.full(r.N_CELLS, int(m.group(1)) if m else 0))
                LAUNCH.append(np.full(r.N_CELLS, li))
                n_img += 1
        print(f"{os.path.basename(ldir)}: {n_img} tagged faces")
    if not X:
        sys.exit("no tagged faces found")
    out = os.path.expanduser(a.out)
    np.savez_compressed(out, x=np.concatenate(X), y=np.concatenate(Y), row=np.concatenate(ROW),
                        idx=np.concatenate(IDX), launch=np.concatenate(LAUNCH), width=np.concatenate(WIDTH))
    print(f"saved {out}: {len(X)} faces = {len(X) * r.N_CELLS} cells")
 
 
if __name__ == "__main__":
    main()
 