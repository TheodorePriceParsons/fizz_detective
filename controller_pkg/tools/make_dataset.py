
#!/usr/bin/env python3
"""Generate the synthetic training set: random boards -> side-camera view -> detect -> rectify -> cells.
 
  python3 tools/make_dataset.py --n 10000 --out ~/clue_data/synth_train.npz --seed 0
  python3 tools/make_dataset.py --n 1500  --out ~/clue_data/synth_val.npz   --seed 1
 
It runs the SAME detect_faces / face_to_texture / preprocess_face as the robot, so the data matches
what the net will see. Output npz: x (N,32,24) uint8, y (N,) class ids, row (0=type,1=value), idx (cell index).
"""
import argparse
import os
import sys
import time
 
import cv2
import numpy as np
 
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from controller_pkg.clue import geometry as g, reader as r  # noqa: E402
import synth_board as sb  # noqa: E402
 
CHARS = r.LABELS[:36]
 
 
def rand_text(rng, n_max, p_space=0.12):
    n = int(rng.integers(1, n_max + 1))
    s = [CHARS[i] for i in rng.integers(0, 36, n)]
    for i in range(1, n - 1):
        if rng.random() < p_space:
            s[i] = " "
    return "".join(s)
 
 
def rand_type(rng):
    return r.TYPES[int(rng.integers(0, 8))] if rng.random() < 0.5 else rand_text(rng, 6, 0.0)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="number of boards to render")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--banner", help="clue_banner.png (adds the logo; optional)")
    a = ap.parse_args()
 
    adv, bbox, real = sb.font_metrics()
    if not real:
        sys.exit("Ubuntu Mono not found: refusing to build a dataset with the wrong font.")
    banner = cv2.imread(os.path.expanduser(a.banner)) if a.banner else None
    rng = np.random.default_rng(a.seed)
    X, Y, ROW, IDX = [], [], [], []
    miss = bad = kept = 0
    t0 = time.time()
    for k in range(a.n):
        t, v = rand_type(rng), rand_text(rng, 12)
        face = sb.render_face(t, v, banner)
        d = rng.uniform(0.25, 0.65)
        bearing = np.deg2rad(rng.uniform(-35, 35))
        img, true_q = sb.render_view(
            face, d=d, phi_deg=rng.uniform(-65, 65), lateral=d * np.tan(bearing),
            light=rng.uniform(0.4, 0.8), noise=rng.uniform(1.0, 4.0), blur=rng.uniform(0.3, 1.2),
            pitch_deg=rng.uniform(8, 12), seed=int(rng.integers(1 << 31)))
        faces = g.detect_faces(img)
        if not faces:
            miss += 1
            continue
        best = max(faces, key=lambda f: f.width)
        if np.linalg.norm(best.quad - true_q, axis=1).mean() > 4:
            bad += 1
            continue
        cells = r.preprocess_face(g.face_to_texture(img, best.quad))
        y = r.encode_clue(t, v)
        X.append(cells)
        Y.append(y)
        ROW.append(np.repeat([0, 1], [6, 12]))
        IDX.append(np.r_[np.arange(6), np.arange(12)])
        kept += 1
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{a.n}  kept {kept}  not detected {miss}  bad corners {bad}  ({time.time() - t0:.0f}s)")
    if not X:
        sys.exit("nothing kept: detection failed on every sample")
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, x=np.concatenate(X), y=np.concatenate(Y),
                        row=np.concatenate(ROW), idx=np.concatenate(IDX))
    print(f"saved {out}: {kept} boards = {kept * r.N_CELLS} cells "
          f"(not detected {miss}, bad corners {bad})")
 
 
if __name__ == "__main__":
    main()
 