
#!/usr/bin/env python3
"""Visual check of clue-board geometry, and collector of real labelled faces.
 
  python3 tools/clue_overlay.py --synthetic [--banner PATH]          # fake boards, prints corner error
  python3 tools/clue_overlay.py --images ~/clue_data/frames          # saved frames (png/jpg)
  python3 tools/clue_overlay.py --live /left_camera/image_raw        # ROS topic ('s' saves a raw frame)
  python3 tools/clue_overlay.py --live /left_camera/image_raw --autosave
        # saves every detected face (rectified) into ~/clue_data/real/<camera>_<time>/pass_NNN/
        # While a board is in view press 1-8 = its location id (row in clues.csv, i.e. the
        # clue type printed on the board) to tag the current pass.
 
Keys: space/any = next (images, synthetic), s = save raw frame (live), 1-8 = tag pass, q = quit.
Debug images go to ~/clue_data/debug/. Images and synthetic modes need no ROS.
"""
import argparse
import glob
import os
import shutil
import sys
import time
 
import cv2
import numpy as np
 
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from controller_pkg.clue import geometry as g  # noqa: E402
 
DEBUG_DIR = os.path.expanduser("~/clue_data/debug")
FRAME_DIR = os.path.expanduser("~/clue_data/frames")
REAL_DIR = os.path.expanduser("~/clue_data/real")
DEFAULT_CLUES = "/home/fizzer/2026_competition_ws/install/world_pkg/share/world_pkg/world/clues.csv"
PASS_GAP_S = 2.0          # no face for this long = the next board is a new pass
COL = {"type": (0, 0, 255), "value": (255, 0, 255)}
 
 
def annotate(bgr):
    """Return (frame with faces drawn, [rectified texture with cell grid per face], faces)."""
    vis = bgr.copy()
    y0 = int(bgr.shape[0] * g.ROI_TOP)
    cv2.line(vis, (0, y0), (vis.shape[1], y0), (255, 255, 0), 1)
    faces = g.detect_faces(bgr)
    tiles = []
    for f in faces:
        cv2.polylines(vis, [f.quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(vis, f"w={f.width:.0f}px", tuple(f.quad[0].astype(int) - (0, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tex = g.face_to_texture(bgr, f.quad)
        for name, _, x0, y0_, x1, y1 in g.cell_boxes():
            cv2.rectangle(tex, (x0, y0_), (x1, y1), COL[name], 1)
        tiles.append(tex)
    return vis, tiles, faces
 
 
def show(name, bgr, headless, wait=True, status=""):
    vis, tiles, faces = annotate(bgr)
    if status:
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if name != "live":                                   # live mode would write 10 files/s
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_frame.png"), vis)
        for i, t in enumerate(tiles):
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_tex{i}.png"), t)
    if headless:
        return faces, 0
    cv2.imshow("frame", vis)
    if tiles:
        cv2.imshow("rectified + cells", cv2.resize(tiles[0], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(0 if wait else 1) & 0xFF
    return faces, key
 
 
class PassSaver:
    """Saves the best face of every frame into pass folders; a digit key tags the current pass."""
 
    def __init__(self, root, cam, clues_path):
        self.dir = os.path.join(root, f"{cam}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.dir)
        if os.path.exists(clues_path):
            shutil.copy(clues_path, os.path.join(self.dir, "clues.csv"))
        else:
            print(f"WARNING: {clues_path} not found. Copy this launch's clues.csv into {self.dir} yourself.")
        self.n, self.k, self.sign, self.cur, self.last_t = -1, 0, 0, None, -1e9
        print(f"autosaving to {self.dir}")
 
    def _base(self):
        return os.path.join(self.dir, f"pass_{self.n:03d}")
 
    def add(self, tex, width, t):
        if t - self.last_t > PASS_GAP_S or self.cur is None:
            self.n, self.k, self.sign = self.n + 1, 0, 0
            self.cur = self._base()
            os.makedirs(self.cur)
        self.last_t = t
        cv2.imwrite(os.path.join(self.cur, f"{self.k:04d}_w{int(width)}.png"), tex)
        self.k += 1
 
    def tag(self, sign):
        if self.cur is None:
            return
        new = f"{self._base()}_sign{sign}"
        if new != self.cur:
            os.rename(self.cur, new)
            self.cur, self.sign = new, sign
        print(f"pass {self.n} tagged as location {sign}")
 
    def status(self):
        tag = f"sign {self.sign}" if self.sign else "UNTAGGED (press 1-8)"
        return f"pass {self.n}: {self.k} faces, {tag}" if self.cur else "waiting for a board"
 
 
def run_images(path, headless):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(path), "*.png")) +
                   glob.glob(os.path.join(os.path.expanduser(path), "*.jpg")))
    if not files:
        sys.exit(f"no png/jpg files in {path}")
    for fn in files:
        faces, key = show(os.path.splitext(os.path.basename(fn))[0], cv2.imread(fn), headless)
        print(f"{os.path.basename(fn)}: {len(faces)} face(s), widths {[round(f.width) for f in faces]}")
        if key == ord("q"):
            break
 
 
def run_synthetic(headless, banner_path):
    import synth_board as sb  # sibling file in tools/ (script dir is on sys.path)
    adv, bbox, real = sb.font_metrics()
    print(f"font: {'Ubuntu Mono' if real else 'FALLBACK (metrics invalid!)'}; advance(M) = {adv:.1f} px "
          f"(geometry assumes {g.CELL_W}); bbox(H) relative to draw.text y = {bbox} "
          f"(geometry assumes rows {g.CELL_Y_OFF}..{g.CELL_Y_OFF + g.CELL_H}, so glyph must sit inside)")
    banner = cv2.imread(os.path.expanduser(banner_path)) if banner_path else None
    face = sb.render_face("CRIME", "SABOTAGE 7", banner)
    print(f"{'d(m)':>5} {'phi':>4} {'found':>6} {'width':>6} {'corner err px':>14}")
    for d in (0.2, 0.3, 0.45, 0.6, 0.8):
        for phi in (0, 30, 50):
            img, true_q = sb.render_view(face, d=d, phi_deg=phi, seed=1)
            faces, key = show(f"synth_d{d}_phi{phi}", img, headless, wait=True)  # any key = next
            if faces:
                best = min(faces, key=lambda f: np.abs(f.quad - true_q).mean())
                err = np.linalg.norm(best.quad - true_q, axis=1).mean()
                print(f"{d:5.2f} {phi:4d} {len(faces):6d} {best.width:6.0f} {err:14.2f}")
            else:
                print(f"{d:5.2f} {phi:4d} {0:6d} {'-':>6} {'-':>14}")
            if key == ord("q"):
                return
 
 
def run_live(topic, headless, autosave, clues):
    import rclpy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node("clue_overlay")
    br, latest = CvBridge(), {}
    node.create_subscription(Image, topic, lambda m: latest.update(f=br.imgmsg_to_cv2(m, "bgr8")), 1)
    os.makedirs(FRAME_DIR, exist_ok=True)
    cam = topic.strip("/").split("/")[0]
    saver = PassSaver(REAL_DIR, cam, os.path.expanduser(clues)) if autosave else None
    if saver and headless:
        print("note: --headless has no window, so passes cannot be tagged with the 1-8 keys")
    t0, got = time.time(), False
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        if "f" not in latest:
            if not got and time.time() - t0 > 3:
                print(f"No frames on {topic} after 3 s. Is the simulation running and is the topic "
                      f"bridged? Check with: ros2 topic list | grep camera")
                t0 = time.time()
            continue
        got = True
        frame = latest.pop("f")
        faces, key = show("live", frame, headless, wait=False, status=saver.status() if saver else "")
        if saver and faces:
            best = max(faces, key=lambda f: f.width)
            saver.add(g.face_to_texture(frame, best.quad), best.width, time.time())
        if key == ord("s"):
            fn = os.path.join(FRAME_DIR, f"{cam}_{int(time.time() * 1000)}.png")
            cv2.imwrite(fn, frame)
            print("saved", fn)
        elif saver and ord("1") <= key <= ord("8"):
            saver.tag(key - ord("0"))
        elif key == ord("q"):
            break
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--live")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--autosave", action="store_true", help="live: save detected faces into ~/clue_data/real/")
    ap.add_argument("--clues", default=DEFAULT_CLUES, help="clues.csv of the running simulation")
    ap.add_argument("--banner", help="clue_banner.png, to include the logo in synthetic faces")
    ap.add_argument("--headless", action="store_true", help="no windows, only write ~/clue_data/debug")
    a = ap.parse_args()
    if a.synthetic:
        run_synthetic(a.headless, a.banner)
    elif a.images:
        run_images(a.images, a.headless)
    elif a.live:
        run_live(a.live, a.headless, a.autosave, a.clues)
    else:
        ap.print_help()
 