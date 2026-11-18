
#!/usr/bin/env python3
"""Visual check of clue-board geometry. Dev tool: draws detected faces and the 18-cell grid.
 
  python3 tools/clue_overlay.py --synthetic [--headless]     # known geometry, prints corner error
  python3 tools/clue_overlay.py --images ~/clue_data/frames  # saved frames (png/jpg)
  python3 tools/clue_overlay.py --live /left_camera/image_raw  # ROS topic; 's' saves a raw frame
 
Keys: space/any = next, s = save, q = quit.  Output images go to ~/clue_data/debug/.
Images and synthetic modes need no ROS.
"""
import argparse
import glob
import os
import sys
import time
 
import cv2
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from controller_pkg.clue import geometry as g  # noqa: E402
 
DEBUG_DIR = os.path.expanduser("~/clue_data/debug")
FRAME_DIR = os.path.expanduser("~/clue_data/frames")
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
 
 
def show(name, bgr, headless, wait=True):
    vis, tiles, faces = annotate(bgr)
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
            faces, key = show(f"synth_d{d}_phi{phi}", img, headless, wait=True)  # press any key for next
            if faces:
                best = min(faces, key=lambda f: np.abs(f.quad - true_q).mean())
                err = np.linalg.norm(best.quad - true_q, axis=1).mean()
                print(f"{d:5.2f} {phi:4d} {len(faces):6d} {best.width:6.0f} {err:14.2f}")
            else:
                print(f"{d:5.2f} {phi:4d} {0:6d} {'-':>6} {'-':>14}")
            if key == ord("q"):
                return
 
 
def run_live(topic, headless):
    import rclpy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node("clue_overlay")
    br, latest = CvBridge(), {}
    node.create_subscription(Image, topic, lambda m: latest.update(f=br.imgmsg_to_cv2(m, "bgr8")), 1)
    os.makedirs(FRAME_DIR, exist_ok=True)
    t0, warned = time.time(), False
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        if "f" not in latest:
            if not warned and time.time() - t0 > 3 and not latest:
                print(f"No frames on {topic} after 3 s. Is the simulation running and is the topic "
                      f"bridged? Check with: ros2 topic list | grep camera")
                warned = True
            continue
        warned = True
        frame = latest.pop("f")
        _, key = show("live", frame, headless, wait=False)
        if key == ord("s"):
            fn = os.path.join(FRAME_DIR, f"{topic.strip('/').replace('/', '_')}_{int(time.time() * 1000)}.png")
            cv2.imwrite(fn, frame)
            print("saved", fn)
        elif key == ord("q"):
            break
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--live")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--banner", help="clue_banner.png, to include the logo in synthetic faces")
    ap.add_argument("--headless", action="store_true", help="no windows, only write ~/clue_data/debug")
    a = ap.parse_args()
    if a.synthetic:
        run_synthetic(a.headless, a.banner)
    elif a.images:
        run_images(a.images, a.headless)
    elif a.live:
        run_live(a.live, a.headless)
    else:
        ap.print_help()
 