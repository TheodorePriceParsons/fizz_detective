"""Clue-board geometry: find a board face in a camera frame and cut it into character cells.
 
Pure OpenCV/numpy. No ROS, no neural net, so everything here can be run on a saved image.
Texture-space constants come from world_pkg/world/clueboard_scripts/clueboard_generator.py.
 
Pipeline (one frame):
    detect_faces(bgr)            -> [Face]      closed blue ring -> 4 corners of the grey face
    face_to_texture(bgr, quad)   -> image       face warped to the 600x400 banner frame (x TEX_SCALE)
    slice_cells(tex)             -> {'type': [..6 cells], 'value': [..12 cells]}
 
cell_boxes() is the single source of truth for the grid: the slicer, the overlay tool and
(later) the training-data generator must all use it so they cannot drift apart.
"""
from dataclasses import dataclass
 
import cv2
import numpy as np
 
# ---------------------------------------------------------------- colour
# Same blue range as driver.count_boards (OpenCV HSV, H 0-180).
BLUE_LO = (105, 150, 50)
BLUE_HI = (130, 255, 255)
_K3 = np.ones((3, 3), np.uint8)
 
# ---------------------------------------------------------------- texture space
TEX_W, TEX_H = 600, 400          # clue_banner.png size == printed face
TEX_SCALE = 0.5                  # rectified image is TEX_SCALE x the texture
_FACE_DST = np.float32([[0, 0], [TEX_W, 0], [TEX_W, TEX_H], [0, TEX_H]])
 
# Ubuntu Mono, size 90: advance = 0.5 em = 45 px. CELL_Y_OFF / CELL_H are the glyph rows
# relative to the y passed to PIL's draw.text. They are PLACEHOLDERS: run
# `tools/clue_overlay.py --synthetic` and compare with the measured font metrics it prints.
CELL_W = 45
CELL_Y_OFF = 5
CELL_H = 80
CELL_PAD = 0.1                   # extra width on each side of a cell, as a fraction of CELL_W
 
# row name -> (x origin, y origin, number of cells)
ROWS = {"type": (250, 30, 6), "value": (30, 250, 12)}
 
# ---------------------------------------------------------------- detection
ROI_TOP = 0.35                   # ignore the top 35% of the frame (face sits below the optical axis)
MIN_FACE_W = 100                 # px, face width in the raw frame; below this it is not readable
APPROX_EPS = 0.03                # approxPolyDP tolerance, fraction of contour perimeter
 
 
@dataclass
class Face:
    quad: np.ndarray             # (4, 2) float32, TL TR BR BL, in raw-frame pixels
    width: float                 # mean of top/bottom edge length, px
    height: float                # mean of left/right edge length, px
 
 
def order_corners(p):
    """Sort 4 points into TL, TR, BR, BL."""
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()        # y - x
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)
 
 
def blue_fraction(bgr, roi_top=ROI_TOP, width=320):
    """Cheap gate: fraction of blue pixels in the lower part of a downscaled frame."""
    h, w = bgr.shape[:2]
    roi = bgr[int(h * roi_top):]
    small = cv2.resize(roi, (width, max(1, int(roi.shape[0] * width / w))), interpolation=cv2.INTER_AREA)
    mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), BLUE_LO, BLUE_HI)
    return cv2.countNonZero(mask) / float(mask.size)
 
 
def blue_mask(hsv):
    """Blue pixels, with a 3x3 close to heal sensor-noise gaps in the ring."""
    return cv2.morphologyEx(cv2.inRange(hsv, BLUE_LO, BLUE_HI), cv2.MORPH_CLOSE, _K3)
 
 
def find_faces(mask, offset=(0, 0), min_w=MIN_FACE_W):
    """Faces = 4-corner convex HOLES inside blue blobs.
 
    A board seen from the front is a blue ring around a grey face, so the face is a hole.
    The solid-blue back has no hole (rejected), and a ring cut by the image edge has no
    closed hole either (rejected), so no explicit edge test is needed.
    Letters are blue islands inside the hole; with RETR_CCOMP they are separate top-level
    contours and do not change the hole's outline.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    if hier is None:
        return faces
    off = np.float32(offset)
    for i, c in enumerate(cnts):
        if hier[0][i][3] == -1:           # top-level contour, not a hole
            continue
        approx = cv2.approxPolyDP(c, APPROX_EPS * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_corners(approx.reshape(4, 2).astype(np.float32))
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if w < min_w:
            continue
        faces.append(Face(q + off, float(w), float(h)))
    return faces
 
 
def detect_faces(bgr, roi_top=ROI_TOP, min_w=MIN_FACE_W):
    """Full-resolution detection on the lower part of the frame. Returns [Face] in raw coords."""
    y0 = int(bgr.shape[0] * roi_top)
    hsv = cv2.cvtColor(bgr[y0:], cv2.COLOR_BGR2HSV)
    return find_faces(blue_mask(hsv), offset=(0, y0), min_w=min_w)
 
 
def face_to_texture(bgr, quad, s=TEX_SCALE):
    """Warp the face quad into the banner texture frame (scaled by s)."""
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), _FACE_DST * s)
    return cv2.warpPerspective(bgr, M, (int(TEX_W * s), int(TEX_H * s)), flags=cv2.INTER_LINEAR)
 
 
def cell_boxes(s=TEX_SCALE):
    """[(row_name, index, x0, y0, x1, y1)] for all 18 cells, in the rectified image."""
    boxes = []
    for name, (ox, oy, n) in ROWS.items():
        y0 = int(round((oy + CELL_Y_OFF) * s))
        y1 = int(round((oy + CELL_Y_OFF + CELL_H) * s))
        for i in range(n):
            x0 = int(round((ox + (i - CELL_PAD) * CELL_W) * s))
            x1 = int(round((ox + (i + 1 + CELL_PAD) * CELL_W) * s))
            boxes.append((name, i, x0, y0, x1, y1))
    return boxes
 
 
def slice_cells(tex, s=TEX_SCALE):
    """Cut the rectified face into {'type': [6 cells], 'value': [12 cells]}."""
    out = {name: [] for name in ROWS}
    for name, _, x0, y0, x1, y1 in cell_boxes(s):
        out[name].append(tex[y0:y1, x0:x1])
    return out
 