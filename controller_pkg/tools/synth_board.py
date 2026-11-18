
"""Render a clue board the way a side camera sees it. Dev tool (not installed with the package).
 
Reproduces the sim's board: blue box 0.20 x 0.16 m, printed face 0.15 x 0.10 m (600x400 texture),
pure-blue text, white banner, all darkened by lighting. The texture is JPEG round-tripped like
the generator's cv2.imwrite. Later, the training-data script should import render_view from here.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
 
FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE = 90
TEX_W, TEX_H = 600, 400
M_PER_PX = 0.15 / TEX_W                       # face is 0.15 m wide
CANVAS = (-100, -40, 700, 440)                # blue box in texture px (x0, y0, x1, y1); bottom clipped by ground
FACE_CENTRE_Z = 0.06                          # world height of the face centre
CAM_Z = 0.105                                 # side camera height
 
 
def _font():
    try:
        return ImageFont.truetype(FONT, FONT_SIZE), True
    except OSError:
        return ImageFont.truetype(FALLBACK_FONT, FONT_SIZE), False
 
 
def font_metrics():
    """(advance of 'M', bbox of 'H' relative to the draw.text origin, is_real_font)."""
    f, real = _font()
    return f.getlength("M"), f.getbbox("H"), real
 
 
def render_face(type_word, value, banner=None):
    """BGR 600x400 texture as the generator would save it (JPEG round trip included)."""
    base = banner if banner is not None else np.full((TEX_H, TEX_W, 3), 255, np.uint8)
    pil = Image.fromarray(base[..., ::-1].copy())
    d, (font, _) = ImageDraw.Draw(pil), _font()
    d.text((250, 30), type_word, (0, 0, 255), font=font)     # RGB pure blue
    d.text((30, 250), value, (0, 0, 255), font=font)
    face = np.array(pil)[..., ::-1]
    return cv2.imdecode(cv2.imencode(".jpg", face)[1], cv2.IMREAD_COLOR)
 
 
def _project(pts_tex, d, phi, lateral, f, W, H):
    lx = (pts_tex[:, 0] - TEX_W / 2) * M_PER_PX
    ly = (pts_tex[:, 1] - TEX_H / 2) * M_PER_PX
    c, s = np.cos(phi), np.sin(phi)
    X, Y, Z = lateral + c * lx, (CAM_Z - FACE_CENTRE_Z) + ly, d - s * lx
    return np.float32(np.stack([f * X / Z + W / 2, f * Y / Z + H / 2], axis=1))
 
 
def render_view(face, d=0.3, phi_deg=0.0, lateral=0.0, light=0.55, noise=1.8, blur=0.6,
                W=960, H=540, f=572.0, seed=None):
    """Return (image BGR, true face quad TL TR BR BL in image px).
 
    d: distance along the optical axis to the face centre (m). phi_deg: board yaw relative to
    facing the camera. lateral: sideways offset of the face centre (m). noise: gaussian sigma in
    8-bit levels (rule minimum 0.007 is about 1.8).
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = CANVAS
    canvas = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    canvas[:] = (255, 0, 0)                                  # blue ring
    canvas[-y0:-y0 + TEX_H, -x0:-x0 + TEX_W] = face
    canvas = (canvas * light).astype(np.uint8)
 
    phi = np.deg2rad(phi_deg)
    box = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    src = np.float32([[0, 0], [x1 - x0, 0], [x1 - x0, y1 - y0], [0, y1 - y0]])
    M = cv2.getPerspectiveTransform(src, _project(box, d, phi, lateral, f, W, H))
    img = cv2.warpPerspective(canvas, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 110, 40))
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    quad = _project(np.float32([[0, 0], [TEX_W, 0], [TEX_W, TEX_H], [0, TEX_H]]), d, phi, lateral, f, W, H)
    return img, quad
 