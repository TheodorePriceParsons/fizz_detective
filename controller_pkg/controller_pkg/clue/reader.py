
"""Character reading: cell preprocessing, label encoding, ONNX model wrapper.
 
The SAME preprocessing is used for training data (tools/make_dataset.py, tools/build_real_set.py)
and at runtime (CellNet.predict), so they cannot drift apart. Never edit it without regenerating
the datasets and retraining.
 
Classes: 0-25 = A-Z, 26-35 = 0-9, 36 = blank (space / nothing).
Model input : float32 (N, 32, 24, 1) in [0, 1]   (NHWC, height 32, width 24)
Model output: float32 (N, 37) softmax probabilities
"""
import cv2
import numpy as np
 
from . import geometry as g
 
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
BLANK = 36
N_CLASSES = 37
OUT_H, OUT_W = 32, 24
N_CELLS = 18                                   # 6 type cells then 12 value cells
 
# Order from the generator docstring (clue id = index + 1). Verify against clues.csv before relying on it.
TYPES = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]
 
 
def preprocess_cell(cell_bgr):
    """BGR cell -> uint8 (32, 24). Red channel (blue ink is dark there, the face is bright),
    divided by the cell's own median so lighting cancels: ink -> ~0, face -> ~1."""
    red = cell_bgr[..., 2].astype(np.float32)
    x = np.clip(red / (np.median(red) + 1.0), 0.0, 1.0)
    x = cv2.resize(x, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return np.round(x * 255).astype(np.uint8)
 
 
def preprocess_face(tex):
    """Rectified face -> uint8 (18, 32, 24): 6 type cells followed by 12 value cells."""
    cells = g.slice_cells(tex)
    return np.stack([preprocess_cell(c) for c in cells["type"] + cells["value"]])
 
 
def encode_row(text, n):
    """'D SPIN' -> n class ids (anything that is not A-Z / 0-9, and anything past the end, is blank)."""
    text = text.upper()[:n].ljust(n)
    return [LABELS.index(c) if c.isalnum() and c in LABELS else BLANK for c in text]
 
 
def encode_clue(type_text, value_text):
    """Labels for all 18 cells of a board."""
    return np.array(encode_row(type_text, 6) + encode_row(value_text, 12), np.int16)
 
 
def decode(ids):
    return "".join(LABELS[i] for i in ids)
 
 
class CellNet:
    """ONNX model wrapper. One forward pass for all cells of a face."""
 
    def __init__(self, model_path):
        import onnxruntime as ort                      # imported here so the rest works without it
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1                  # do not steal CPU from the simulator
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
 
    def predict_arrays(self, arr_u8):
        """arr_u8: (N, 32, 24) uint8 from preprocess_face -> (N, 37) probabilities."""
        x = (arr_u8.astype(np.float32) / 255.0)[..., None]
        return self.sess.run(None, {self.inp: x})[0]
 
    def predict_face(self, tex):
        """Rectified face -> (18, 37) probabilities."""
        return self.predict_arrays(preprocess_face(tex))
 