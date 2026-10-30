"""Synthetic clue banners rendered exactly like the competition's clueboard_generator.py
(same template, UbuntuMono-R 90 pt, blue text, type at (250,30), value at (30,250)).

Outputs data/banners/*.png + data/labels.csv, and data/chars/<C>/*.png crops on the
monospace grid (the same crop function is reused on rectified boards at runtime).

    python3 -I gen_clue_data.py --n 2000
"""
import argparse, csv, random, string
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf'   # same as the sim
TYPES = ['SIZE', 'VICTIM', 'CRIME', 'TIME', 'PLACE', 'MOTIVE', 'WEAPON', 'BANDIT']
ALPHABET = string.ascii_uppercase + string.digits
TYPE_XY, VALUE_XY, SIZE = (250, 30), (30, 250), 90


def load_font(path):
    try:
        return ImageFont.truetype(path, SIZE)
    except OSError:
        print(f'!! {path} missing, using DejaVu fallback (install fonts-ubuntu for real data)')
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', SIZE)


def random_value(max_len=12):
    """Non-dictionary words with spaces, <= 12 chars total."""
    n = random.randint(1, max_len)
    s = ''.join(random.choice(ALPHABET) for _ in range(n))
    for _ in range(random.randint(0, 2)):            # sprinkle spaces, never leading/trailing
        if len(s) > 2:
            i = random.randint(1, len(s) - 2)
            s = s[:i] + ' ' + s[i + 1:]
    return s


def render(template, font, ctype, value):
    img = Image.fromarray(template.copy())
    d = ImageDraw.Draw(img)
    # Same quirk as the official script: (255,0,0) drawn on a BGR array => text is BLUE on the board.
    d.text(TYPE_XY, ctype, (255, 0, 0), font=font)
    d.text(VALUE_XY, value, (255, 0, 0), font=font)
    return np.array(img)


def char_boxes(font, origin, n):
    """Pixel boxes (x0,y0,x1,y1) for n monospace slots starting at origin."""
    adv = font.getlength('M')
    asc, desc = font.getmetrics()
    x, y = origin
    return [(int(x + i * adv), y, int(x + (i + 1) * adv), y + asc + desc) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--font', default=FONT)
    ap.add_argument('--out', default=str(HERE / 'data'))
    a = ap.parse_args()
    out = Path(a.out); (out / 'banners').mkdir(parents=True, exist_ok=True)
    font = load_font(a.font)
    import cv2
    template = cv2.imread(str(HERE / 'clue_banner.png'))
    with open(out / 'labels.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['file', 'type', 'value'])
        for k in range(a.n):
            ctype, value = random.choice(TYPES), random_value()
            img = render(template, font, ctype, value)
            name = f'{k:05d}.png'
            cv2.imwrite(str(out / 'banners' / name), img)
            w.writerow([name, ctype, value])
            for j, (box, ch) in enumerate(zip(char_boxes(font, VALUE_XY, len(value)), value)):
                if ch == ' ' or box[2] > img.shape[1]:   # fallback font can overflow
                    continue
                d = out / 'chars' / ch; d.mkdir(parents=True, exist_ok=True)
                x0, y0, x1, y1 = box
                cv2.imwrite(str(d / f'{k:05d}_{j}.png'), img[y0:y1, x0:x1])
    print(f'wrote {a.n} banners to {out}; char pitch {font.getlength("M"):.1f}px')


if __name__ == '__main__':
    main()
