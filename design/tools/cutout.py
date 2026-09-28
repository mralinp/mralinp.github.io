"""Cut the portrait out of its plain studio background, for the mugshot in design/index.html.

The background is a near-uniform light grey, so a flood fill from the frame edges finds it without
a segmentation model: pixels connected to the border that are light and low-saturation are
background. The mask is cleaned up (holes filled, small specks removed) and feathered so the hair
edge doesn't look cut with scissors.

    python3 design/tools/cutout.py assets/images/me.jpeg assets/images/me-cutout.png
"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGB")
a = np.asarray(img).astype(np.float32)

lum = a.mean(axis=2)
sat = a.max(axis=2) - a.min(axis=2)
# ponytail: fixed thresholds tuned for a light, even studio wall; a busy background needs a real
# segmentation model (e.g. rembg) instead.
# the wall is light grey (~235); the shirt collar is blown-out white (~254) and touches the wall at
# the shoulders, so pure white is excluded or the fill leaks through the collar and beheads the
# subject
light = (lum > 200) & (lum < 248) & (sat < 28)

# keep only light regions connected to the frame edge (the wall), not the white shirt collar
# open first so thin bridges (a collar tip touching the wall) break, then grow back into the light
# pixels: the wall stays one piece, the shirt stays with the person
core = ndimage.binary_opening(light, iterations=6)
labels, _ = ndimage.label(core)
edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
bg = np.isin(labels, edge[edge > 0])
bg = ndimage.binary_dilation(bg, iterations=6) & light

person = ~bg
person = ndimage.binary_opening(person, iterations=2)
person = ndimage.binary_fill_holes(person)
lab, n = ndimage.label(person)
if n > 1:  # drop specks, keep every real part of the person
    sizes = ndimage.sum(person, lab, range(1, n + 1))
    person = np.isin(lab, 1 + np.flatnonzero(sizes > 0.01 * person.size))

person = ndimage.binary_erosion(person, iterations=2)  # trim the wall-coloured fringe on the hair
alpha = Image.fromarray((person * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
out = img.copy()
out.putalpha(alpha)
out.save(dst, optimize=True)
print(f"{dst}: {person.mean():.0%} of the frame is the person")
