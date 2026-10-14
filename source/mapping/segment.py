"""Floor / sky / wall segmentation of one frame and the floor-wall boundary per column
(stage 2). Pipeline code: it uses only constants measured once on the simulator (palette,
far-plane rule), never ground truth.

Palette (BGR), measured in stage 0: blue tile (218, 181, 157) drifting to about
(222, 195, 178) with distance (fog), grey tile (232, 232, 232), and white (255, 255, 255)
for both the sky and everything beyond the 5 m far plane.

Per column the boundary is the row where the floor, connected to the bottom of the image,
meets a wall. It is sub-pixel: the first non-floor pixel may be an anti-aliased blend of floor
and wall, and its floor share is solved against the pixel above (wall) and below (floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WALL_RETURN, FAR, INVALID, TOP_RETURN = 0, 1, 2, 3  # per-column status
_CLOSE = np.ones((3, 1), np.uint8)  # close vertical gaps of up to 2 px (floor texture seams)


_BLUE = np.array([218.0, 181.0, 157.0])
_GRAY = np.array([232.0, 232.0, 232.0])
_DIR = _GRAY - _BLUE
_DIR2 = float(_DIR @ _DIR)
PALETTE_TOL = 5.0  # strict: max colour distance (any channel) from the blue-grey line
LOOSE_TOL = (
    30.0  # loose: tolerates JPEG ringing next to dark walls (10 to 25 levels for 4 to 5 rows)
)


def floor_palette(img: np.ndarray, tol: float = PALETTE_TOL) -> np.ndarray:
    """Pixels whose colour lies on the line from the blue tile to the grey tile.

    Stage 2 found that this one line covers everything the floor looks like: the two tiles
    themselves, the distance fog that drifts the blue tile towards grey, and the 1 to 2 px
    anti-aliased blends where the two tiles meet. White (sky, far plane) lies beyond the grey
    end of the line and is excluded, as are blends of floor with a wall."""
    p = img.astype(np.float32) - _BLUE
    t = (p @ _DIR) / _DIR2
    resid = np.abs(p - t[..., None] * _DIR).max(-1)
    return (t >= -0.1) & (t <= 1.08) & (resid <= tol)


def sky_mask(img: np.ndarray) -> np.ndarray:
    return (img == 255).all(-1)


def connected_floor(palette: np.ndarray) -> np.ndarray:
    """Floor = palette pixels connected to the bottom row (border-connectivity prior), after
    closing 1 to 2 px vertical gaps where one floor texture meets the other. Floor-coloured
    wall texture higher up in the image is therefore a wall."""
    closed = cv2.morphologyEx(palette.astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE)
    n, lab = cv2.connectedComponents(closed, connectivity=8)
    bottom = np.unique(lab[-1][closed[-1] > 0])
    return np.isin(lab, bottom[bottom > 0]) & (closed > 0)


@dataclass
class Segmentation:
    floor: np.ndarray  # (H, W) bool
    sky: np.ndarray  # (H, W) bool
    wall: np.ndarray  # (H, W) bool: everything else
    rows: np.ndarray  # (W,) sub-pixel boundary row (continuous coordinates), NaN unless WALL_RETURN
    status: np.ndarray  # (W,) WALL_RETURN, FAR (floor reaches the far plane) or INVALID
    top_rows: np.ndarray | None = None  # (W,) sub-pixel wall-top row, NaN where not visible


def segment(img: np.ndarray) -> Segmentation:
    strict = floor_palette(img)
    loose = floor_palette(img, LOOSE_TOL)
    sky = sky_mask(img)
    rows, status, b = boundary(img, strict, loose, sky)
    floor = connected_floor(loose)
    # where a boundary was found, the floor is exactly what lies below it
    valid = status != INVALID
    below = np.arange(img.shape[0])[:, None] >= np.where(valid, b, img.shape[0])[None, :]
    floor = np.where(valid[None, :], below, floor)
    wall = ~floor & ~sky
    return Segmentation(floor, sky, wall, rows, status, top_edge(img))


def step_boundary(strict: np.ndarray, loose: np.ndarray) -> np.ndarray:
    """Per column, the first floor row ``b`` of the best step fit: the ``b`` minimising
    strictly floor-coloured pixels above it plus not-even-loosely floor-coloured pixels below
    it. Isolated noise pixels on either side cost one each and do not move it. Searched over
    ``b`` in ``[2, H - 1]``."""
    h, w = strict.shape
    above = np.vstack([np.zeros((1, w), np.int32), np.cumsum(strict.astype(np.int32), 0)])
    nonfloor = 1 - loose.astype(np.int32)
    below = np.vstack([np.cumsum(nonfloor[::-1], 0)[::-1], np.zeros((1, w), np.int32)])
    cost = above + below  # index b = 0..h
    cost[:2] = 10**6
    cost[h:] = 10**6
    return cost.argmin(0)


def boundary(
    img: np.ndarray, strict: np.ndarray, loose: np.ndarray, sky: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel floor-wall boundary row, status and integer step row per column.

    A column is a **wall return** only if the floor reaches the bottom of the image, the pixel
    just above the floor is not sky, and the pixel two rows above that is not white (an MSAA blend
    sits between the far-plane edge and the white background). If it is white the floor
    reaches the far plane: status FAR, no wall (never a wall at 5 m). If the bottom of the
    column is not floor (a wall closer than about 0.16 m) the column is INVALID.

    Sub-pixel: the first non-floor pixel ``j`` may be a blend of floor and wall. Its floor
    share is solved against a clean floor pixel (the first strictly floor-coloured one below,
    since the rows next to the edge carry JPEG ringing) and the wall pixel above.
    """
    h, w = strict.shape
    rows = np.full(w, np.nan)
    status = np.full(w, INVALID, np.int8)
    b = step_boundary(strict, loose)
    u = np.arange(w)
    ok = loose[h - 2 : h].sum(0) >= 1  # the bottom of the column is floor
    ok &= b < h - 1
    j = np.clip(b - 1, 2, h - 1)  # first row above the floor
    far = ok & (sky[j, u] | sky[j - 1, u] | sky[np.clip(j - 2, 0, h - 1), u])
    wall_ok = ok & ~far
    status[far] = FAR
    idx = np.where(wall_ok)[0]
    if len(idx):
        jw = j[idx]
        # a clean floor colour (the first strictly floor-coloured pixel below the step, within
        # 8 rows: the rows next to the edge carry JPEG ringing) and the wall colour above it
        k = np.arange(0, 9)[:, None]
        rr = np.minimum(jw[None, :] + 1 + k, h - 1)
        isf = strict[rr, idx[None, :]]
        first = np.where(isf.any(0), isf.argmax(0), 0)
        floor_c = img[np.minimum(jw + 1 + first, h - 1), idx].astype(float)
        wall_c = img[jw - 1, idx].astype(float)
        d = floor_c - wall_c
        den = (d * d).sum(1)

        def floor_share(rows_):
            pix = img[rows_, idx].astype(float)
            return np.where(
                den > 100.0, np.clip(((pix - wall_c) * d).sum(1) / np.maximum(den, 1), 0, 1), 0.0
            )

        # the edge lies in pixel j (the first non-floor row) or in pixel b = j + 1 (the first
        # floor row, which the loose palette accepts even when it is a blend): the boundary is
        # the wall coverage of the two pixels added to the top of j
        s_j = floor_share(jw)
        s_b = floor_share(np.minimum(jw + 1, h - 1))
        rows[idx] = jw + (1.0 - s_j) + (1.0 - s_b)
        status[idx] = WALL_RETURN
    return rows, status, b


SKY_LOOSE = 232  # all channels at least this: sky, with JPEG ringing next to dark walls


def top_edge(img: np.ndarray) -> np.ndarray:
    """Sub-pixel row of the wall top (wall against sky) per column, NaN if not visible.

    The wall top is the first run of non-sky pixels from the top of the image: a pixel is wall
    if it is not sky-white and the next two rows are not either (isolated specks are ignored).
    The first such pixel may be a blend of sky and wall; its wall share is solved against the
    white sky and the wall pixel two rows below. A column whose top is in the first row has no
    sky above its wall (the wall top is above the image) and gets NaN.
    """
    h, w = img.shape[:2]
    sky = (img >= SKY_LOOSE).all(-1)
    solid = ~sky & ~np.roll(sky, -1, 0) & ~np.roll(sky, -2, 0)
    solid[-2:] = False
    has = solid.any(0)
    j = solid.argmax(0)
    out = np.full(w, np.nan)
    u = np.arange(w)
    ok = has & (j >= 1)
    idx = np.where(ok)[0]
    if len(idx):
        jj = j[idx]
        wall_c = img[np.minimum(jj + 2, h - 1), idx].astype(float)
        d = wall_c - 255.0
        den = (d * d).sum(1)
        pix = img[jj, idx].astype(float) - 255.0
        share = np.where(den > 400.0, np.clip((pix * d).sum(1) / np.maximum(den, 1), 0, 1), 1.0)
        out[idx] = jj + 1 - share
    del u
    return out
