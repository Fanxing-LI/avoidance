# python
import os
import cv2
import numpy as np

def build_occupancy(h, w, square, circle):
    occ = np.zeros((h, w), dtype=np.uint8)
    sx, sy, hs = square
    x1, y1 = int(sx - hs), int(sy - hs)
    x2, y2 = int(sx + hs), int(sy + hs)
    cv2.rectangle(occ, (x1, y1), (x2, y2), color=1, thickness=-1)
    cx, cy, r = circle
    cv2.circle(occ, (int(cx), int(cy)), int(r), color=1, thickness=-1)
    return occ

def compute_esdf(occ, resolution=0.01):
    occ_u8 = (occ.astype(np.uint8) * 255)
    free_u8 = ((1 - occ).astype(np.uint8) * 255)
    dist_out_px = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)
    dist_in_px = cv2.distanceTransform(occ_u8, cv2.DIST_L2, 3)
    esdf_m = (dist_out_px - dist_in_px) * float(resolution)
    return esdf_m

def colorize_esdf(esdf_m, occ=None, max_abs_m=None, cmap=cv2.COLORMAP_TURBO, use_log=False):
    d_pos = np.maximum(esdf_m.astype(np.float32), 0.0)
    eps = 1e-6
    if use_log:
        if np.any(d_pos > 0):
            s = float(np.percentile(d_pos[d_pos > 0], 95))
            s = max(s, eps)
        else:
            s = 1.0
        norm01 = 1.0 - np.exp(-d_pos / s)
    else:
        denom = (max_abs_m if (max_abs_m is not None and max_abs_m > 0) else float(d_pos.max()))
        if denom < eps:
            denom = 1.0
        norm01 = np.clip(d_pos / denom, 0.0, 1.0)
    u8 = (norm01 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cmap)
    if occ is not None:
        mask_in = esdf_m < 0
        colored[mask_in] = (255, 255, 255)
    return colored

def overlay_obstacles(color_bgr, occ, color=(255, 255, 255)):
    edges = cv2.Canny((occ * 255).astype(np.uint8), 50, 150)
    color_bgr[edges > 0] = color
    return color_bgr

def draw_gradient_arrows(base_img_bgr, esdf_m, occ, step=16, arrow_len_px=12, thickness=1, color=(0, 0, 0)):
    """
    Draw normalized ESDF gradient arrows.
    - step: sampling stride in pixels.
    - arrow_len_px: arrow length in pixels.
    """
    H, W = esdf_m.shape
    dy, dx = np.gradient(esdf_m.astype(np.float32))  # returns (dy, dx)
    mag = np.sqrt(dx * dx + dy * dy) + 1e-8
    nx, ny = dx / mag, dy / mag

    img = base_img_bgr.copy()
    for y in range(0, H, step):
        for x in range(0, W, step):
            if occ[y, x] == 1:
                continue
            if mag[y, x] < 1e-3:
                continue
            x2 = int(x + nx[y, x] * arrow_len_px)
            y2 = int(y + ny[y, x] * arrow_len_px)
            cv2.arrowedLine(img, (x, y), (x2, y2), color, thickness, tipLength=0.35)
    return img

def main():
    H, W = 712, 1024
    resolution = 0.01
    square = (160, 230, 60)
    circle = (720, 470, 80)

    occ = build_occupancy(H, W, square, circle)
    esdf_m = compute_esdf(occ, resolution=resolution)

    color = colorize_esdf(esdf_m, occ=occ, max_abs_m=None, cmap=cv2.COLORMAP_PARULA, use_log=False)
    color = overlay_obstacles(color, occ, color=(255, 255, 255))

    # Draw gradient directions (black arrows)
    grad_img = draw_gradient_arrows(color, esdf_m, occ, step=16, arrow_len_px=12, thickness=1, color=(0, 0, 0))

    out_dir = os.path.dirname(__file__)
    cv2.imwrite(os.path.join(out_dir, "esdf_color.png"), color)
    cv2.imwrite(os.path.join(out_dir, "esdf_grad.png"), grad_img)
    cv2.imwrite(os.path.join(out_dir, "esdf_float.exr"), esdf_m.astype(np.float32))

if __name__ == "__main__":
    main()