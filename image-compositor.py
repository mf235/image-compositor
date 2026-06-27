# -*- coding: utf-8 -*-
"""
image-compositor.py

元画像に複数素材を配置し、輪郭なじませ・透明度・回転・拡大縮小・色合わせ・影で
合成の手間を減らすためのGUIツール。

必要ライブラリ:
    pip install PyQt5 opencv-python numpy
"""

import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QByteArray, QMimeData, QPoint, QPointF, QRect, QRectF, QSize, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QDrag, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "画像合成ツール"
SETTINGS_FILE = "image-compositor-settings.json"
PARTS_DIR = "_parts"
DEFAULT_PARTS_FOLDER = "default"
PART_MIME = "application/x-image-compositor-part-id"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WARP_GRID_ROWS = 4
WARP_GRID_COLS = 4


# -----------------------------------------------------------------------------
# 日本語パス対応 I/O
# -----------------------------------------------------------------------------

def script_dir() -> Path:
    return Path(__file__).resolve().parent


def imread_japanese(filename: str):
    try:
        data = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as exc:
        print(f"Read Error: {exc}")
        return None


def imwrite_japanese(filename: str, img) -> bool:
    try:
        ext = os.path.splitext(filename)[1]
        ok, encoded = cv2.imencode(ext, img)
        if ok:
            with open(filename, "w+b") as f:
                encoded.tofile(f)
            return True
        return False
    except Exception as exc:
        print(f"Write Error: {exc}")
        return False


def ensure_bgra(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 4:
        return img.copy()
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img.copy()


def bgra_to_qimage(img):
    img = ensure_bgra(img)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    h, w, ch = rgba.shape
    qimg = QImage(rgba.data, w, h, ch * w, QImage.Format_RGBA8888)
    return qimg.copy()


def image_files_from_urls(mime_data):
    if not mime_data.hasUrls():
        return []
    paths = []
    for url in mime_data.urls():
        path = url.toLocalFile()
        if path and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def image_file_from_urls(mime_data):
    paths = image_files_from_urls(mime_data)
    return paths[0] if paths else None


# -----------------------------------------------------------------------------
# 画像処理
# -----------------------------------------------------------------------------

def apply_feather_alpha(src_bgra, feather_px: int):
    """素材の輪郭から内側へ feather_px 分だけ、徐々に不透明になるように透過を加える。"""
    if feather_px <= 0:
        return src_bgra
    img = ensure_bgra(src_bgra)
    alpha = img[:, :, 3].astype(np.float32)
    mask = (alpha > 5).astype(np.uint8)
    if cv2.countNonZero(mask) == 0:
        return img

    # 画像の端まで不透明な素材でも必ず「外側 0」を持てるよう、
    # ゼロの枠を付けてから距離変換する。
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5).astype(np.float32)[1:-1, 1:-1]

    # 輪郭(外側に接する最初の不透明ピクセル)が最も薄く、
    # 内側へ行くほど 1.0 に近づく。
    if feather_px <= 1:
        factor = np.where(dist >= 2.0, 1.0, 0.0).astype(np.float32)
    else:
        factor = np.clip((dist - 1.0) / float(feather_px), 0.0, 1.0)
    img[:, :, 3] = np.clip(alpha * factor, 0, 255).astype(np.uint8)
    return img


def apply_opacity(src_bgra, opacity_percent: int):
    img = ensure_bgra(src_bgra)
    ratio = max(0.0, min(1.0, opacity_percent / 100.0))
    img[:, :, 3] = np.clip(img[:, :, 3].astype(np.float32) * ratio, 0, 255).astype(np.uint8)
    return img


def apply_flip(src_bgra, flip_h=False, flip_v=False):
    img = ensure_bgra(src_bgra)
    if flip_h and flip_v:
        return cv2.flip(img, -1)
    if flip_h:
        return cv2.flip(img, 1)
    if flip_v:
        return cv2.flip(img, 0)
    return img


def default_warp_points(rows=WARP_GRID_ROWS, cols=WARP_GRID_COLS):
    points = []
    for r in range(rows):
        y = r / float(rows - 1) if rows > 1 else 0.0
        for c in range(cols):
            x = c / float(cols - 1) if cols > 1 else 0.0
            points.append([float(x), float(y)])
    return points


def normalized_warp_points(points, rows=WARP_GRID_ROWS, cols=WARP_GRID_COLS):
    expected = rows * cols
    if not isinstance(points, list) or len(points) != expected:
        return default_warp_points(rows, cols)
    out = []
    try:
        for p in points:
            out.append([float(p[0]), float(p[1])])
    except Exception:
        return default_warp_points(rows, cols)
    return out


def get_item_warp_points(item):
    return normalized_warp_points(
        item.get("warp_points"),
        int(item.get("warp_grid_rows", WARP_GRID_ROWS)),
        int(item.get("warp_grid_cols", WARP_GRID_COLS)),
    )


def warp_geometry(width, height, points, rows=WARP_GRID_ROWS, cols=WARP_GRID_COLS):
    width = max(1, int(width))
    height = max(1, int(height))
    pts = normalized_warp_points(points, rows, cols)
    src_pts = []
    dst_pts = []
    for r in range(rows):
        sy = r * (height - 1) / float(rows - 1) if rows > 1 else 0.0
        for c in range(cols):
            sx = c * (width - 1) / float(cols - 1) if cols > 1 else 0.0
            idx = r * cols + c
            dx = pts[idx][0] * (width - 1)
            dy = pts[idx][1] * (height - 1)
            src_pts.append([sx, sy])
            dst_pts.append([dx, dy])

    dst_arr = np.array(dst_pts, dtype=np.float32)
    min_x = int(math.floor(float(np.min(dst_arr[:, 0]))))
    min_y = int(math.floor(float(np.min(dst_arr[:, 1]))))
    max_x = int(math.ceil(float(np.max(dst_arr[:, 0]))))
    max_y = int(math.ceil(float(np.max(dst_arr[:, 1]))))
    out_w = max(1, max_x - min_x + 1)
    out_h = max(1, max_y - min_y + 1)
    shift_x = -min_x
    shift_y = -min_y
    shifted_dst = [[x + shift_x, y + shift_y] for x, y in dst_pts]
    return {
        "src_pts": np.array(src_pts, dtype=np.float32),
        "dst_pts": np.array(dst_pts, dtype=np.float32),
        "shifted_dst_pts": np.array(shifted_dst, dtype=np.float32),
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "out_w": int(out_w),
        "out_h": int(out_h),
        "rows": int(rows),
        "cols": int(cols),
    }


def rotation_matrix_for_image(width, height, angle_degrees):
    angle = float(angle_degrees)
    width = max(1, int(width))
    height = max(1, int(height))
    if abs(angle) < 1e-4:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return matrix, width, height
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int((height * sin) + (width * cos)))
    new_h = max(1, int((height * cos) + (width * sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return matrix.astype(np.float32), new_w, new_h


def resize_image(src_bgra, scale_w: float, scale_h: float = None):
    img = ensure_bgra(src_bgra)
    scale_w = max(0.02, float(scale_w))
    scale_h = scale_w if scale_h is None else max(0.02, float(scale_h))
    if abs(scale_w - 1.0) <= 1e-4 and abs(scale_h - 1.0) <= 1e-4:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale_w)))
    nh = max(1, int(round(h * scale_h)))
    # 画質優先。縮小・拡大とも Bicubic に統一する。
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def rotate_image(src_bgra, angle_degrees: float):
    img = ensure_bgra(src_bgra)
    h, w = img.shape[:2]
    matrix, new_w, new_h = rotation_matrix_for_image(w, h, angle_degrees)
    if abs(float(angle_degrees)) < 1e-4:
        return img
    return cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _warp_triangle(src, dst, tri_src, tri_dst):
    tri_src = np.array(tri_src, dtype=np.float32)
    tri_dst = np.array(tri_dst, dtype=np.float32)
    r1 = cv2.boundingRect(tri_src)
    r2 = cv2.boundingRect(tri_dst)
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return

    src_crop = src[y1:y1 + h1, x1:x1 + w1]
    if src_crop.size == 0:
        return
    tri_src_rect = tri_src - np.array([x1, y1], dtype=np.float32)
    tri_dst_rect = tri_dst - np.array([x2, y2], dtype=np.float32)
    matrix = cv2.getAffineTransform(tri_src_rect, tri_dst_rect)
    warped = cv2.warpAffine(
        src_crop,
        matrix,
        (w2, h2),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(tri_dst_rect)), 255, lineType=cv2.LINE_AA)
    warped = ensure_bgra(warped)
    warped[:, :, 3] = np.clip(warped[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
    alpha_composite_bgra(dst, warped, x2, y2)


def apply_mesh_warp(src_bgra, points, rows=WARP_GRID_ROWS, cols=WARP_GRID_COLS):
    img = ensure_bgra(src_bgra)
    h, w = img.shape[:2]
    pts = normalized_warp_points(points, rows, cols)
    if pts == default_warp_points(rows, cols):
        return img
    geom = warp_geometry(w, h, pts, rows, cols)
    out = np.zeros((geom["out_h"], geom["out_w"], 4), dtype=np.uint8)
    src_pts = geom["src_pts"]
    dst_pts = geom["shifted_dst_pts"]

    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i10 = r * cols + c + 1
            i01 = (r + 1) * cols + c
            i11 = (r + 1) * cols + c + 1
            _warp_triangle(img, out, [src_pts[i00], src_pts[i10], src_pts[i11]], [dst_pts[i00], dst_pts[i10], dst_pts[i11]])
            _warp_triangle(img, out, [src_pts[i00], src_pts[i11], src_pts[i01]], [dst_pts[i00], dst_pts[i11], dst_pts[i01]])
    return out


def item_scale_w(item):
    return max(0.02, float(item.get("scale_w", item.get("scale", 1.0))))


def item_scale_h(item):
    if bool(item.get("scale_lock", True)):
        return item_scale_w(item)
    return max(0.02, float(item.get("scale_h", item.get("scale", 1.0))))


def resize_and_rotate(src_bgra, scale_w: float, angle_degrees: float, scale_h: float = None):
    img = resize_image(src_bgra, scale_w, scale_h)
    return rotate_image(img, angle_degrees)


def alpha_composite_bgra(dst_bgra, src_bgra, x: int, y: int):
    """dst_bgra上の(x, y)にsrc_bgraをアルファ合成。x,yは左上。"""
    dst = dst_bgra
    src = ensure_bgra(src_bgra)
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(dw, x + sw)
    y2 = min(dh, y + sh)
    if x1 >= x2 or y1 >= y2:
        return dst

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    src_roi = src[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
    dst_roi = dst[y1:y2, x1:x2].astype(np.float32) / 255.0

    src_a = src_roi[:, :, 3:4]
    dst_a = dst_roi[:, :, 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = np.zeros_like(dst_roi[:, :, :3])
    denom = np.maximum(out_a, 1e-6)
    out_rgb = (src_roi[:, :, :3] * src_a + dst_roi[:, :, :3] * dst_a * (1.0 - src_a)) / denom

    out = np.dstack([out_rgb, out_a])
    dst[y1:y2, x1:x2] = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return dst


def make_shadow_image(transformed_part, opacity_percent: int, blur_px: int):
    part = ensure_bgra(transformed_part)
    alpha = part[:, :, 3]
    if cv2.countNonZero((alpha > 0).astype(np.uint8)) == 0:
        return None
    blur_px = max(0, int(blur_px))
    if blur_px > 0:
        k = blur_px * 2 + 1
        shadow_alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    else:
        shadow_alpha = alpha.copy()
    ratio = max(0.0, min(1.0, opacity_percent / 100.0))
    shadow_alpha = np.clip(shadow_alpha.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
    shadow = np.zeros_like(part)
    shadow[:, :, 3] = shadow_alpha
    return shadow


# -----------------------------------------------------------------------------
# 色合わせ: color-matcher.py の4モード + 追加1モード
# -----------------------------------------------------------------------------

def color_match_bgra(source_bgra, target_bgra, algo_idx: int, strength_percent: int):
    source = ensure_bgra(source_bgra)
    target = ensure_bgra(target_bgra)
    if source is None or target is None or source.size == 0 or target.size == 0:
        return source

    strength = max(0.0, min(1.0, strength_percent / 100.0))
    if strength <= 0:
        return source

    src_alpha = source[:, :, 3]
    tgt_alpha = target[:, :, 3]
    src_mask = (src_alpha >= 10).astype("uint8")
    tgt_mask = (tgt_alpha >= 10).astype("uint8")
    if cv2.countNonZero(src_mask) < 10 or cv2.countNonZero(tgt_mask) < 10:
        return source

    src_rgb = source[:, :, :3]
    tgt_rgb = target[:, :, :3]
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(tgt_rgb, cv2.COLOR_BGR2LAB).astype("float32")

    try:
        def safe_mask(mask):
            """有効ピクセルが少なすぎる場合は全体マスクへ戻す。"""
            if cv2.countNonZero(mask) < 10:
                return np.ones(mask.shape, dtype="uint8")
            return mask

        def mean_std_channel(ch, mask):
            """OpenCVのmeanStdDevを使った通常統計。"""
            mask = safe_mask(mask)
            mean, std = cv2.meanStdDev(ch, mask=mask)
            return float(mean[0][0]), float(std[0][0])

        def trimmed_mean_std_channel(ch, mask, trim=5):
            """上下の外れ値を除外した平均・標準偏差。"""
            mask = safe_mask(mask)
            pixels = ch[mask > 0].astype("float32")
            if pixels.size < 10:
                pixels = ch.reshape(-1).astype("float32")
            if pixels.size == 0:
                return 0.0, 1.0

            low, high = np.percentile(pixels, [trim, 100 - trim])
            trimmed = pixels[(pixels >= low) & (pixels <= high)]
            if trimmed.size < 10:
                trimmed = pixels

            return float(np.mean(trimmed)), float(np.std(trimmed))

        def transfer_channel(src_ch, src_mean, src_std, tgt_mean, tgt_std):
            return (src_ch - src_mean) * (tgt_std / (src_std + 1e-5)) + tgt_mean

        def transfer_lab_by_stats(s_lab, t_lab, s_mask, t_mask, stats_func=mean_std_channel, keep_l=False):
            """LABの平均・標準偏差を合わせる共通処理。keep_l=Trueなら明度は元画像を維持する。"""
            s_l_mean, s_l_std = stats_func(s_lab[:, :, 0], s_mask)
            s_a_mean, s_a_std = stats_func(s_lab[:, :, 1], s_mask)
            s_b_mean, s_b_std = stats_func(s_lab[:, :, 2], s_mask)

            t_l_mean, t_l_std = stats_func(t_lab[:, :, 0], t_mask)
            t_a_mean, t_a_std = stats_func(t_lab[:, :, 1], t_mask)
            t_b_mean, t_b_std = stats_func(t_lab[:, :, 2], t_mask)

            if keep_l:
                out_l = s_lab[:, :, 0].copy()
            else:
                out_l = transfer_channel(s_lab[:, :, 0], s_l_mean, s_l_std, t_l_mean, t_l_std)
            out_a = transfer_channel(s_lab[:, :, 1], s_a_mean, s_a_std, t_a_mean, t_a_std)
            out_b = transfer_channel(s_lab[:, :, 2], s_b_mean, s_b_std, t_b_mean, t_b_std)
            return out_l, out_a, out_b

        if algo_idx in [0, 1, 2, 4, 6]:
            if algo_idx == 1:
                # 白黒除外: 明るすぎる/暗すぎる画素を統計から外す。
                lum_min, lum_max = 20, 235
                src_lum_mask = ((src_lab[:, :, 0] >= lum_min) & (src_lab[:, :, 0] <= lum_max)).astype("uint8")
                tgt_lum_mask = ((tgt_lab[:, :, 0] >= lum_min) & (tgt_lab[:, :, 0] <= lum_max)).astype("uint8")

                s_mask = cv2.bitwise_and(src_mask, src_lum_mask)
                t_mask = cv2.bitwise_and(tgt_mask, tgt_lum_mask)

                if cv2.countNonZero(s_mask) < 10:
                    s_mask = src_mask
                if cv2.countNonZero(t_mask) < 10:
                    t_mask = tgt_mask
            else:
                s_mask = src_mask
                t_mask = tgt_mask

            if algo_idx == 2:
                def get_kmeans_stats(lab_img, mask, k=5):
                    pixels = lab_img[mask > 0]
                    if len(pixels) < k:
                        return np.mean(lab_img, axis=(0, 1)), np.std(lab_img, axis=(0, 1))

                    np.random.seed(42)
                    if len(pixels) > 10000:
                        indices = np.random.choice(len(pixels), 10000, replace=False)
                        pixels = pixels[indices]

                    pixels = np.float32(pixels)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    return np.mean(centers, axis=0), np.std(centers, axis=0)

                src_mean, src_std = get_kmeans_stats(src_lab, s_mask)
                tgt_mean, tgt_std = get_kmeans_stats(tgt_lab, t_mask)

                l = transfer_channel(src_lab[:, :, 0], src_mean[0], src_std[0], tgt_mean[0], tgt_std[0])
                a = transfer_channel(src_lab[:, :, 1], src_mean[1], src_std[1], tgt_mean[1], tgt_std[1])
                b = transfer_channel(src_lab[:, :, 2], src_mean[2], src_std[2], tgt_mean[2], tgt_std[2])

            elif algo_idx == 4:
                # 明度保持: Lは元画像のまま、a/bだけ参考画像へ寄せる。
                l, a, b = transfer_lab_by_stats(src_lab, tgt_lab, s_mask, t_mask, keep_l=True)

            elif algo_idx == 6:
                # 安定補正: 上下5%の外れ値を除外して統計を取る。
                l, a, b = transfer_lab_by_stats(
                    src_lab, tgt_lab, s_mask, t_mask,
                    stats_func=lambda ch, mask: trimmed_mean_std_channel(ch, mask, trim=5),
                    keep_l=False
                )

            else:
                l, a, b = transfer_lab_by_stats(src_lab, tgt_lab, s_mask, t_mask, keep_l=False)

        elif algo_idx == 3:
            def match_hist(src_ch, tgt_ch, s_mask, t_mask):
                s_mask = safe_mask(s_mask)
                t_mask = safe_mask(t_mask)
                src_hist, _ = np.histogram(src_ch[s_mask > 0], bins=256, range=[0, 256])
                tgt_hist, _ = np.histogram(tgt_ch[t_mask > 0], bins=256, range=[0, 256])

                if src_hist.sum() == 0 or tgt_hist.sum() == 0:
                    return src_ch

                src_cdf = src_hist.cumsum() / src_hist.sum()
                tgt_cdf = tgt_hist.cumsum() / tgt_hist.sum()

                lut = np.zeros(256, dtype="uint8")
                for i in range(256):
                    idx = np.abs(tgt_cdf - src_cdf[i]).argmin()
                    lut[i] = idx

                src_ch_uint8 = np.clip(src_ch, 0, 255).astype("uint8")
                res = cv2.LUT(src_ch_uint8, lut)
                return res.astype("float32")

            l = match_hist(src_lab[:, :, 0], tgt_lab[:, :, 0], src_mask, tgt_mask)
            a = match_hist(src_lab[:, :, 1], tgt_lab[:, :, 1], src_mask, tgt_mask)
            b = match_hist(src_lab[:, :, 2], tgt_lab[:, :, 2], src_mask, tgt_mask)

        elif algo_idx == 5:
            # 明暗別: 暗部/中間/明部で別々に統計を取り、元画像Lに応じてなめらかに合成する。
            ranges = [
                (0, 95, 42.0),
                (80, 185, 128.0),
                (160, 255, 213.0),
            ]
            weight_width = 95.0
            weight_sum = np.zeros(src_lab.shape[:2], dtype="float32") + 1e-5
            l = np.zeros(src_lab.shape[:2], dtype="float32")
            a = np.zeros(src_lab.shape[:2], dtype="float32")
            b = np.zeros(src_lab.shape[:2], dtype="float32")

            for low, high, center in ranges:
                src_l_range = ((src_lab[:, :, 0] >= low) & (src_lab[:, :, 0] <= high)).astype("uint8")
                tgt_l_range = ((tgt_lab[:, :, 0] >= low) & (tgt_lab[:, :, 0] <= high)).astype("uint8")
                s_mask = cv2.bitwise_and(src_mask, src_l_range)
                t_mask = cv2.bitwise_and(tgt_mask, tgt_l_range)

                if cv2.countNonZero(s_mask) < 10:
                    s_mask = src_mask
                if cv2.countNonZero(t_mask) < 10:
                    t_mask = tgt_mask

                part_l, part_a, part_b = transfer_lab_by_stats(src_lab, tgt_lab, s_mask, t_mask, keep_l=False)
                weight = np.maximum(0.0, 1.0 - np.abs(src_lab[:, :, 0] - center) / weight_width).astype("float32")

                l += part_l * weight
                a += part_a * weight
                b += part_b * weight
                weight_sum += weight

            l /= weight_sum
            a /= weight_sum
            b /= weight_sum

        else:
            return source

        l = np.ascontiguousarray(np.clip(l, 0, 255).astype("float32"))
        a = np.ascontiguousarray(np.clip(a, 0, 255).astype("float32"))
        b = np.ascontiguousarray(np.clip(b, 0, 255).astype("float32"))
        transfer_lab = cv2.merge([l, a, b]).astype("uint8")
        transfer_rgb = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)

        transfer_bgra = cv2.cvtColor(transfer_rgb, cv2.COLOR_BGR2BGRA)
        transfer_bgra[:, :, 3] = src_alpha
        if strength < 1.0:
            out = cv2.addWeighted(transfer_bgra, strength, source, 1.0 - strength, 0)
            out[:, :, 3] = src_alpha
            return out
        return transfer_bgra
    except Exception as exc:
        print(f"Color match error: {exc}")
        return source


# -----------------------------------------------------------------------------
# UI widgets
# -----------------------------------------------------------------------------


class PartsFolderComboBox(QComboBox):
    """開く直前に _parts 配下のフォルダを拾い直すコンボボックス。"""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def showPopup(self):
        self.main_window.refresh_part_folders(keep_selection=True)
        super().showPopup()

class PartsListWidget(QListWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setDropIndicatorShown(True)
        self.setViewMode(QListWidget.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setIconSize(QSize(80, 80))
        self.setGridSize(QSize(92, 92))
        self.setSpacing(8)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setMinimumWidth(320)
        self.setStyleSheet("QListWidget { padding: 8px; } QListWidget::item { margin: 4px; }")

    def mimeData(self, items):
        mime = QMimeData()
        if items:
            part_id = items[0].data(Qt.UserRole)
            if part_id:
                mime.setData(PART_MIME, QByteArray(str(part_id).encode("utf-8")))
        return mime

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        part_id = item.data(Qt.UserRole)
        if not part_id:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(PART_MIME, QByteArray(str(part_id).encode("utf-8")))
        drag.setMimeData(mime)
        icon = item.icon()
        if not icon.isNull():
            drag.setPixmap(icon.pixmap(80, 80))
            drag.setHotSpot(QPoint(40, 40))
        drag.exec_(Qt.CopyAction)

    def dragEnterEvent(self, event):
        if self._has_image_urls(event.mimeData()) or event.mimeData().hasFormat(PART_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._has_image_urls(event.mimeData()) or event.mimeData().hasFormat(PART_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._image_paths_from_mime(event.mimeData())
        if paths:
            self.main_window.add_part_files(paths)
            event.acceptProposedAction()
            return
        if event.mimeData().hasFormat(PART_MIME):
            # リスト内の並べ替えはv1ではしない。素材配置用ドラッグだけ受け持つ。
            event.acceptProposedAction()
            return
        event.ignore()

    def _has_image_urls(self, mime):
        return bool(self._image_paths_from_mime(mime))

    def _image_paths_from_mime(self, mime):
        if not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
        return paths


class LoupeView(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dragging_loupe = False
        self.drag_start_widget_pos = QPoint()
        self.drag_start_image_pos = None
        self.setMinimumSize(60, 60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #151515;")

    def sizeHint(self):
        return QSize(220, 160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        painter.setPen(QPen(QColor(95, 95, 95), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        img = self.main_window.current_loupe_source_image()
        pos = self.main_window.loupe_image_pos
        if img is None or pos is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "右クリック位置を\n拡大表示")
            return

        h, w = img.shape[:2]
        zoom = max(1.0, self.main_window.loupe_slider.value() / 100.0)
        side = max(8, min(w, h, int(round(min(self.width(), self.height()) / zoom))))
        cx = int(round(pos.x()))
        cy = int(round(pos.y()))
        x1 = max(0, min(w - side, cx - side // 2))
        y1 = max(0, min(h - side, cy - side // 2))
        crop = img[y1:y1 + side, x1:x1 + side]
        if crop.size == 0:
            return

        crop_qimg = bgra_to_qimage(crop)
        pix = QPixmap.fromImage(crop_qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        painter.drawPixmap(x, y, pix)

    def wheelEvent(self, event):
        self.main_window.adjust_loupe_zoom_by_wheel(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        if self.main_window.base_image is None:
            return
        if self.main_window.loupe_image_pos is None:
            h, w = self.main_window.base_image.shape[:2]
            self.main_window.loupe_image_pos = QPointF(w / 2.0, h / 2.0)
        self.dragging_loupe = True
        self.drag_start_widget_pos = event.pos()
        self.drag_start_image_pos = QPointF(self.main_window.loupe_image_pos)
        self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging_loupe or self.drag_start_image_pos is None:
            super().mouseMoveEvent(event)
            return
        if self.main_window.base_image is None:
            return
        h, w = self.main_window.base_image.shape[:2]
        zoom = max(1.0, self.main_window.loupe_slider.value() / 100.0)
        dx = (event.pos().x() - self.drag_start_widget_pos.x()) / zoom
        dy = (event.pos().y() - self.drag_start_widget_pos.y()) / zoom
        # 画像ビューの手のひら移動に合わせる。ドラッグした方向へ表示が動くよう、参照位置は逆へずらす。
        nx = max(0.0, min(float(w - 1), self.drag_start_image_pos.x() - dx))
        ny = max(0.0, min(float(h - 1), self.drag_start_image_pos.y() - dy))
        self.main_window.update_loupe_position(QPointF(nx, ny))

    def mouseReleaseEvent(self, event):
        if self.dragging_loupe:
            self.dragging_loupe = False
            self.drag_start_image_pos = None
            self.unsetCursor()
            self.update()
            return
        super().mouseReleaseEvent(event)


class LoupeToolWindow(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window, Qt.Tool | Qt.WindowTitleHint)
        self.main_window = main_window
        self.setWindowTitle("ルーペ")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setMinimumSize(120, 110)
        self.resize(280, 280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.main_window.loupe_view = LoupeView(main_window)
        layout.addWidget(self.main_window.loupe_view, 1)

        control = QHBoxLayout()
        self.main_window.loupe_label = QLabel("300%")
        self.main_window.loupe_slider = QSlider(Qt.Horizontal)
        self.main_window.loupe_slider.setRange(100, 800)
        self.main_window.loupe_slider.setSingleStep(25)
        self.main_window.loupe_slider.setPageStep(50)
        self.main_window.loupe_slider.setValue(300)
        self.main_window.loupe_slider.valueChanged.connect(self.main_window.on_loupe_zoom_changed)
        control.addWidget(QLabel("拡大率"))
        control.addWidget(self.main_window.loupe_slider, 1)
        control.addWidget(self.main_window.loupe_label)
        layout.addLayout(control)

    def wheelEvent(self, event):
        self.main_window.adjust_loupe_zoom_by_wheel(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        if hasattr(self.main_window, "sync_view_buttons"):
            save = not getattr(self.main_window, "loading_ui", False) and not getattr(self.main_window, "_closing_app", False)
            self.main_window.sync_view_buttons(save=save)

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self.main_window, "sync_view_buttons"):
            save = not getattr(self.main_window, "loading_ui", False) and not getattr(self.main_window, "_closing_app", False)
            self.main_window.sync_view_buttons(save=save)

    def closeEvent(self, event):
        # ×で閉じた時はルーペを非表示にして、表示状態を保存する。
        event.accept()
        self.hide()
        if hasattr(self.main_window, "sync_view_buttons"):
            save = not getattr(self.main_window, "loading_ui", False) and not getattr(self.main_window, "_closing_app", False)
            self.main_window.sync_view_buttons(save=save)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self.main_window, "save_settings") and not getattr(self.main_window, "loading_ui", False):
            self.main_window.save_settings()

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self.main_window, "save_settings") and not getattr(self.main_window, "loading_ui", False):
            self.main_window.save_settings()


class PlacedListWidget(QListWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def row_at_pos_loose(self, pos):
        index = self.indexAt(pos)
        if index.isValid():
            return index.row()
        for row in range(self.count()):
            rect = self.visualItemRect(self.item(row))
            rect.setLeft(0)
            rect.setRight(self.viewport().width())
            if rect.contains(pos):
                return row
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            row = self.row_at_pos_loose(event.pos())
            if row >= 0:
                self.setCurrentRow(row)
                self.main_window.select_item(row)
            else:
                self.clearSelection()
                self.main_window.select_item(None)
        super().mousePressEvent(event)


class CanvasWidget(QWidget):
    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_rect = QRect()
        self.dragging_item = False
        self.drag_offset = QPointF(0, 0)
        self.panning_view = False
        self.pan_last_pos = None
        self.zoom_scale = 1.0
        self.view_center = None
        self._last_drag_ui_update = 0.0
        self._bg_cache_image_id = None
        self._bg_cache_size = QSize()
        self._bg_cache_pixmap = None
        self._bg_cache_rect = QRect()
        self.setStyleSheet("background-color: #202020;")
        self.setFocusPolicy(Qt.StrongFocus)

    def current_canvas_image(self):
        if self.dragging_item and self.main_window.drag_preview_image is not None:
            return self.main_window.drag_preview_image
        return self.main_window.preview_image

    def image_shape(self):
        img = self.current_canvas_image()
        if img is None:
            img = self.main_window.base_image
        if img is None:
            return None
        return img.shape[:2]

    def ensure_view_center(self):
        shape = self.image_shape()
        if shape is None:
            self.view_center = None
            return
        h, w = shape
        if self.view_center is None:
            self.view_center = QPointF(w / 2.0, h / 2.0)
        self.clamp_view_center()

    def get_fit_scale(self):
        shape = self.image_shape()
        if shape is None:
            return 1.0
        h, w = shape
        area_w = max(1, self.width() - 8)
        area_h = max(1, self.height() - 8)
        return max(self.MIN_ZOOM, min(area_w / max(1, w), area_h / max(1, h)))

    def get_current_scale(self):
        return max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.get_fit_scale() * self.zoom_scale))

    def clamp_view_center(self):
        shape = self.image_shape()
        if shape is None or self.view_center is None:
            return
        h, w = shape
        scale = self.get_current_scale()
        view_w = self.width() / max(scale, 1e-6)
        view_h = self.height() / max(scale, 1e-6)

        if view_w >= w:
            cx = w / 2.0
        else:
            half = view_w / 2.0
            cx = min(max(self.view_center.x(), half), w - half)

        if view_h >= h:
            cy = h / 2.0
        else:
            half = view_h / 2.0
            cy = min(max(self.view_center.y(), half), h - half)

        self.view_center = QPointF(cx, cy)

    def reset_view(self):
        self.zoom_scale = 1.0
        self.view_center = None
        self.ensure_view_center()
        self.invalidate_background_cache()
        self.update()

    def set_zoom(self, new_zoom_scale, anchor_pos=None):
        shape = self.image_shape()
        if shape is None:
            return
        self.ensure_view_center()
        if self.view_center is None:
            return
        old_scale = self.get_current_scale()
        if anchor_pos is not None:
            anchor_img = QPointF(
                self.view_center.x() + (anchor_pos.x() - self.width() / 2.0) / max(old_scale, 1e-6),
                self.view_center.y() + (anchor_pos.y() - self.height() / 2.0) / max(old_scale, 1e-6),
            )
        else:
            anchor_img = None

        self.zoom_scale = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(new_zoom_scale)))
        new_scale = self.get_current_scale()
        if anchor_img is not None:
            self.view_center = QPointF(
                anchor_img.x() - (anchor_pos.x() - self.width() / 2.0) / max(new_scale, 1e-6),
                anchor_img.y() - (anchor_pos.y() - self.height() / 2.0) / max(new_scale, 1e-6),
            )
        self.clamp_view_center()
        self.invalidate_background_cache()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.clamp_view_center()
        self.invalidate_background_cache()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(PART_MIME) or image_file_from_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(PART_MIME) or image_file_from_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(PART_MIME):
            part_id = bytes(event.mimeData().data(PART_MIME)).decode("utf-8")
            pos = self.view_to_image(event.pos())
            if pos is not None:
                self.main_window.place_part(part_id, pos.x(), pos.y())
                event.acceptProposedAction()
            return

        path = image_file_from_urls(event.mimeData())
        if path:
            self.main_window.load_base_image(path)
            event.acceptProposedAction()
            return
        event.ignore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(32, 32, 32))

        image = self.current_canvas_image()
        if image is None:
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(self.rect(), Qt.AlignCenter, "ここに元画像をドロップ\nまたは [画像]")
            self.display_rect = QRect()
            return

        self.ensure_view_center()
        scale = self.get_current_scale()
        h, w = image.shape[:2]
        if self.view_center is None:
            return

        self.display_rect = QRect(
            int(round(self.width() / 2.0 - self.view_center.x() * scale)),
            int(round(self.height() / 2.0 - self.view_center.y() * scale)),
            int(round(w * scale)),
            int(round(h * scale)),
        )

        view_w_img = self.width() / max(scale, 1e-6)
        view_h_img = self.height() / max(scale, 1e-6)
        x0_f = self.view_center.x() - view_w_img / 2.0
        y0_f = self.view_center.y() - view_h_img / 2.0
        x1_f = self.view_center.x() + view_w_img / 2.0
        y1_f = self.view_center.y() + view_h_img / 2.0

        x0 = max(0, int(math.floor(x0_f)))
        y0 = max(0, int(math.floor(y0_f)))
        x1 = min(w, int(math.ceil(x1_f)))
        y1 = min(h, int(math.ceil(y1_f)))
        if x1 > x0 and y1 > y0:
            crop = image[y0:y1, x0:x1]
            pix = QPixmap.fromImage(bgra_to_qimage(crop))
            target_x = self.width() / 2.0 - (self.view_center.x() - x0) * scale
            target_y = self.height() / 2.0 - (self.view_center.y() - y0) * scale
            target_w = (x1 - x0) * scale
            target_h = (y1 - y0) * scale
            painter.drawPixmap(QRectF(target_x, target_y, target_w, target_h), pix, QRectF(pix.rect()))

        if self.dragging_item:
            self.draw_fast_drag_item(painter)

        selected = self.main_window.selected_item()
        if self.main_window.always_show_frames and self.main_window.base_image is not None:
            for item in self.main_window.items:
                if not item.get("visible", True):
                    continue
                poly = self.item_polygon_view(item)
                if poly is None:
                    continue
                is_selected = (selected is item)
                color = QColor(255, 210, 70) if is_selected else QColor(80, 210, 255)
                width = 2 if is_selected else 1
                painter.setPen(QPen(color, width, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(poly)
        elif self.dragging_item and selected is not None and self.main_window.base_image is not None:
            poly = self.item_polygon_view(selected)
            if poly is not None:
                painter.setPen(QPen(QColor(255, 230, 80), 2, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(poly)

        painter.setPen(QColor(220, 220, 220))
        info = "素材D&Dで配置 / ホイール:拡大縮小 / 空白ドラッグ:表示移動 / 素材ドラッグ:移動 / Deleteで削除"
        painter.drawText(12, self.height() - 14, info)

    def get_cached_background_pixmap(self, image):
        # 互換用。現在はクロップ描画なので基本的に使わない。
        return None, QRect(self.display_rect)

    def invalidate_background_cache(self):
        self._bg_cache_image_id = None
        self._bg_cache_size = QSize()
        self._bg_cache_pixmap = None
        self._bg_cache_rect = QRect()

    def draw_fast_drag_item(self, painter):
        item = self.main_window.selected_item()
        if item is None or self.main_window.base_image is None or self.display_rect.isNull():
            return
        part_img = self.main_window.drag_item_image
        if part_img is None:
            return
        bh, bw = self.main_window.base_image.shape[:2]
        ih, iw = part_img.shape[:2]
        scale_x = self.display_rect.width() / float(bw)
        scale_y = self.display_rect.height() / float(bh)
        center = self.image_to_view(QPointF(item.get("x", 0.0), item.get("y", 0.0)))
        if center is None:
            return
        draw_w = max(1, int(round(iw * scale_x)))
        draw_h = max(1, int(round(ih * scale_y)))
        src_pix = self.main_window.drag_item_pixmap
        if src_pix is None:
            src_pix = QPixmap.fromImage(bgra_to_qimage(part_img))
            self.main_window.drag_item_pixmap = src_pix
        pix = src_pix.scaled(
            draw_w, draw_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        left = int(round(center.x() - pix.width() / 2.0))
        top = int(round(center.y() - pix.height() / 2.0))
        painter.drawPixmap(left, top, pix)

    def image_to_view(self, p: QPointF):
        if self.main_window.base_image is None:
            return None
        self.ensure_view_center()
        if self.view_center is None:
            return None
        scale = self.get_current_scale()
        return QPointF(
            self.width() / 2.0 + (p.x() - self.view_center.x()) * scale,
            self.height() / 2.0 + (p.y() - self.view_center.y()) * scale,
        )

    def view_to_image(self, p: QPoint):
        if self.main_window.base_image is None:
            return None
        self.ensure_view_center()
        if self.view_center is None:
            return None
        h, w = self.main_window.base_image.shape[:2]
        scale = self.get_current_scale()
        x = self.view_center.x() + (p.x() - self.width() / 2.0) / max(scale, 1e-6)
        y = self.view_center.y() + (p.y() - self.height() / 2.0) / max(scale, 1e-6)
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return QPointF(x, y)

    def item_polygon_view(self, item):
        if bool(item.get("warp_enabled", False)):
            part_img = self.main_window.render_item_image_fast(item)
            if part_img is None:
                return None
            h, w = part_img.shape[:2]
            cx = float(item.get("x", 0.0))
            cy = float(item.get("y", 0.0))
            corners = [
                QPointF(cx - w / 2.0, cy - h / 2.0),
                QPointF(cx + w / 2.0, cy - h / 2.0),
                QPointF(cx + w / 2.0, cy + h / 2.0),
                QPointF(cx - w / 2.0, cy + h / 2.0),
            ]
            pts = [self.image_to_view(p) for p in corners]
            if any(p is None for p in pts):
                return None
            return QPolygonF(pts)

        part = self.main_window.get_part_image(item.get("part_id"))
        if part is None:
            return None
        h, w = part.shape[:2]
        scale_w = item_scale_w(item)
        scale_h = item_scale_h(item)
        angle = math.radians(float(item.get("rotation", 0.0) if item.get("rotation_enabled", False) else 0.0))
        hw = w * scale_w / 2.0
        hh = h * scale_h / 2.0
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cx = float(item.get("x", 0.0))
        cy = float(item.get("y", 0.0))
        for px, py in corners:
            rx = px * cos_a + py * sin_a
            ry = -px * sin_a + py * cos_a
            vp = self.image_to_view(QPointF(cx + rx, cy + ry))
            if vp:
                pts.append(vp)
        if len(pts) != 4:
            return None
        return QPolygonF(pts)

    def wheelEvent(self, event):
        if self.main_window.base_image is None:
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            event.ignore()
            return
        factor = 1.25 ** steps
        self.set_zoom(self.zoom_scale * factor, anchor_pos=event.pos())
        event.accept()

    def mousePressEvent(self, event):
        img_pos = self.view_to_image(event.pos())
        if event.button() == Qt.RightButton:
            self.main_window.update_loupe_position(img_pos)
            return
        if event.button() != Qt.LeftButton:
            return
        if img_pos is None:
            self.main_window.select_item(None)
            return
        idx = self.main_window.hit_test_item(img_pos.x(), img_pos.y())
        self.main_window.select_item(idx)
        if idx is not None:
            item = self.main_window.items[idx]
            self.dragging_item = True
            self.drag_offset = QPointF(item["x"] - img_pos.x(), item["y"] - img_pos.y())
            self.main_window.begin_item_drag(idx)
            self.update()
            return
        self.panning_view = True
        self.pan_last_pos = event.pos()
        self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging_item:
            img_pos = self.view_to_image(event.pos())
            idx = self.main_window.selected_index
            if img_pos is None or idx is None:
                return
            item = self.main_window.items[idx]
            item["x"] = float(img_pos.x() + self.drag_offset.x())
            item["y"] = float(img_pos.y() + self.drag_offset.y())
            self.update()
            now = time.monotonic()
            if now - self._last_drag_ui_update >= 0.05:
                self._last_drag_ui_update = now
                if hasattr(self.main_window, "loupe_view"):
                    self.main_window.loupe_view.update()
                self.main_window.update_placed_list_text()
            return
        if self.panning_view and self.pan_last_pos is not None and self.view_center is not None:
            scale = self.get_current_scale()
            delta = event.pos() - self.pan_last_pos
            self.view_center = QPointF(
                self.view_center.x() - delta.x() / max(scale, 1e-6),
                self.view_center.y() - delta.y() / max(scale, 1e-6),
            )
            self.pan_last_pos = event.pos()
            self.clamp_view_center()
            self.invalidate_background_cache()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_item:
            self.dragging_item = False
            self.main_window.update_placed_list_text()
            if hasattr(self.main_window, "loupe_view"):
                self.main_window.loupe_view.update()
            self.main_window.end_item_drag()
            event.accept()
            return
        if self.panning_view:
            self.panning_view = False
            self.pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.main_window.delete_selected_item()
        else:
            super().keyPressEvent(event)


class WarpEditorCanvas(QWidget):
    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.setMinimumSize(720, 520)
        self.setMouseTracking(True)
        self.display_rect = QRect()
        self.drag_index = None
        self.panning = False
        self.pan_last_pos = None
        self.zoom_scale = 1.0
        self.center = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: #202020;")

    def preview_shape(self):
        img = self.dialog.preview_image()
        if img is None:
            return None
        return img.shape[:2]

    def ensure_center(self):
        shape = self.preview_shape()
        if shape is None:
            self.center = None
            return
        h, w = shape
        if self.center is None:
            self.center = QPointF(w / 2.0, h / 2.0)
        self.clamp_center()

    def get_fit_scale(self):
        shape = self.preview_shape()
        if shape is None:
            return 1.0
        h, w = shape
        area_w = max(1, self.width() - 8)
        area_h = max(1, self.height() - 8)
        return max(self.MIN_ZOOM, min(area_w / max(1, w), area_h / max(1, h)))

    def get_current_scale(self):
        return max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.get_fit_scale() * self.zoom_scale))

    def clamp_center(self):
        shape = self.preview_shape()
        if shape is None or self.center is None:
            return
        h, w = shape
        scale = self.get_current_scale()
        view_w = self.width() / max(scale, 1e-6)
        view_h = self.height() / max(scale, 1e-6)

        if view_w >= w:
            cx = w / 2.0
        else:
            half = view_w / 2.0
            cx = min(max(self.center.x(), half), w - half)

        if view_h >= h:
            cy = h / 2.0
        else:
            half = view_h / 2.0
            cy = min(max(self.center.y(), half), h - half)

        self.center = QPointF(cx, cy)

    def image_to_view(self, p: QPointF):
        shape = self.preview_shape()
        if shape is None:
            return None
        self.ensure_center()
        if self.center is None:
            return None
        scale = self.get_current_scale()
        return QPointF(
            self.width() / 2.0 + (p.x() - self.center.x()) * scale,
            self.height() / 2.0 + (p.y() - self.center.y()) * scale,
        )

    def view_to_image(self, p: QPoint):
        shape = self.preview_shape()
        if shape is None:
            return None
        self.ensure_center()
        if self.center is None:
            return None
        h, w = shape
        scale = self.get_current_scale()
        ix = self.center.x() + (p.x() - self.width() / 2.0) / max(scale, 1e-6)
        iy = self.center.y() + (p.y() - self.height() / 2.0) / max(scale, 1e-6)
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return QPointF(ix, iy)

    def set_zoom(self, new_zoom_scale, anchor_pos=None):
        shape = self.preview_shape()
        if shape is None:
            return
        self.ensure_center()
        old_scale = self.get_current_scale()
        if anchor_pos is not None and self.center is not None:
            anchor_img = QPointF(
                self.center.x() + (anchor_pos.x() - self.width() / 2.0) / max(old_scale, 1e-6),
                self.center.y() + (anchor_pos.y() - self.height() / 2.0) / max(old_scale, 1e-6),
            )
        else:
            anchor_img = None

        self.zoom_scale = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(new_zoom_scale)))
        new_scale = self.get_current_scale()
        if anchor_img is not None:
            self.center = QPointF(
                anchor_img.x() - (anchor_pos.x() - self.width() / 2.0) / max(new_scale, 1e-6),
                anchor_img.y() - (anchor_pos.y() - self.height() / 2.0) / max(new_scale, 1e-6),
            )
        self.clamp_center()
        self.update()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.center = None
        self.ensure_center()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.clamp_center()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(32, 32, 32))

        img = self.dialog.preview_image()
        if img is None:
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(self.rect(), Qt.AlignCenter, "プレビューを作成できません")
            return

        self.ensure_center()
        scale = self.get_current_scale()
        qimg = bgra_to_qimage(img)
        pix = QPixmap.fromImage(qimg)
        target = QRectF(
            self.width() / 2.0 - self.center.x() * scale,
            self.height() / 2.0 - self.center.y() * scale,
            pix.width() * scale,
            pix.height() * scale,
        )
        painter.drawPixmap(target, pix, QRectF(pix.rect()))

        view_points = []
        for idx in range(WARP_GRID_ROWS * WARP_GRID_COLS):
            p = self.dialog.control_point_image_pos(idx)
            vp = self.image_to_view(p) if p is not None else None
            view_points.append(vp)

        pen_line = QPen(QColor(80, 210, 255), 1, Qt.SolidLine)
        painter.setPen(pen_line)
        for r in range(WARP_GRID_ROWS):
            for c in range(WARP_GRID_COLS - 1):
                a = view_points[r * WARP_GRID_COLS + c]
                b = view_points[r * WARP_GRID_COLS + c + 1]
                if a is not None and b is not None:
                    painter.drawLine(a, b)
        for c in range(WARP_GRID_COLS):
            for r in range(WARP_GRID_ROWS - 1):
                a = view_points[r * WARP_GRID_COLS + c]
                b = view_points[(r + 1) * WARP_GRID_COLS + c]
                if a is not None and b is not None:
                    painter.drawLine(a, b)

        for idx, vp in enumerate(view_points):
            if vp is None:
                continue
            selected = idx == self.drag_index
            radius = 4 if selected else 3
            color = QColor(255, 230, 80) if selected else QColor(80, 210, 255)
            painter.setPen(QPen(QColor(15, 15, 15), 1))
            painter.setBrush(color)
            painter.drawEllipse(vp, radius, radius)

        painter.setPen(QColor(230, 230, 230))
        painter.drawText(10, self.height() - 12, "制御点をドラッグ / ホイール:拡大縮小 / 空白ドラッグ:表示移動 / R:リセット / Enter:適用 / Esc:キャンセル")

    def nearest_control_point(self, pos):
        best_idx = None
        best_dist = 10.0
        for idx in range(WARP_GRID_ROWS * WARP_GRID_COLS):
            p = self.dialog.control_point_image_pos(idx)
            vp = self.image_to_view(p) if p is not None else None
            if vp is None:
                continue
            dx = vp.x() - pos.x()
            dy = vp.y() - pos.y()
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            event.ignore()
            return
        factor = 1.25 ** steps
        self.set_zoom(self.zoom_scale * factor, anchor_pos=event.pos())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self.nearest_control_point(event.pos())
            if idx is not None:
                self.drag_index = idx
                self.dialog.set_dragged_point_from_image(idx, self.view_to_image(event.pos()))
                self.update()
                event.accept()
                return
            self.panning = True
            self.pan_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_index is not None:
            self.dialog.set_dragged_point_from_image(self.drag_index, self.view_to_image(event.pos()))
            self.update()
            event.accept()
            return
        if self.panning and self.pan_last_pos is not None and self.center is not None:
            scale = self.get_current_scale()
            delta = event.pos() - self.pan_last_pos
            self.center = QPointF(
                self.center.x() - delta.x() / max(scale, 1e-6),
                self.center.y() - delta.y() / max(scale, 1e-6),
            )
            self.pan_last_pos = event.pos()
            self.clamp_center()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_index is not None:
            self.dialog.set_dragged_point_from_image(self.drag_index, self.view_to_image(event.pos()))
            self.drag_index = None
            self.update()
            event.accept()
            return
        if self.panning:
            self.panning = False
            self.pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.dialog.reset_points()
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.dialog.accept()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.dialog.reject()
            event.accept()
            return
        super().keyPressEvent(event)


class WarpDialog(QDialog):
    def __init__(self, main_window, item_index):
        super().__init__(main_window)
        self.main_window = main_window
        self.item_index = item_index
        self.setWindowTitle("ワープ編集")
        self.resize(980, 720)
        self.points = get_item_warp_points(self.item()).copy()
        self._preview_cache = None
        self._preview_cache_key = None

        layout = QVBoxLayout(self)
        self.canvas = WarpEditorCanvas(self)
        layout.addWidget(self.canvas, 1)

        button_row = QHBoxLayout()
        self.reset_btn = QPushButton("リセット")
        self.reset_btn.clicked.connect(self.reset_points)
        self.apply_btn = QPushButton("適用")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(self.reset_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.apply_btn)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)

    def item(self):
        return self.main_window.items[self.item_index]

    def draft_item(self):
        draft = dict(self.item())
        draft["warp_enabled"] = True
        draft["warp_grid_rows"] = WARP_GRID_ROWS
        draft["warp_grid_cols"] = WARP_GRID_COLS
        draft["warp_points"] = [[float(x), float(y)] for x, y in self.points]
        return draft

    def reset_points(self):
        self.points = default_warp_points()
        self.invalidate_preview()
        self.canvas.update()

    def invalidate_preview(self):
        self._preview_cache = None
        self._preview_cache_key = None

    def preview_image(self):
        base = self.main_window.base_image
        if base is None:
            return None
        key = tuple((round(x, 4), round(y, 4)) for x, y in self.points)
        if self._preview_cache is not None and key == self._preview_cache_key:
            return self._preview_cache
        canvas = base.copy()
        item = self.draft_item()
        part_img = self.main_window.render_item_image(item, base)
        if part_img is not None:
            ih, iw = part_img.shape[:2]
            left = int(round(float(item.get("x", 0.0)) - iw / 2.0))
            top = int(round(float(item.get("y", 0.0)) - ih / 2.0))
            alpha_composite_bgra(canvas, part_img, left, top)
        self._preview_cache = canvas
        self._preview_cache_key = key
        return canvas

    def pre_warp_image(self):
        return self.main_window.render_item_pre_warp(self.draft_item(), self.main_window.base_image)

    def warp_display_geometry(self):
        pre = self.pre_warp_image()
        if pre is None:
            return None
        h, w = pre.shape[:2]
        geom = warp_geometry(w, h, self.points)
        angle = float(self.item().get("rotation", 0.0)) if bool(self.item().get("rotation_enabled", False)) else 0.0
        matrix, final_w, final_h = rotation_matrix_for_image(geom["out_w"], geom["out_h"], angle)
        left = float(self.item().get("x", 0.0)) - final_w / 2.0
        top = float(self.item().get("y", 0.0)) - final_h / 2.0
        return pre.shape, geom, matrix, final_w, final_h, left, top

    def control_point_image_pos(self, idx):
        data = self.warp_display_geometry()
        if data is None:
            return None
        _, geom, matrix, _, _, left, top = data
        p = geom["shifted_dst_pts"][idx]
        rx = float(matrix[0, 0] * p[0] + matrix[0, 1] * p[1] + matrix[0, 2])
        ry = float(matrix[1, 0] * p[0] + matrix[1, 1] * p[1] + matrix[1, 2])
        return QPointF(left + rx, top + ry)

    def set_dragged_point_from_image(self, idx, image_pos):
        if image_pos is None:
            return
        data = self.warp_display_geometry()
        if data is None:
            return
        pre_shape, geom, matrix, _, _, left, top = data
        h, w = pre_shape[:2]
        local_x = float(image_pos.x()) - left
        local_y = float(image_pos.y()) - top
        m3 = np.array([[matrix[0, 0], matrix[0, 1], matrix[0, 2]], [matrix[1, 0], matrix[1, 1], matrix[1, 2]], [0.0, 0.0, 1.0]], dtype=np.float32)
        inv = np.linalg.inv(m3)
        wx, wy, _ = inv.dot(np.array([local_x, local_y, 1.0], dtype=np.float32))
        raw_x = float(wx) - float(geom["shift_x"])
        raw_y = float(wy) - float(geom["shift_y"])
        nx = raw_x / max(1.0, float(w - 1))
        ny = raw_y / max(1.0, float(h - 1))
        self.points[idx] = [float(nx), float(ny)]
        self.invalidate_preview()

    def accept(self):
        item = self.item()
        item["warp_enabled"] = True
        item["warp_grid_rows"] = WARP_GRID_ROWS
        item["warp_grid_cols"] = WARP_GRID_COLS
        item["warp_points"] = [[float(x), float(y)] for x, y in self.points]
        self.main_window.update_controls_from_item()
        self.main_window.refresh_preview(save=True)
        super().accept()


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------

class ImageCompositor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.setStyleSheet("QGroupBox { font-weight: bold; } QGroupBox::title { font-weight: bold; subcontrol-origin: margin; left: 8px; padding: 0 2px 0 2px; }")

        self.settings_path = script_dir() / SETTINGS_FILE
        self.parts_root_dir = script_dir() / PARTS_DIR
        self.parts_root_dir.mkdir(exist_ok=True)
        self.current_parts_folder = DEFAULT_PARTS_FOLDER
        self.parts_dir = self.parts_root_dir / self.current_parts_folder
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        self.base_image_path = None
        self.base_image = None
        self.preview_image = None
        self.parts = []
        self.items = []
        self.selected_index = None
        self.part_cache = {}
        self.loading_ui = True
        self.loupe_image_pos = None
        self.drag_preview_image = None
        self.dragging_index = None
        self.drag_item_image = None
        self.drag_item_pixmap = None
        self.placed_list_pressed = False
        self.always_show_frames = False
        self._closing_app = False

        self.build_ui()
        self.load_settings()
        self.refresh_part_folders(keep_selection=True)
        self.refresh_parts_list()
        self.refresh_placed_list()
        self.refresh_preview(save=False)
        self.loading_ui = False

    # ---------------- UI ----------------
    def build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        splitter.splitterMoved.connect(lambda *args: None if self.loading_ui else self.save_settings())
        root.addWidget(splitter)
        self.setCentralWidget(central)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # 上部ボタン群
        top_groups_layout = QHBoxLayout()

        file_group = QGroupBox("ファイル")
        file_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(8, 14, 8, 6)
        file_layout.setSpacing(6)
        self.open_base_btn = QPushButton("画像")
        self.open_base_btn.clicked.connect(self.open_base_dialog)
        self.add_part_btn = QPushButton("素材")
        self.add_part_btn.clicked.connect(self.open_parts_dialog)
        self.open_project_btn = QPushButton("P開")
        self.open_project_btn.clicked.connect(self.open_project_dialog)
        self.save_project_btn = QPushButton("P保")
        self.save_project_btn.clicked.connect(self.save_project_dialog)
        self.export_btn = QPushButton("書出")
        self.export_btn.clicked.connect(self.export_dialog)
        for btn in [self.open_base_btn, self.add_part_btn, self.open_project_btn, self.save_project_btn, self.export_btn]:
            btn.setFixedWidth(44)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        file_layout.addWidget(self.open_base_btn)
        file_layout.addWidget(self.add_part_btn)
        file_layout.addWidget(self.open_project_btn)
        file_layout.addWidget(self.save_project_btn)
        file_layout.addWidget(self.export_btn)
        top_groups_layout.addWidget(file_group, 0)

        view_group = QGroupBox("表示")
        view_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        view_layout = QHBoxLayout(view_group)
        view_layout.setContentsMargins(8, 14, 8, 6)
        view_layout.setSpacing(6)
        self.loupe_toggle_btn = QPushButton("ル")
        self.loupe_toggle_btn.setCheckable(True)
        self.loupe_toggle_btn.toggled.connect(self.on_loupe_toggle_clicked)
        self.frame_toggle_btn = QPushButton("枠")
        self.frame_toggle_btn.setCheckable(True)
        self.frame_toggle_btn.toggled.connect(self.on_frame_toggle_clicked)
        toggle_style = "QPushButton:checked { background-color: #3f7cff; color: white; border: 1px solid #2f58b8; }"
        self.loupe_toggle_btn.setStyleSheet(toggle_style)
        self.frame_toggle_btn.setStyleSheet(toggle_style)
        for btn in [self.loupe_toggle_btn, self.frame_toggle_btn]:
            btn.setFixedWidth(32)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        view_layout.addWidget(self.loupe_toggle_btn)
        view_layout.addWidget(self.frame_toggle_btn)
        top_groups_layout.addWidget(view_group, 0)
        top_groups_layout.addStretch(1)

        left_layout.addLayout(top_groups_layout)

        self.left_vertical_splitter = QSplitter(Qt.Vertical)
        self.left_vertical_splitter.setChildrenCollapsible(False)
        self.left_vertical_splitter.splitterMoved.connect(lambda *args: None if self.loading_ui else self.save_settings())

        # 素材一覧
        parts_group = QGroupBox("素材一覧（ここへD&Dで追加）")
        parts_layout = QVBoxLayout(parts_group)
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("フォルダ"))
        self.parts_folder_combo = PartsFolderComboBox(self)
        self.parts_folder_combo.currentIndexChanged.connect(self.on_parts_folder_changed)
        folder_row.addWidget(self.parts_folder_combo, 1)
        self.open_parts_folder_btn = QPushButton("開く")
        self.open_parts_folder_btn.clicked.connect(self.open_current_parts_folder)
        folder_row.addWidget(self.open_parts_folder_btn)
        parts_layout.addLayout(folder_row)
        self.parts_list = PartsListWidget(self)
        self.parts_list.itemDoubleClicked.connect(self.place_selected_part_center)
        parts_layout.addWidget(self.parts_list)
        self.left_vertical_splitter.addWidget(parts_group)

        # 配置済み素材
        placed_group = QGroupBox("配置済み素材")
        placed_layout = QVBoxLayout(placed_group)
        self.placed_list = PlacedListWidget(self)
        self.placed_list.currentRowChanged.connect(self.on_placed_row_changed)
        self.placed_list.itemChanged.connect(self.on_placed_item_changed)
        placed_layout.addWidget(self.placed_list)
        placed_buttons = QHBoxLayout()
        self.delete_item_btn = QPushButton("削除")
        self.delete_item_btn.clicked.connect(self.delete_selected_item)
        self.duplicate_item_btn = QPushButton("複製")
        self.duplicate_item_btn.clicked.connect(self.duplicate_selected_item)
        self.front_btn = QPushButton("前面")
        self.front_btn.clicked.connect(self.move_selected_front)
        self.back_btn = QPushButton("背面")
        self.back_btn.clicked.connect(self.move_selected_back)
        placed_buttons.addWidget(self.delete_item_btn)
        placed_buttons.addWidget(self.duplicate_item_btn)
        placed_buttons.addWidget(self.back_btn)
        placed_buttons.addWidget(self.front_btn)
        placed_layout.addLayout(placed_buttons)
        self.left_vertical_splitter.addWidget(placed_group)

        # 設定パネルはスクロール
        settings_group = QGroupBox("選択中素材の合成方法")
        settings_layout = QVBoxLayout(settings_group)
        settings_inner = QWidget()
        settings_form = QVBoxLayout(settings_inner)
        settings_form.setContentsMargins(4, 4, 4, 4)

        # サイズ
        scale_box = QGroupBox("サイズ")
        scale_form = QFormLayout(scale_box)
        scale_row = QWidget()
        scale_row_layout = QHBoxLayout(scale_row)
        scale_row_layout.setContentsMargins(0, 0, 0, 0)
        scale_row_layout.setSpacing(6)

        self.scale_w_spin = QDoubleSpinBox()
        self.scale_w_spin.setRange(0.02, 10.0)
        self.scale_w_spin.setSingleStep(0.05)
        self.scale_w_spin.setDecimals(2)
        self.scale_w_spin.setValue(1.0)
        self.scale_w_spin.valueChanged.connect(self.on_scale_w_changed)

        self.scale_h_spin = QDoubleSpinBox()
        self.scale_h_spin.setRange(0.02, 10.0)
        self.scale_h_spin.setSingleStep(0.05)
        self.scale_h_spin.setDecimals(2)
        self.scale_h_spin.setValue(1.0)
        self.scale_h_spin.valueChanged.connect(self.on_scale_h_changed)

        self.scale_lock_check = QCheckBox("固定")
        self.scale_lock_check.setChecked(True)
        self.scale_lock_check.toggled.connect(self.on_scale_lock_changed)

        scale_row_layout.addWidget(QLabel("W"))
        scale_row_layout.addWidget(self.scale_w_spin, 1)
        scale_row_layout.addWidget(QLabel("H"))
        scale_row_layout.addWidget(self.scale_h_spin, 1)
        scale_row_layout.addWidget(self.scale_lock_check)
        scale_form.addRow("倍率", scale_row)
        settings_form.addWidget(scale_box)

        # 輪郭
        feather_box = QGroupBox("輪郭なじませ")
        feather_box.setCheckable(True)
        feather_box.setChecked(False)
        feather_box.toggled.connect(self.controls_to_item)
        self.feather_check = feather_box
        feather_form = QFormLayout(feather_box)
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 300)
        self.feather_spin.setValue(8)
        self.feather_spin.valueChanged.connect(self.controls_to_item)
        feather_form.addRow("幅 px", self.feather_spin)
        settings_form.addWidget(feather_box)

        # 透明度
        opacity_box = QGroupBox("透明度")
        opacity_box.setCheckable(True)
        opacity_box.setChecked(False)
        opacity_box.toggled.connect(self.controls_to_item)
        self.opacity_check = opacity_box
        opacity_form = QFormLayout(opacity_box)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_slider)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.valueChanged.connect(self.on_opacity_spin)
        opacity_form.addRow("不透明度 %", self.opacity_slider)
        opacity_form.addRow("値", self.opacity_spin)
        settings_form.addWidget(opacity_box)

        # 回転
        rot_box = QGroupBox("回転")
        rot_box.setCheckable(True)
        rot_box.setChecked(False)
        rot_box.toggled.connect(self.controls_to_item)
        self.rotation_check = rot_box
        rot_form = QFormLayout(rot_box)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-360.0, 360.0)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.valueChanged.connect(self.controls_to_item)
        rot_form.addRow("角度", self.rotation_spin)
        settings_form.addWidget(rot_box)

        # 反転
        flip_box = QGroupBox("反転")
        flip_form = QFormLayout(flip_box)
        self.flip_h_check = QCheckBox("左右反転")
        self.flip_v_check = QCheckBox("上下反転")
        self.flip_h_check.toggled.connect(self.controls_to_item)
        self.flip_v_check.toggled.connect(self.controls_to_item)
        flip_form.addRow(self.flip_h_check)
        flip_form.addRow(self.flip_v_check)
        settings_form.addWidget(flip_box)

        # ワープ
        warp_box = QGroupBox("ワープ")
        warp_box.setCheckable(True)
        warp_box.setChecked(False)
        warp_box.toggled.connect(self.controls_to_item)
        self.warp_check = warp_box
        warp_layout = QHBoxLayout(warp_box)
        self.warp_edit_btn = QPushButton("ワープ編集")
        self.warp_edit_btn.clicked.connect(self.open_warp_dialog)
        self.warp_reset_btn = QPushButton("リセット")
        self.warp_reset_btn.clicked.connect(self.reset_selected_warp)
        warp_layout.addWidget(self.warp_edit_btn)
        warp_layout.addWidget(self.warp_reset_btn)
        settings_form.addWidget(warp_box)

        # 色合わせ
        color_box = QGroupBox("色を合わせる")
        color_box.setCheckable(True)
        color_box.setChecked(False)
        color_box.toggled.connect(self.controls_to_item)
        self.color_check = color_box
        color_form = QFormLayout(color_box)
        self.color_target_combo = QComboBox()
        self.color_target_combo.addItems(["元画像全体", "配置場所"])
        self.color_target_combo.currentIndexChanged.connect(self.controls_to_item)
        self.color_algo_combo = QComboBox()
        self.color_algo_combo.addItems([
            "1. 標準 (全体平均)",
            "2. 白黒除外 (輝度マスク)",
            "3. 主要色抽出 (K-Means)",
            "4. ヒストグラムマッチング",
            "5. 明度保持 (色味のみ)",
            "6. 明暗別マッチング",
            "7. 安定補正 (外れ値除外)",
        ])
        self.color_algo_combo.setCurrentIndex(1)
        self.color_algo_combo.currentIndexChanged.connect(self.controls_to_item)
        self.color_strength_spin = QSpinBox()
        self.color_strength_spin.setRange(0, 100)
        self.color_strength_spin.setValue(60)
        self.color_strength_spin.valueChanged.connect(self.controls_to_item)
        self.local_margin_spin = QSpinBox()
        self.local_margin_spin.setRange(0, 1000)
        self.local_margin_spin.setValue(0)
        self.local_margin_spin.valueChanged.connect(self.controls_to_item)
        color_form.addRow("参照", self.color_target_combo)
        color_form.addRow("モード", self.color_algo_combo)
        color_form.addRow("強度 %", self.color_strength_spin)
        color_form.addRow("拡張幅 px", self.local_margin_spin)
        settings_form.addWidget(color_box)

        # 影
        shadow_box = QGroupBox("影")
        shadow_box.setCheckable(True)
        shadow_box.setChecked(False)
        shadow_box.toggled.connect(self.controls_to_item)
        self.shadow_check = shadow_box
        shadow_form = QFormLayout(shadow_box)
        self.shadow_opacity_spin = QSpinBox()
        self.shadow_opacity_spin.setRange(0, 100)
        self.shadow_opacity_spin.setValue(40)
        self.shadow_opacity_spin.valueChanged.connect(self.controls_to_item)
        self.shadow_blur_spin = QSpinBox()
        self.shadow_blur_spin.setRange(0, 100)
        self.shadow_blur_spin.setValue(12)
        self.shadow_blur_spin.valueChanged.connect(self.controls_to_item)
        self.shadow_x_spin = QSpinBox()
        self.shadow_x_spin.setRange(-500, 500)
        self.shadow_x_spin.setValue(6)
        self.shadow_x_spin.valueChanged.connect(self.controls_to_item)
        self.shadow_y_spin = QSpinBox()
        self.shadow_y_spin.setRange(-500, 500)
        self.shadow_y_spin.setValue(8)
        self.shadow_y_spin.valueChanged.connect(self.controls_to_item)
        shadow_form.addRow("濃さ %", self.shadow_opacity_spin)
        shadow_form.addRow("ぼかし px", self.shadow_blur_spin)
        shadow_form.addRow("X", self.shadow_x_spin)
        shadow_form.addRow("Y", self.shadow_y_spin)
        settings_form.addWidget(shadow_box)

        settings_form.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_inner)
        settings_layout.addWidget(scroll)
        self.left_vertical_splitter.addWidget(settings_group)
        self.left_vertical_splitter.setStretchFactor(0, 4)
        self.left_vertical_splitter.setStretchFactor(1, 1)
        self.left_vertical_splitter.setStretchFactor(2, 5)
        self.left_vertical_splitter.setSizes([360, 90, 470])
        left_layout.addWidget(self.left_vertical_splitter, 1)

        self.canvas = CanvasWidget(self)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1020])

        self.statusBar().showMessage("元画像と素材をドラッグ＆ドロップできます / 素材は _parts の選択フォルダに入ります")
        self.resize(1380, 920)

        self.loupe_window = LoupeToolWindow(self)
        self.loupe_window.show()
        self.loupe_window.raise_()
        self.sync_view_buttons(save=False)

    def update_loupe_position(self, img_pos):
        self.loupe_image_pos = img_pos
        if hasattr(self, "loupe_view"):
            self.loupe_view.update()

    def on_loupe_zoom_changed(self, value):
        if hasattr(self, "loupe_label"):
            self.loupe_label.setText(f"{int(value)}%")
        if hasattr(self, "loupe_view"):
            self.loupe_view.update()
        if not self.loading_ui:
            self.save_settings()

    def adjust_loupe_zoom_by_wheel(self, event):
        if not hasattr(self, "loupe_slider"):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        steps = max(1, abs(delta) // 120)
        direction = 1 if delta > 0 else -1
        slider = self.loupe_slider
        slider.setValue(max(slider.minimum(), min(slider.maximum(), slider.value() + direction * 25 * steps)))
        event.accept()

    def set_loupe_visible(self, visible, save=True):
        if not hasattr(self, "loupe_window"):
            return
        if visible:
            self.loupe_window.show()
            self.loupe_window.raise_()
            self.loupe_window.activateWindow()
        else:
            self.loupe_window.hide()
        self.sync_view_buttons(save=save)

    def on_loupe_toggle_clicked(self, checked):
        self.set_loupe_visible(bool(checked), save=True)

    def on_frame_toggle_clicked(self, checked):
        self.always_show_frames = bool(checked)
        self.sync_view_buttons(save=True)
        if hasattr(self, "canvas"):
            self.canvas.update()

    def sync_view_buttons(self, save=False):
        if hasattr(self, "loupe_toggle_btn") and hasattr(self, "loupe_window"):
            self.loupe_toggle_btn.blockSignals(True)
            self.loupe_toggle_btn.setChecked(self.loupe_window.isVisible())
            self.loupe_toggle_btn.blockSignals(False)
        if hasattr(self, "frame_toggle_btn"):
            self.frame_toggle_btn.blockSignals(True)
            self.frame_toggle_btn.setChecked(bool(self.always_show_frames))
            self.frame_toggle_btn.blockSignals(False)
        if save and not getattr(self, "loading_ui", False):
            self.save_settings()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_view_buttons(save=False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "canvas"):
            self.canvas.invalidate_background_cache()
        if not getattr(self, "loading_ui", False):
            self.save_settings()

    def moveEvent(self, event):
        super().moveEvent(event)
        if not getattr(self, "loading_ui", False):
            self.save_settings()

    def dragEnterEvent(self, event):
        if image_files_from_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if image_files_from_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = image_files_from_urls(event.mimeData())
        if not paths:
            super().dropEvent(event)
            return

        # 子ウィジェットが取りこぼした時の保険。
        # キャンバス上なら元画像、左列/素材一覧側なら素材登録にする。
        canvas_pos = self.canvas.mapFrom(self, event.pos()) if hasattr(self, "canvas") else None
        if canvas_pos is not None and self.canvas.rect().contains(canvas_pos):
            self.load_base_image(paths[0])
        else:
            self.add_part_files(paths)
        event.acceptProposedAction()

    # ---------------- settings ----------------
    def default_item(self, part_id, x, y):
        return {
            "uid": str(uuid.uuid4()),
            "part_id": part_id,
            "x": float(x),
            "y": float(y),
            "visible": True,
            "scale": float(self.scale_w_spin.value()),
            "scale_w": float(self.scale_w_spin.value()),
            "scale_h": float(self.scale_h_spin.value()),
            "scale_lock": bool(self.scale_lock_check.isChecked()),
            "feather_enabled": bool(self.feather_check.isChecked()),
            "feather_px": int(self.feather_spin.value()),
            "opacity_enabled": bool(self.opacity_check.isChecked()),
            "opacity_percent": int(self.opacity_spin.value()),
            "rotation_enabled": bool(self.rotation_check.isChecked()),
            "rotation": float(self.rotation_spin.value()),
            "flip_h": bool(self.flip_h_check.isChecked()),
            "flip_v": bool(self.flip_v_check.isChecked()),
            "warp_enabled": bool(self.warp_check.isChecked()),
            "warp_grid_rows": WARP_GRID_ROWS,
            "warp_grid_cols": WARP_GRID_COLS,
            "warp_points": default_warp_points(),
            "color_enabled": bool(self.color_check.isChecked()),
            "color_target": "local" if self.color_target_combo.currentIndex() == 1 else "global",
            "color_algo": int(self.color_algo_combo.currentIndex()),
            "color_strength": int(self.color_strength_spin.value()),
            "local_margin": int(self.local_margin_spin.value()),
            "shadow_enabled": bool(self.shadow_check.isChecked()),
            "shadow_opacity": int(self.shadow_opacity_spin.value()),
            "shadow_blur": int(self.shadow_blur_spin.value()),
            "shadow_x": int(self.shadow_x_spin.value()),
            "shadow_y": int(self.shadow_y_spin.value()),
        }

    def load_settings(self):
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"settings load error: {exc}")
            return

        geom = data.get("window") or {}
        if geom.get("width") and geom.get("height"):
            self.resize(int(geom.get("width")), int(geom.get("height")))
        if geom.get("x") is not None and geom.get("y") is not None:
            self.move(int(geom.get("x")), int(geom.get("y")))

        if data.get("loupe_zoom") is not None:
            self.loupe_slider.setValue(int(data.get("loupe_zoom", 300)))
        self.always_show_frames = bool(data.get("always_show_frames", False))
        lw = data.get("loupe_window") or {}
        if hasattr(self, "loupe_window"):
            if lw.get("width") and lw.get("height"):
                self.loupe_window.resize(int(lw.get("width")), int(lw.get("height")))
            if lw.get("x") is not None and lw.get("y") is not None:
                self.loupe_window.move(int(lw.get("x")), int(lw.get("y")))
            if lw.get("visible", True):
                self.loupe_window.show()
                self.loupe_window.raise_()
                self.loupe_window.activateWindow()
            else:
                self.loupe_window.hide()

        sizes = data.get("left_splitter_sizes")
        if sizes and hasattr(self, "left_vertical_splitter"):
            try:
                self.left_vertical_splitter.setSizes([int(x) for x in sizes])
            except Exception:
                pass
        main_sizes = data.get("main_splitter_sizes")
        if main_sizes and hasattr(self, "main_splitter"):
            try:
                self.main_splitter.setSizes([int(x) for x in main_sizes])
            except Exception:
                pass

        folder = data.get("current_parts_folder") or DEFAULT_PARTS_FOLDER
        self.current_parts_folder = self.normalize_parts_folder_name(folder)
        self.parts_dir = self.parts_root_dir / self.current_parts_folder
        self.parts_dir.mkdir(parents=True, exist_ok=True)

        self.parts = []
        # 起動時は元画像・配置済み素材を自動復元しない。
        self.items = []
        self.selected_index = None
        self.base_image_path = None
        self.base_image = None
        self.preview_image = None
        self.sync_view_buttons(save=False)

    def save_settings(self):
        if getattr(self, "loading_ui", False):
            return
        try:
            data = {
                "window": {"x": self.x(), "y": self.y(), "width": self.width(), "height": self.height()},
                "current_parts_folder": self.current_parts_folder,
                "loupe_zoom": int(self.loupe_slider.value()) if hasattr(self, "loupe_slider") else 300,
                "loupe_window": {
                    "x": self.loupe_window.x() if hasattr(self, "loupe_window") else None,
                    "y": self.loupe_window.y() if hasattr(self, "loupe_window") else None,
                    "width": self.loupe_window.width() if hasattr(self, "loupe_window") else None,
                    "height": self.loupe_window.height() if hasattr(self, "loupe_window") else None,
                    "visible": self.loupe_window.isVisible() if hasattr(self, "loupe_window") else False,
                },
                "left_splitter_sizes": self.left_vertical_splitter.sizes() if hasattr(self, "left_vertical_splitter") else [],
                "main_splitter_sizes": self.main_splitter.sizes() if hasattr(self, "main_splitter") else [],
                "always_show_frames": bool(self.always_show_frames),
            }
            self.settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"settings save error: {exc}")

    def closeEvent(self, event):
        self._closing_app = True
        self.save_settings()
        super().closeEvent(event)

    # ---------------- file ops ----------------
    def open_base_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "元画像を開く",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self.load_base_image(path)

    def open_parts_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "素材を追加",
            "",
            "画像ファイル (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if paths:
            self.add_part_files(paths)

    def load_base_image(self, path, save=True):
        img = imread_japanese(path)
        if img is None:
            QMessageBox.warning(self, "読み込み失敗", "元画像を読み込めませんでした。")
            return
        self.base_image_path = path
        self.base_image = ensure_bgra(img)
        if hasattr(self, "canvas"):
            self.canvas.reset_view()
        self.statusBar().showMessage(f"元画像: {path}")
        self.refresh_preview(save=False)
        if save:
            self.save_settings()

    def normalize_parts_folder_name(self, name):
        name = str(name or DEFAULT_PARTS_FOLDER).replace("\\", "/").strip("/")
        # コンボ対象は _parts 直下のフォルダだけ。危ない相対パスは default に戻す。
        if not name or "/" in name or name in {".", ".."}:
            return DEFAULT_PARTS_FOLDER
        return name

    def part_rel_id(self, folder, filename):
        return f"{self.normalize_parts_folder_name(folder)}/{Path(filename).name}"

    def refresh_part_folders(self, keep_selection=True):
        self.parts_root_dir.mkdir(exist_ok=True)
        (self.parts_root_dir / DEFAULT_PARTS_FOLDER).mkdir(parents=True, exist_ok=True)
        current = self.current_parts_folder if keep_selection else DEFAULT_PARTS_FOLDER
        folders = [p.name for p in self.parts_root_dir.iterdir() if p.is_dir()]
        folders = sorted(set(folders), key=lambda x: (x != DEFAULT_PARTS_FOLDER, x.lower()))
        if current not in folders:
            current = DEFAULT_PARTS_FOLDER
        if hasattr(self, "parts_folder_combo"):
            self.parts_folder_combo.blockSignals(True)
            self.parts_folder_combo.clear()
            for name in folders:
                self.parts_folder_combo.addItem(f"_parts/{name}", name)
            idx = self.parts_folder_combo.findData(current)
            if idx >= 0:
                self.parts_folder_combo.setCurrentIndex(idx)
            self.parts_folder_combo.blockSignals(False)
        self.current_parts_folder = current
        self.parts_dir = self.parts_root_dir / self.current_parts_folder
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    def on_parts_folder_changed(self, index):
        if self.loading_ui or index < 0:
            return
        folder = self.parts_folder_combo.itemData(index) or self.parts_folder_combo.currentText()
        self.set_current_parts_folder(folder)

    def set_current_parts_folder(self, folder, save=True):
        folder = self.normalize_parts_folder_name(folder)
        path = self.parts_root_dir / folder
        if not path.exists() or not path.is_dir():
            folder = DEFAULT_PARTS_FOLDER
            path = self.parts_root_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        self.current_parts_folder = folder
        self.parts_dir = path
        self.refresh_part_folders(keep_selection=True)
        self.refresh_parts_list()
        if save:
            self.save_settings()
        self.statusBar().showMessage(f"素材フォルダ: _parts/{self.current_parts_folder}")

    def open_current_parts_folder(self):
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.parts_dir.resolve())))

    def scan_current_parts_folder(self):
        """現在の素材フォルダにある画像を自動登録扱いにする。"""
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for path in sorted(self.parts_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel_id = self.part_rel_id(self.current_parts_folder, path.name)
            parts.append({
                "id": rel_id,
                "file": str(Path(PARTS_DIR) / self.current_parts_folder / path.name),
                "folder": self.current_parts_folder,
                "original_name": path.name,
            })
        self.parts = parts

    def next_part_id(self):
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            filename = f"{n:03d}.png"
            if not (self.parts_dir / filename).exists():
                return f"{n:03d}"
            n += 1

    def add_part_files(self, paths):
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for path in paths:
            src_path = Path(path)
            img = imread_japanese(str(src_path))
            if img is None:
                continue
            pid = self.next_part_id()
            out_path = self.parts_dir / f"{pid}.png"
            bgra = ensure_bgra(img)
            if not imwrite_japanese(str(out_path), bgra):
                try:
                    shutil.copy2(str(src_path), str(out_path))
                except Exception:
                    continue
            rel_id = self.part_rel_id(self.current_parts_folder, out_path.name)
            self.part_cache[rel_id] = bgra
            added += 1
        if added:
            self.refresh_parts_list()
            self.save_settings()
            self.statusBar().showMessage(f"_parts/{self.current_parts_folder} に素材を追加: {added}個")

    def save_project_dialog(self):
        if self.base_image is None:
            QMessageBox.warning(self, "保存不可", "元画像がありません。")
            return
        default_dir = str(Path(self.base_image_path).parent) if self.base_image_path else str(script_dir())
        default_name = "project.icp.json"
        if self.base_image_path:
            p = Path(self.base_image_path)
            default_name = f"{p.stem}.icp.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "プロジェクトを保存",
            str(Path(default_dir) / default_name),
            "Image Compositor Project (*.json)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        data = {
            "project_version": 1,
            "base_image_path": self.base_image_path,
            "current_parts_folder": self.current_parts_folder,
            "items": self.items,
        }
        try:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"プロジェクト保存: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失敗", f"プロジェクト保存に失敗しました。\n{exc}")

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "プロジェクトを開く",
            "",
            "Image Compositor Project (*.json)",
        )
        if path:
            self.load_project(path)

    def load_project(self, path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "読み込み失敗", f"プロジェクトを読み込めませんでした。\n{exc}")
            return

        base_path = data.get("base_image_path")
        if not base_path or not Path(base_path).exists():
            QMessageBox.warning(self, "読み込み失敗", "プロジェクトの元画像が見つかりません。")
            return

        folder = data.get("current_parts_folder") or DEFAULT_PARTS_FOLDER
        self.set_current_parts_folder(folder, save=False)
        self.items = data.get("items", []) or []
        self.selected_index = 0 if self.items else None
        self.load_base_image(base_path, save=False)
        self.refresh_placed_list()
        self.refresh_preview(save=False)
        self.save_settings()
        self.statusBar().showMessage(f"プロジェクト読込: {path}")

    def export_dialog(self):
        if self.base_image is None:
            QMessageBox.warning(self, "書き出し不可", "元画像がありません。")
            return
        default_dir = str(Path(self.base_image_path).parent) if self.base_image_path else str(script_dir())
        default_name = "composited.png"
        if self.base_image_path:
            p = Path(self.base_image_path)
            default_name = f"{p.stem}.composited.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "書き出し",
            str(Path(default_dir) / default_name),
            "PNG (*.png);;JPEG (*.jpg);;WEBP (*.webp)",
        )
        if not path:
            return
        if not Path(path).suffix:
            if "JPEG" in selected_filter:
                path += ".jpg"
            elif "WEBP" in selected_filter:
                path += ".webp"
            else:
                path += ".png"
        out = self.render_composite(full_quality=True)
        ext = Path(path).suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            # JPEGはアルファ非対応なので白背景に合成。
            white = np.zeros_like(out)
            white[:, :, :3] = 255
            white[:, :, 3] = 255
            out = alpha_composite_bgra(white, out, 0, 0)
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        ok = imwrite_japanese(path, out)
        if ok:
            self.statusBar().showMessage(f"書き出し完了: {path}")
        else:
            QMessageBox.critical(self, "保存失敗", "画像の保存に失敗しました。")

    # ---------------- list / selection ----------------
    def refresh_parts_list(self):
        self.scan_current_parts_folder()
        self.parts_list.clear()
        for p in self.parts:
            pid = p.get("id")
            img = self.get_part_image(pid)
            if img is None:
                continue
            thumb = self.make_thumbnail_icon(img)
            item = QListWidgetItem(QIcon(thumb), "")
            item.setToolTip(f"_parts/{pid}")
            item.setData(Qt.UserRole, pid)
            item.setSizeHint(QSize(92, 92))
            self.parts_list.addItem(item)

    def make_thumbnail_icon(self, img):
        bgra = ensure_bgra(img)
        h, w = bgra.shape[:2]
        canvas = np.zeros((80, 80, 4), dtype=np.uint8)
        canvas[:, :, :3] = 48
        canvas[:, :, 3] = 255
        scale = min(76 / max(w, 1), 76 / max(h, 1))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        thumb = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = (80 - nw) // 2
        y = (80 - nh) // 2
        alpha_composite_bgra(canvas, thumb, x, y)
        return QPixmap.fromImage(bgra_to_qimage(canvas))

    def refresh_placed_list(self):
        self.loading_ui = True
        self.placed_list.clear()
        for i, it in enumerate(self.items):
            part = self.part_by_id(it.get("part_id"))
            name = part.get("original_name", it.get("part_id")) if part else it.get("part_id")
            row_item = QListWidgetItem(f"{i + 1:02d}: {name}  ({float(it.get('x', 0.0)):.0f}, {float(it.get('y', 0.0)):.0f})")
            row_item.setFlags(row_item.flags() | Qt.ItemIsUserCheckable)
            row_item.setCheckState(Qt.Checked if it.get("visible", True) else Qt.Unchecked)
            self.placed_list.addItem(row_item)
        if self.selected_index is not None and 0 <= self.selected_index < self.placed_list.count():
            self.placed_list.setCurrentRow(self.selected_index)
        self.loading_ui = False
        self.update_controls_from_item()

    def update_placed_list_text(self):
        if self.selected_index is None or not (0 <= self.selected_index < self.placed_list.count()):
            return
        it = self.items[self.selected_index]
        part = self.part_by_id(it.get("part_id"))
        name = part.get("original_name", it.get("part_id")) if part else it.get("part_id")
        self.placed_list.item(self.selected_index).setText(f"{self.selected_index + 1:02d}: {name}  ({it['x']:.0f}, {it['y']:.0f})")

    def on_placed_item_changed(self, list_item):
        if self.loading_ui:
            return
        row = self.placed_list.row(list_item)
        if 0 <= row < len(self.items):
            self.items[row]["visible"] = list_item.checkState() == Qt.Checked
            self.refresh_preview(save=True)

    def on_placed_row_changed(self, row):
        if self.loading_ui:
            return
        if row < 0:
            self.select_item(None)
        else:
            self.select_item(row)

    def part_by_id(self, part_id):
        if not part_id:
            return None
        for p in self.parts:
            if p.get("id") == part_id:
                return p
        # 配置済み素材が別フォルダの素材を参照している場合でも名前を表示できるようにする。
        rel = Path(str(part_id))
        path = self.parts_root_dir / rel
        if path.exists() and path.is_file():
            folder = rel.parent.as_posix() if rel.parent.as_posix() != "." else DEFAULT_PARTS_FOLDER
            return {
                "id": str(part_id),
                "file": str(Path(PARTS_DIR) / rel),
                "folder": folder,
                "original_name": path.name,
            }
        return None

    def get_part_image(self, part_id):
        if not part_id:
            return None
        part_id = str(part_id).replace("\\", "/")
        if part_id in self.part_cache:
            return self.part_cache[part_id]
        rel = Path(part_id)
        # id は _parts からの相対パス。例: default/001.png
        path = self.parts_root_dir / rel
        if not path.exists():
            return None
        img = imread_japanese(str(path))
        if img is None:
            return None
        bgra = ensure_bgra(img)
        self.part_cache[part_id] = bgra
        return bgra

    def place_selected_part_center(self, item):
        if self.base_image is None:
            return
        part_id = item.data(Qt.UserRole)
        h, w = self.base_image.shape[:2]
        self.place_part(part_id, w / 2.0, h / 2.0)

    def place_part(self, part_id, x, y):
        if self.base_image is None:
            QMessageBox.warning(self, "配置不可", "先に元画像を読み込んでください。")
            return
        if self.get_part_image(part_id) is None:
            return
        self.items.append(self.default_item(part_id, x, y))
        self.selected_index = len(self.items) - 1
        self.refresh_placed_list()
        self.refresh_preview(save=True)
        self.canvas.setFocus()

    def selected_item(self):
        if self.selected_index is None:
            return None
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def select_item(self, idx):
        if idx is None or not (0 <= idx < len(self.items)):
            self.selected_index = None
            self.loading_ui = True
            self.placed_list.clearSelection()
            self.loading_ui = False
        else:
            self.selected_index = idx
            self.loading_ui = True
            self.placed_list.setCurrentRow(idx)
            self.loading_ui = False
        self.update_controls_from_item()
        self.canvas.update()
        self.placed_list.viewport().update()
        self.save_settings()

    def delete_selected_item(self):
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.items):
            del self.items[self.selected_index]
            if not self.items:
                self.selected_index = None
            else:
                self.selected_index = min(self.selected_index, len(self.items) - 1)
            self.refresh_placed_list()
            self.refresh_preview(save=True)

    def duplicate_selected_item(self):
        item = self.selected_item()
        if item is None:
            return
        new_item = dict(item)
        new_item["uid"] = str(uuid.uuid4())
        new_item["x"] = float(item.get("x", 0.0)) + 20.0
        new_item["y"] = float(item.get("y", 0.0)) + 20.0
        self.items.append(new_item)
        self.selected_index = len(self.items) - 1
        self.refresh_placed_list()
        self.refresh_preview(save=True)

    def move_selected_front(self):
        idx = self.selected_index
        if idx is None or idx >= len(self.items) - 1:
            return
        item = self.items.pop(idx)
        self.items.append(item)
        self.selected_index = len(self.items) - 1
        self.refresh_placed_list()
        self.refresh_preview(save=True)

    def move_selected_back(self):
        idx = self.selected_index
        if idx is None or idx <= 0:
            return
        item = self.items.pop(idx)
        self.items.insert(0, item)
        self.selected_index = 0
        self.refresh_placed_list()
        self.refresh_preview(save=True)

    def open_warp_dialog(self):
        idx = self.selected_index
        if idx is None or not (0 <= idx < len(self.items)):
            return
        if self.base_image is None:
            QMessageBox.warning(self, "ワープ不可", "先に元画像を読み込んでください。")
            return
        if self.get_part_image(self.items[idx].get("part_id")) is None:
            QMessageBox.warning(self, "ワープ不可", "素材画像を読み込めません。")
            return
        dialog = WarpDialog(self, idx)
        dialog.exec_()

    def reset_selected_warp(self):
        item = self.selected_item()
        if item is None:
            return
        item["warp_grid_rows"] = WARP_GRID_ROWS
        item["warp_grid_cols"] = WARP_GRID_COLS
        item["warp_points"] = default_warp_points()
        self.refresh_preview(save=True)

    def hit_test_item(self, x, y):
        # 前面から判定。透明部分でも、表示枠の内側なら掴めるようにする。
        for idx in reversed(range(len(self.items))):
            item = self.items[idx]
            if not item.get("visible", True):
                continue
            if bool(item.get("warp_enabled", False)):
                part_img = self.render_item_image_fast(item)
                if part_img is None:
                    continue
                h, w = part_img.shape[:2]
                cx = float(item.get("x", 0.0))
                cy = float(item.get("y", 0.0))
                if cx - w / 2.0 <= float(x) <= cx + w / 2.0 and cy - h / 2.0 <= float(y) <= cy + h / 2.0:
                    return idx
                continue

            part = self.get_part_image(item.get("part_id"))
            if part is None:
                continue
            h, w = part.shape[:2]
            scale_w = max(0.02, item_scale_w(item))
            scale_h = max(0.02, item_scale_h(item))
            angle = math.radians(float(item.get("rotation", 0.0) if item.get("rotation_enabled", False) else 0.0))
            dx = float(x) - float(item.get("x", 0.0))
            dy = float(y) - float(item.get("y", 0.0))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            sx = dx * cos_a - dy * sin_a
            sy = dx * sin_a + dy * cos_a
            lx = sx / scale_w
            ly = sy / scale_h
            if -w / 2.0 <= lx <= w / 2.0 and -h / 2.0 <= ly <= h / 2.0:
                return idx
        return None

    # ---------------- control sync ----------------
    def update_controls_from_item(self):
        item = self.selected_item()
        self.loading_ui = True
        enabled = item is not None
        controls = [
            self.scale_w_spin,
            self.scale_h_spin,
            self.scale_lock_check,
            self.feather_check,
            self.feather_spin,
            self.opacity_check,
            self.opacity_slider,
            self.opacity_spin,
            self.rotation_check,
            self.rotation_spin,
            self.flip_h_check,
            self.flip_v_check,
            self.warp_check,
            self.warp_edit_btn,
            self.warp_reset_btn,
            self.color_check,
            self.color_target_combo,
            self.color_algo_combo,
            self.color_strength_spin,
            self.local_margin_spin,
            self.shadow_check,
            self.shadow_opacity_spin,
            self.shadow_blur_spin,
            self.shadow_x_spin,
            self.shadow_y_spin,
        ]
        for c in controls:
            c.setEnabled(enabled)
        if item:
            scale_w = item_scale_w(item)
            scale_h = item_scale_h(item)
            scale_lock = bool(item.get("scale_lock", True))
            self.scale_w_spin.setValue(scale_w)
            self.scale_h_spin.setValue(scale_h)
            self.scale_lock_check.setChecked(scale_lock)
            self.scale_h_spin.setEnabled(not scale_lock)
            self.feather_check.setChecked(bool(item.get("feather_enabled", False)))
            self.feather_spin.setValue(int(item.get("feather_px", 8)))
            self.opacity_check.setChecked(bool(item.get("opacity_enabled", False)))
            op = int(item.get("opacity_percent", 100))
            self.opacity_slider.setValue(op)
            self.opacity_spin.setValue(op)
            self.rotation_check.setChecked(bool(item.get("rotation_enabled", False)))
            self.rotation_spin.setValue(float(item.get("rotation", 0.0)))
            self.flip_h_check.setChecked(bool(item.get("flip_h", False)))
            self.flip_v_check.setChecked(bool(item.get("flip_v", False)))
            self.warp_check.setChecked(bool(item.get("warp_enabled", False)))
            if not isinstance(item.get("warp_points"), list) or len(item.get("warp_points", [])) != WARP_GRID_ROWS * WARP_GRID_COLS:
                item["warp_points"] = default_warp_points()
                item["warp_grid_rows"] = WARP_GRID_ROWS
                item["warp_grid_cols"] = WARP_GRID_COLS
            self.color_check.setChecked(bool(item.get("color_enabled", False)))
            self.color_target_combo.setCurrentIndex(1 if item.get("color_target", "local") == "local" else 0)
            self.color_algo_combo.setCurrentIndex(int(item.get("color_algo", 1)))
            self.color_strength_spin.setValue(int(item.get("color_strength", 60)))
            self.local_margin_spin.setValue(int(item.get("local_margin", 0)))
            self.shadow_check.setChecked(bool(item.get("shadow_enabled", False)))
            self.shadow_opacity_spin.setValue(int(item.get("shadow_opacity", 40)))
            self.shadow_blur_spin.setValue(int(item.get("shadow_blur", 12)))
            self.shadow_x_spin.setValue(int(item.get("shadow_x", 6)))
            self.shadow_y_spin.setValue(int(item.get("shadow_y", 8)))
        else:
            self.scale_h_spin.setEnabled(False)
        self.loading_ui = False

    def controls_to_item(self, *args):
        if self.loading_ui:
            return
        item = self.selected_item()
        if item is None:
            return
        item["scale"] = float(self.scale_w_spin.value())
        item["scale_w"] = float(self.scale_w_spin.value())
        item["scale_h"] = float(self.scale_h_spin.value())
        item["scale_lock"] = bool(self.scale_lock_check.isChecked())
        item["feather_enabled"] = bool(self.feather_check.isChecked())
        item["feather_px"] = int(self.feather_spin.value())
        item["opacity_enabled"] = bool(self.opacity_check.isChecked())
        item["opacity_percent"] = int(self.opacity_spin.value())
        item["rotation_enabled"] = bool(self.rotation_check.isChecked())
        item["rotation"] = float(self.rotation_spin.value())
        item["flip_h"] = bool(self.flip_h_check.isChecked())
        item["flip_v"] = bool(self.flip_v_check.isChecked())
        item["warp_enabled"] = bool(self.warp_check.isChecked())
        item["warp_grid_rows"] = WARP_GRID_ROWS
        item["warp_grid_cols"] = WARP_GRID_COLS
        if not isinstance(item.get("warp_points"), list) or len(item.get("warp_points", [])) != WARP_GRID_ROWS * WARP_GRID_COLS:
            item["warp_points"] = default_warp_points()
        item["color_enabled"] = bool(self.color_check.isChecked())
        item["color_target"] = "local" if self.color_target_combo.currentIndex() == 1 else "global"
        item["color_algo"] = int(self.color_algo_combo.currentIndex())
        item["color_strength"] = int(self.color_strength_spin.value())
        item["local_margin"] = int(self.local_margin_spin.value())
        item["shadow_enabled"] = bool(self.shadow_check.isChecked())
        item["shadow_opacity"] = int(self.shadow_opacity_spin.value())
        item["shadow_blur"] = int(self.shadow_blur_spin.value())
        item["shadow_x"] = int(self.shadow_x_spin.value())
        item["shadow_y"] = int(self.shadow_y_spin.value())
        self.refresh_preview(save=True)

    def on_scale_w_changed(self, value):
        if self.loading_ui:
            return
        if self.scale_lock_check.isChecked() and abs(self.scale_h_spin.value() - value) > 1e-6:
            self.scale_h_spin.blockSignals(True)
            self.scale_h_spin.setValue(value)
            self.scale_h_spin.blockSignals(False)
        self.controls_to_item()

    def on_scale_h_changed(self, value):
        if self.loading_ui:
            return
        self.controls_to_item()

    def on_scale_lock_changed(self, checked):
        self.scale_h_spin.setEnabled(self.selected_item() is not None and not checked)
        if checked and abs(self.scale_h_spin.value() - self.scale_w_spin.value()) > 1e-6:
            self.scale_h_spin.blockSignals(True)
            self.scale_h_spin.setValue(self.scale_w_spin.value())
            self.scale_h_spin.blockSignals(False)
        self.controls_to_item()

    def on_opacity_slider(self, value):
        if self.opacity_spin.value() != value:
            self.opacity_spin.blockSignals(True)
            self.opacity_spin.setValue(value)
            self.opacity_spin.blockSignals(False)
        self.controls_to_item()

    def on_opacity_spin(self, value):
        if self.opacity_slider.value() != value:
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(value)
            self.opacity_slider.blockSignals(False)
        self.controls_to_item()

    # ---------------- render ----------------
    def begin_item_drag(self, idx):
        self.dragging_index = idx
        self.drag_preview_image = self.render_composite(full_quality=False, skip_index=idx)
        item = self.selected_item()
        self.drag_item_image = self.render_item_image_fast(item) if item is not None else None
        self.drag_item_pixmap = QPixmap.fromImage(bgra_to_qimage(self.drag_item_image)) if self.drag_item_image is not None else None
        self.canvas.invalidate_background_cache()
        self.canvas.update()
        if hasattr(self, "loupe_view"):
            self.loupe_view.update()

    def end_item_drag(self):
        self.dragging_index = None
        self.drag_preview_image = None
        self.drag_item_image = None
        self.drag_item_pixmap = None
        self.refresh_preview(save=True)


    def current_loupe_source_image(self):
        """ルーペ表示用の現在画像。素材ドラッグ中は軽量移動表示も反映する。"""
        if (
            self.dragging_index is not None
            and self.drag_preview_image is not None
            and self.base_image is not None
        ):
            canvas = self.drag_preview_image.copy()
            item = self.selected_item()
            part_img = self.drag_item_image
            if item is not None and part_img is not None:
                ih, iw = part_img.shape[:2]
                left = int(round(float(item.get("x", 0.0)) - iw / 2.0))
                top = int(round(float(item.get("y", 0.0)) - ih / 2.0))
                alpha_composite_bgra(canvas, part_img, left, top)
            return canvas
        return self.preview_image

    def render_item_image_fast(self, item):
        """移動中だけ使う軽量表示。色合わせ・輪郭なじませ・影は省く。"""
        img = self.render_item_pre_warp(item, self.base_image, apply_color=False)
        if img is None:
            return None
        if bool(item.get("warp_enabled", False)):
            img = apply_mesh_warp(img, get_item_warp_points(item))
        angle = float(item.get("rotation", 0.0)) if bool(item.get("rotation_enabled", False)) else 0.0
        img = rotate_image(img, angle)
        if bool(item.get("opacity_enabled", False)):
            img = apply_opacity(img, int(item.get("opacity_percent", 100)))
        return img

    def refresh_preview(self, save=True):
        if self.base_image is None:
            self.preview_image = None
        else:
            self.preview_image = self.render_composite(full_quality=False)
        self.canvas.invalidate_background_cache()
        self.canvas.update()
        if hasattr(self, "loupe_view"):
            self.loupe_view.update()
        if save:
            self.save_settings()

    def local_target_crop(self, base, item, part_shape):
        h, w = base.shape[:2]
        ph, pw = part_shape[:2]
        scale_w = item_scale_w(item)
        scale_h = item_scale_h(item)
        margin = int(item.get("local_margin", 0))
        # 回転後でもだいたい覆えるように対角線ベースで矩形を広めに取る。
        radius_x = (pw * scale_w) / 2.0
        radius_y = (ph * scale_h) / 2.0
        r = math.sqrt(radius_x * radius_x + radius_y * radius_y)
        cx = float(item.get("x", 0.0))
        cy = float(item.get("y", 0.0))
        x1 = max(0, int(cx - r - margin))
        y1 = max(0, int(cy - r - margin))
        x2 = min(w, int(cx + r + margin))
        y2 = min(h, int(cy + r + margin))
        if x1 >= x2 or y1 >= y2:
            return base
        return base[y1:y2, x1:x2]

    def render_item_pre_warp(self, item, base, apply_color=True):
        part = self.get_part_image(item.get("part_id"))
        if part is None:
            return None
        img = part.copy()

        if apply_color and bool(item.get("color_enabled", False)):
            if item.get("color_target", "local") == "local":
                target = self.local_target_crop(base, item, img.shape)
            else:
                target = base
            img = color_match_bgra(
                img,
                target,
                int(item.get("color_algo", 1)),
                int(item.get("color_strength", 60)),
            )

        img = apply_flip(img, bool(item.get("flip_h", False)), bool(item.get("flip_v", False)))
        img = resize_image(img, item_scale_w(item), item_scale_h(item))
        return img

    def render_item_image(self, item, base):
        img = self.render_item_pre_warp(item, base, apply_color=True)
        if img is None:
            return None

        if bool(item.get("warp_enabled", False)):
            img = apply_mesh_warp(img, get_item_warp_points(item))

        angle = float(item.get("rotation", 0.0)) if bool(item.get("rotation_enabled", False)) else 0.0
        img = rotate_image(img, angle)

        # 輪郭なじませは最終サイズ・最終角度になった画像に対して行う。
        # これで n px は「配置後の見た目のピクセル数」基準になる。
        if bool(item.get("feather_enabled", False)):
            img = apply_feather_alpha(img, int(item.get("feather_px", 8)))

        if bool(item.get("opacity_enabled", False)):
            img = apply_opacity(img, int(item.get("opacity_percent", 100)))
        return img

    def render_composite(self, full_quality=True, skip_index=None):
        if self.base_image is None:
            return None
        canvas = self.base_image.copy()
        for idx, item in enumerate(self.items):
            if skip_index is not None and idx == skip_index:
                continue
            if not item.get("visible", True):
                continue
            part_img = self.render_item_image(item, self.base_image)
            if part_img is None:
                continue
            ih, iw = part_img.shape[:2]
            left = int(round(float(item.get("x", 0.0)) - iw / 2.0))
            top = int(round(float(item.get("y", 0.0)) - ih / 2.0))

            if bool(item.get("shadow_enabled", False)):
                shadow = make_shadow_image(
                    part_img,
                    int(item.get("shadow_opacity", 40)),
                    int(item.get("shadow_blur", 12)),
                )
                if shadow is not None:
                    sx = left + int(item.get("shadow_x", 6))
                    sy = top + int(item.get("shadow_y", 8))
                    alpha_composite_bgra(canvas, shadow, sx, sy)

            alpha_composite_bgra(canvas, part_img, left, top)
        return canvas


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = ImageCompositor()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
