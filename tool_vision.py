import cv2, os, numpy as np
from skimage.metrics import structural_similarity as ssim
from cfg import (
    DEFAULT_WEIGHTS, KIT_WEIGHTS,
    UPLOAD_FOLDER_REQ, UPLOAD_FOLDER_ITEMS
)

DEBUG_MATCH = True

# ========================================KIT========================================

def _segment_components_from_bgr_kit(bgr: np.ndarray):
    """
    Спец-сегментация для наборов: разрешаем огромный объект, касание границ,
    оставляем только самый большой контур.
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 25, 25)

    _, m_otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m_otsu_dir = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m_adapt = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    41, 5)
    v = np.median(gray)
    lo = int(max(0, (1.0 - 0.33) * v))
    hi = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lo, hi)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    cnts_e, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_edges = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(m_edges, cnts_e, -1, 255, thickness=cv2.FILLED)

    mask = cv2.bitwise_or(m_otsu_inv, m_adapt)
    mask = cv2.bitwise_or(mask, m_edges)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(m_otsu_dir))

    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kx, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ky, iterations=1)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
    mask = cv2.dilate(mask, k3, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    comps = []
    area_min = 0.10 * (H * W)
    area_max = 0.98 * (H * W)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue

        x, y, w, h = cv2.boundingRect(c)

        extent = area / float(w * h + 1e-6)
        if extent < 0.05:
            continue

        m_comp = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m_comp, [c], -1, 255, thickness=cv2.FILLED)
        comps.append({'contour': c, 'bbox': (x, y, w, h), 'mask': m_comp})

    if comps:
        comps.sort(key=lambda z: z['bbox'][2]*z['bbox'][3], reverse=True)
        comps = [comps[0]]

    return comps

def _build_expected_ref_map_kit(req: 'Request'):
    expected = {}
    for link in req.items:
        item = link.item
        views = []

        def _add_views_from_bgr(bgr):
            if bgr is None:
                return
            segs = _segment_components_from_bgr_kit(bgr)
            if not segs:
                H, W = bgr.shape[:2]
                m = np.zeros((H, W), dtype=np.uint8)
                margin = 4
                m[margin:H-margin, margin:W-margin] = 255
                desc = _compute_desc(bgr, m)
                if desc:
                    views.append({'bgr': bgr, 'mask': m, 'desc': desc})
                return
            seg = segs[0]
            desc = _compute_desc(bgr, seg['mask'])
            if desc:
                views.append({'bgr': bgr, 'mask': seg['mask'], 'desc': desc})

        if item.photo_filename:
            bgr = _load_bgr_from_uploads('items', item.photo_filename)
            _add_views_from_bgr(bgr)

        for ref in item.ref_photos:
            bgr = _load_bgr_from_uploads('items', ref.filename)
            _add_views_from_bgr(bgr)

        if views:
            expected[item.inv] = views

    return expected


# ========================================SINGLE========================================

def _segment_components_from_bgr(bgr: np.ndarray):
    """
    Сегментация инструмента на однотонном фоне (устойчива к бликам) + фильтрация мусора.
    Возвращает список компонент: {'contour','bbox','mask'} в координатах кадра.
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 25, 25)

    # --- маски ---
    # Otsu (тёмный объект)
    _, m_otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Otsu (светлый объект) — полезно на блеске
    _, m_otsu_dir = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Адаптивный порог
    m_adapt = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    41, 5)
    v = np.median(gray)
    lo = int(max(0, (1.0 - 0.33) * v))
    hi = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lo, hi)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    cnts_e, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_edges = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(m_edges, cnts_e, -1, 255, thickness=cv2.FILLED)

    # Объединяем всё
    mask = cv2.bitwise_or(m_otsu_inv, m_adapt)
    mask = cv2.bitwise_or(mask, m_edges)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(m_otsu_dir))

    # Анизотропное закрытие (вдоль/поперёк оси) + мягкая очистка
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kx, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ky, iterations=1)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
    mask = cv2.dilate(mask, k3, iterations=1)

    comps = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_min = 0.0015 * (H * W)
    area_max = 0.45   * (H * W)
    border_eps = 2

    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if x <= border_eps or y <= border_eps or (x + w) >= (W - border_eps) or (y + h) >= (H - border_eps):
            continue

        extent = area / float(w * h + 1e-6)
        long_side = max(w, h)
        long_frac = long_side / max(W, H)
        aspect = long_side / max(1, min(w, h))

        # базовый порог заполненности
        ok_extent = extent >= 0.08
        # послабление для длинных/узких объектов
        if not ok_extent and aspect > 4.0 and long_frac >= 0.10 and extent >= 0.02:
            ok_extent = True
        if not ok_extent:
            continue

        per = cv2.arcLength(c, True)
        circularity = 4.0 * np.pi * area / (per * per + 1e-6)
        if circularity < 0.008:
            continue

        m_comp = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m_comp, [c], -1, 255, thickness=cv2.FILLED)
        comps.append({'contour': c, 'bbox': (x, y, w, h), 'mask': m_comp})

    return comps


def _build_expected_ref_map(req: 'Request'):
    """
    Для каждого ожидаемого экземпляра собираем набор эталонных представлений
    (основное фото + все ref-фото). Для каждого ракурса сохраняем bgr, mask и desc.
    """
    expected = {}
    for link in req.items:
        item = link.item
        views = []

        def _add_views_from_bgr(bgr):
            if bgr is None:
                return
            segs = _segment_components_from_bgr(bgr)
            if not segs:
                return
            seg = max(segs, key=lambda s: cv2.countNonZero(s['mask']))
            desc = _compute_desc(bgr, seg['mask'])
            if not desc:
                return
            views.append({'bgr': bgr, 'mask': seg['mask'], 'desc': desc})

        if item.photo_filename:
            bgr = _load_bgr_from_uploads('items', item.photo_filename)
            _add_views_from_bgr(bgr)

        for ref in item.ref_photos:
            bgr = _load_bgr_from_uploads('items', ref.filename)
            _add_views_from_bgr(bgr)

        if views:
            expected[item.inv] = views

    return expected


# ========================================COMMON========================================


def _load_bgr_from_uploads(subfolder: str, filename: str):
    if subfolder == 'requests':
        folder = UPLOAD_FOLDER_REQ
    elif subfolder == 'items':
        folder = UPLOAD_FOLDER_ITEMS
    else:
        raise ValueError('bad subfolder')
    path = os.path.join(folder, filename)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)
    return img

def _compute_desc(bgr: np.ndarray, mask: np.ndarray):
    """Контур/формы + цветовая гистограмма + ORB-фичи."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / max(h, 1)

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten()

    orb = cv2.ORB_create(nfeatures=800)
    kp, des = orb.detectAndCompute(bgr, mask)
    return {'cnt': cnt, 'area': area, 'aspect': aspect, 'hu': hu, 'hist': hist, 'kp': kp, 'des': des}

def _score_pair_parts(comp, ref, comp_bgr, ref_bgr, comp_mask=None, ref_mask=None, *, weights=None):
    w = (weights or DEFAULT_WEIGHTS)  # <-- новые веса

    # --- SHAPE ---
    shape_dist  = cv2.matchShapes(comp['cnt'], ref['cnt'], cv2.CONTOURS_MATCH_I3, 0.0)
    area_ratio  = min(comp['area'], ref['area']) / max(comp['area'], ref['area'])
    aspect_diff = abs(comp['aspect'] - ref['aspect'])
    shape_score = 1.0 / (1.0 + shape_dist + 0.4 * abs(1 - area_ratio) + 0.2 * aspect_diff)
    shape_score = float(np.clip(shape_score, 0, 1))

    # --- ORB ---
    feat_score = 0.0
    if comp['des'] is not None and ref['des'] is not None and len(comp['des']) and len(ref['des']):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(comp['des'], ref['des'], k=2)
        good = 0
        for mn in matches:
            if len(mn) == 2:
                m, n = mn
                if m.distance < 0.75 * n.distance:
                    good += 1
        feat_score = min(good / 60.0, 1.0)

    # --- COLOR ---
    corr = cv2.compareHist(comp['hist'].astype('float32'),
                           ref['hist'].astype('float32'),
                           cv2.HISTCMP_CORREL)
    color_score = float((corr + 1) / 2)

    # --- SSIM ---
    ssim_score = 0.0
    if comp_mask is not None and ref_mask is not None:
        ssim_score = _masked_ssim_from_pair(comp, ref, comp_bgr, ref_bgr, comp_mask, ref_mask)

    total = (
        w['shape'] * shape_score +
        w['feat']  * feat_score  +
        w['color'] * color_score +
        w['ssim']  * ssim_score
    )
    return float(np.clip(total, 0, 1)), shape_score, feat_score, color_score, ssim_score


def _score_pair(comp, ref, comp_bgr, ref_bgr, comp_mask=None, ref_mask=None, *, weights=None) -> float:
    total, *_ = _score_pair_parts(comp, ref, comp_bgr, ref_bgr, comp_mask, ref_mask, weights=weights)
    return total

def _collect_request_components(req: 'Request'):
    components = []
    for ph in req.photos:
        bgr = _load_bgr_from_uploads('requests', ph.filename)
        if bgr is None:
            continue
        segs = _segment_components_from_bgr(bgr)
        for seg in segs:
            desc = _compute_desc(bgr, seg['mask'])
            if desc:
                components.append({'bgr': bgr, 'mask': seg['mask'], 'desc': desc, 'photo_id': ph.id})
    return components

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _match_components_to_expected( components: list, expected_map: dict, threshold, weights=None, allow_margin=True):
    n_comp = len(components)
    exp_invs = list(expected_map.keys())
    n_exp = len(exp_invs)

    if n_comp == 0 or n_exp == 0:
        return [], exp_invs[:], list(range(n_comp))

    S = np.zeros((n_comp, n_exp), dtype=np.float32)
    for i, comp in enumerate(components):
        c_desc = comp.get('desc')
        c_bgr  = comp.get('bgr')
        c_mask = comp.get('mask')
        for j, inv in enumerate(exp_invs):
            best = 0.0
            for ref in expected_map[inv]:
                r_desc = ref.get('desc')
                r_bgr  = ref.get('bgr')
                r_mask = ref.get('mask')

                try:
                    s = _score_pair(c_desc, r_desc, c_bgr, r_bgr, c_mask, r_mask, weights=weights)
                except Exception:
                    s = 0.0
                if s > best:
                    best = s
            S[i, j] = best

    if 'DEBUG_MATCH' in globals() and DEBUG_MATCH:
        print("\n=== MATCH DEBUG ===")
        print(f"components: {n_comp}; expected: {n_exp} -> {exp_invs}")
        for i in range(n_comp):
            row = S[i]
            best_j = int(np.argmax(row))
            best_inv = exp_invs[best_j]
            parts = None
            best_ref_score = -1.0
            for ref in expected_map[best_inv]:
                sc, sh, ft, col, ss = _score_pair_parts(
                    components[i]['desc'], ref['desc'],
                    components[i]['bgr'],  ref['bgr'],
                    components[i].get('mask'), ref.get('mask'),
                    weights=weights
                )
                if sc > best_ref_score:
                    best_ref_score = sc
                    parts = (sh, ft, col, ss, sc)
            parts_str = ""
            if parts:
                sh, ft, col, ss, sc = parts
                parts_str = f" | shape={sh:.2f} feat={ft:.2f} color={col:.2f} ssim={ss:.2f} -> total={sc:.2f}"
            print(f" comp[{i}] best → {best_inv}: score={row[best_j]:.3f} "
                  f"| all={[f'{v:.2f}' for v in row]}{parts_str}")
        print("===================\n")

    if _HAS_SCIPY:
        cost = 1.0 - S
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows, cols))
    else:
        pairs = []
        S_copy = S.copy()
        used_rows, used_cols = set(), set()
        while True:
            max_val = -1.0
            max_rc = None
            for i in range(n_comp):
                if i in used_rows:
                    continue
                for j in range(n_exp):
                    if j in used_cols:
                        continue
                    if S_copy[i, j] > max_val:
                        max_val = float(S_copy[i, j])
                        max_rc = (i, j)
            if max_rc is None or max_val < 0.0:
                break
            r, c = max_rc
            pairs.append((r, c))
            used_rows.add(r)
            used_cols.add(c)

    assignments = []
    used_invs = set()
    used_comps = set()

    THR_HI = threshold
    THR_LO = 0.45
    MARGIN = 0.12
    SHAPE_MIN = 0.25

    for r, c in pairs:
        s = float(S[r, c])
        inv = exp_invs[c]
        if inv in used_invs or r in used_comps:
            continue

        row = S[r]
        if row.shape[0] > 1:
            second_best = float(np.max(np.delete(row, c)))
        else:
            second_best = 0.0
        margin = s - second_best

        best_shape = 0.0
        try:
            best_ref_score = -1.0
            for ref in expected_map[inv]:
                tot, sh, ft, col, ss = _score_pair_parts(
                    components[r]['desc'], ref['desc'],
                    components[r]['bgr'], ref['bgr'],
                    components[r].get('mask'), ref.get('mask'),
                    weights=weights
                )
                if tot > best_ref_score:
                    best_ref_score = tot
                    best_shape = sh
        except Exception:
            best_shape = 0.0

        accept = False
        reason = ""
        if s >= THR_HI:
            accept = True
            reason = "abs"
        elif allow_margin and s >= THR_LO and margin >= MARGIN and best_shape >= SHAPE_MIN:
            accept = True;
            reason = "margin"

        if accept:
            assignments.append({'component_index': r, 'inv': inv, 'score': s})
            used_invs.add(inv)
            used_comps.add(r)

        if 'DEBUG_MATCH' in globals() and DEBUG_MATCH:
            print(f" accept? {accept} [{reason}] r={r} inv={inv} s={s:.3f} "
                  f"second={second_best:.3f} margin={margin:.3f} shape={best_shape:.2f}")

    matched_invs = {a['inv'] for a in assignments}
    missing_invs = [inv for inv in exp_invs if inv not in matched_invs]

    matched_comps = {a['component_index'] for a in assignments}
    extras_idx = [i for i in range(n_comp) if i not in matched_comps]

    return assignments, missing_invs, extras_idx



def _mask_from_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 25, 25)

    _, m1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m2 = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                               41, 5)
    v = np.median(gray)
    lo = int(max(0, (1.0 - 0.33) * v))
    hi = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray, lo, hi)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m3 = np.zeros_like(gray)
    cv2.drawContours(m3, cnts, -1, 255, thickness=cv2.FILLED)

    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
    mask = cv2.dilate(mask, k3, iterations=1)
    return mask

def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union

def _suppress_nested_components(comps, iou_thr=0.65, inside_thr=0.85, area_min_frac=0.015):
    if not comps:
        return comps

    areas = [float(c['bbox'][2] * c['bbox'][3]) for c in comps]
    largest = max(areas)
    order = sorted(range(len(comps)), key=lambda i: areas[i], reverse=True)
    keep = []

    for i in order:
        bi = comps[i]['bbox']; ai = areas[i]
        if ai < area_min_frac * largest:
            continue

        drop = False
        for k in keep:
            bk = comps[k]['bbox']; ak = areas[k]

            x1 = max(bi[0], bk[0]); y1 = max(bi[1], bk[1])
            x2 = min(bi[0] + bi[2], bk[0] + bk[2]); y2 = min(bi[1] + bi[3], bk[1] + bk[3])
            iw = max(0, x2 - x1); ih = max(0, y2 - y1)
            inter = iw * ih
            inside_ratio = inter / (bi[2] * bi[3] + 1e-6)

            if inside_ratio >= inside_thr and ai < 0.2 * ak:
                drop = True; break

            if _bbox_iou(bi, bk) > iou_thr:
                drop = True; break

        if not drop:
            keep.append(i)

    return [comps[i] for i in keep]


def _masked_ssim_from_pair(comp_desc, ref_desc, comp_bgr, ref_bgr, comp_mask, ref_mask):
    xc, yc, wc, hc = cv2.boundingRect(comp_desc['cnt'])
    xr, yr, wr, hr = cv2.boundingRect(ref_desc['cnt'])

    cg = cv2.cvtColor(comp_bgr[yc:yc+hc, xc:xc+wc], cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref_bgr [yr:yr+hr, xr:xr+wr], cv2.COLOR_BGR2GRAY)

    cm = (comp_mask[yc:yc+hc, xc:xc+wc] > 0)
    rm = (ref_mask [yr:yr+hr, xr:xr+wr] > 0)

    cg = np.where(cm, cg, 255).astype('uint8')
    rg = np.where(rm, rg, 255).astype('uint8')

    cg = cv2.resize(cg, (128, 128))
    rg = cv2.resize(rg, (128, 128))

    try:
        return float(ssim(cg, rg))
    except Exception:
        return 0.0
