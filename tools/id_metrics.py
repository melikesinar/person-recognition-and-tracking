"""
Identity metrics for multi-camera tracking, computed against ground truth.

Input: per (camera, frame), the ground-truth boxes with person ids, and the
system's boxes with the P number it showed (or None = "unknown / not yet").

Step 1, detection matching: in each (camera, frame), ground-truth and system
boxes are paired one-to-one (Hungarian) when they overlap enough (IoU >= 0.5).

Step 2, identity matching (the IDF1 idea, Ristani et al. 2016): each true person
is paired with at most one P number, choosing the pairing that maximises how
often they coincide. A box counts as "identity-correct" (IDTP) only if its P
number is the one paired with its true person.

    IDR  (identity recall)    = IDTP / ground-truth boxes
         "of all moments a person was visible, how often did they carry the
          right number"
    IDP  (identity precision) = IDTP / numbered system boxes that matched a person
    IDF1                      = 2 IDTP / (ground-truth boxes + numbered matched boxes)

Global = one pairing across all cameras (the same person must keep the same
number from camera to camera). Per-camera = each camera paired separately
(within-camera tracking quality only).

Note: system boxes that match no ground-truth box are left out of IDP/IDF1
(see `unmatched` in the result), because datasets like WILDTRACK only label
people inside an area of interest; people outside it are real, not errors.

Also reported:
    fragmentation : how many different P numbers a true person received on
                    average (1.0 = never split)
    mixing        : how many different true people one P number was given to on
                    average (1.0 = never mixed up)
Both only count a P number / person once it covers >= `min_share` of the
other's matched boxes, so single-frame glitches do not dominate.
"""
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Tuple[float, float, float, float]
Key = Tuple[Hashable, Hashable]  # (camera, frame)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_detections(gt: Dict[Key, List[Tuple[Hashable, Box]]],
                     pred: Dict[Key, List[Tuple[Optional[Hashable], Box]]],
                     iou_min: float = 0.5):
    """Returns [(key, gt_id, pred_id or None)] for matched pairs, plus counts."""
    pairs, n_gt, n_pred = [], 0, 0
    for key in set(gt) | set(pred):
        g = gt.get(key, [])
        p = pred.get(key, [])
        n_gt += len(g)
        n_pred += len(p)
        if not g or not p:
            continue
        ious = iou_matrix(np.array([b for _, b in g], float), np.array([b for _, b in p], float))
        rows, cols = linear_sum_assignment(-ious)
        for r, c in zip(rows, cols):
            if ious[r, c] >= iou_min:
                pairs.append((key, g[r][0], p[c][0]))
    return pairs, n_gt, n_pred


def _identity_scores(pairs, n_gt: int, group=lambda key: None) -> dict:
    """IDTP etc. for matched pairs; `group` splits ids into independent namespaces."""
    co = Counter()
    n_numbered = 0
    for key, gid, pid in pairs:
        if pid is None:
            continue
        n_numbered += 1
        g = group(key)
        co[((g, gid), (g, pid))] += 1

    gids = sorted({k[0] for k in co}, key=str)
    pids = sorted({k[1] for k in co}, key=str)
    idtp = 0
    if gids and pids:
        gi = {g: i for i, g in enumerate(gids)}
        pi = {p: i for i, p in enumerate(pids)}
        m = np.zeros((len(gids), len(pids)))
        for (g, p), n in co.items():
            m[gi[g], pi[p]] = n
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    return {
        "IDTP": idtp,
        "IDR": idtp / max(1, n_gt),
        "IDP": idtp / max(1, n_numbered),
        "IDF1": 2 * idtp / max(1, n_gt + n_numbered),
        "numbered_matched": n_numbered,
        "cooccurrence": co,
    }


def _spread(co: Counter, side: int, min_share: float) -> float:
    """Average number of partners per id on `side` (0 = per true person, 1 = per P number)."""
    per = defaultdict(Counter)
    for pair, n in co.items():
        per[pair[side]][pair[1 - side]] += n
    if not per:
        return 0.0
    counts = []
    for partners in per.values():
        total = sum(partners.values())
        counts.append(sum(1 for n in partners.values() if n >= min_share * total))
    return float(np.mean(counts))


def evaluate(gt, pred, iou_min: float = 0.5, min_share: float = 0.1) -> dict:
    pairs, n_gt, n_pred = match_detections(gt, pred, iou_min)
    numbered = sum(1 for _, _, p in pairs if p is not None)
    glob = _identity_scores(pairs, n_gt)

    per_cam = {}
    cams = sorted({k[0] for k in gt}, key=str)
    for cam in cams:
        cam_pairs = [x for x in pairs if x[0][0] == cam]
        cam_gt = sum(len(v) for k, v in gt.items() if k[0] == cam)
        s = _identity_scores(cam_pairs, cam_gt)
        per_cam[cam] = {k: s[k] for k in ("IDF1", "IDR", "IDP")}
    within = _identity_scores(pairs, n_gt, group=lambda key: key[0])

    return {
        "gt_boxes": n_gt,
        "system_boxes": n_pred,
        "detected": len(pairs) / max(1, n_gt),                 # detection recall
        "unmatched": n_pred - len(pairs),                       # system boxes on no labelled person
        "numbered": numbered / max(1, len(pairs)),             # matched boxes that carried a P number
        "global": {k: glob[k] for k in ("IDF1", "IDR", "IDP")},
        "within_camera": {k: within[k] for k in ("IDF1", "IDR", "IDP")},
        "per_camera": per_cam,
        "fragmentation": _spread(glob["cooccurrence"], 0, min_share),
        "mixing": _spread(glob["cooccurrence"], 1, min_share),
        "true_people": len({g for _, g, _ in pairs}),
        "p_numbers": len({p for _, _, p in pairs if p is not None}),
    }


def format_report(r: dict) -> List[str]:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        f"ground-truth boxes: {r['gt_boxes']}   system boxes: {r['system_boxes']}   "
        f"(on no labelled person: {r['unmatched']})",
        f"detected people (IoU>=0.5): {pct(r['detected'])}   of those carrying a P number: {pct(r['numbered'])}",
        f"GLOBAL (across cameras)  IDF1 {pct(r['global']['IDF1'])}   IDR {pct(r['global']['IDR'])}   "
        f"IDP {pct(r['global']['IDP'])}",
        f"within each camera       IDF1 {pct(r['within_camera']['IDF1'])}   IDR {pct(r['within_camera']['IDR'])}   "
        f"IDP {pct(r['within_camera']['IDP'])}",
        f"true people: {r['true_people']}   P numbers used: {r['p_numbers']}   "
        f"fragmentation {r['fragmentation']:.2f} numbers/person   mixing {r['mixing']:.2f} people/number",
    ]
    for cam, s in r["per_camera"].items():
        lines.append(f"  camera {cam}: IDF1 {pct(s['IDF1'])}  IDR {pct(s['IDR'])}  IDP {pct(s['IDP'])}")
    return lines
