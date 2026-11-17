"""
Measures how well the ReID pipeline recognises the same person across different cameras.

Dataset: Market-1501 (1,501 people filmed by 6 cameras). Only the folder layout is
needed:
    <root>/query/              3,368 query images
    <root>/bounding_box_test/  19,732 gallery images
File names encode identity and camera: 0002_c1s1_000451_03.jpg -> person 2, camera 1.

Protocol (standard): for every query image, rank all gallery images by similarity,
ignoring gallery images of the same person *from the same camera* (that would be
trivial). So every correct match is necessarily a cross-camera match, which is
exactly the situation we care about.

Metrics:
    Rank-1 : % of queries whose single most similar gallery image is the same person.
    Rank-5 : % of queries with the same person somewhere in the top 5.
    mAP    : mean average precision; rewards ranking *all* images of the person high.

Variants compare preprocessing options of the real ReidExtractor:
    current       RGB input + CLAHE  (the pipeline as it is now)
    legacy_bgr    BGR input + CLAHE  (the pipeline before the colour-order fix)
    no_clahe      RGB input, no CLAHE

Usage (from the project root):
    python tools/eval_reid.py --data D:/datasets/Market-1501-v15.09.15
    python tools/eval_reid.py --data ... --num-ids 200 --variants current,legacy_bgr
"""
import sys
import time
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.config import Config  # noqa: E402
from core.reid_extractor import ReidExtractor  # noqa: E402


# --------------------------------------------------------------------------- data

def parse_name(path: Path):
    """'0002_c1s1_000451_03.jpg' -> (2, 1). Returns None for junk images (pid -1)."""
    parts = path.stem.split("_")
    pid = int(parts[0])
    if pid == -1:
        return None
    cam = int(parts[1][1])
    return pid, cam


def load_split(folder: Path):
    items = []
    for p in sorted(folder.glob("*.jpg")):
        parsed = parse_name(p)
        if parsed is not None:
            items.append((p, *parsed))
    return items


def subsample(query, gallery, num_ids: int, distractors: int, seed: int = 0):
    """Keeps `num_ids` random identities plus `distractors` other gallery images, for quick CPU runs."""
    rng = random.Random(seed)
    query_ids = sorted({pid for _, pid, _ in query})
    keep = set(rng.sample(query_ids, min(num_ids, len(query_ids))))
    q = [it for it in query if it[1] in keep]
    g_keep = [it for it in gallery if it[1] in keep]
    g_other = [it for it in gallery if it[1] not in keep]
    g = g_keep + rng.sample(g_other, min(distractors, len(g_other)))
    return q, g


# --------------------------------------------------------------------------- metric

def evaluate(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams, max_rank: int = 10):
    """Standard Market-1501 CMC + mAP (same person & same camera gallery entries are ignored)."""
    sims = q_feats @ g_feats.T
    cmc_sum = np.zeros(max_rank)
    aps = []
    valid_queries = 0
    for i in range(len(q_pids)):
        order = np.argsort(-sims[i])
        keep = ~((g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i]))
        matches = (g_pids[order][keep] == q_pids[i]).astype(np.int32)
        if not matches.any():
            continue  # this person has no image from another camera in the gallery
        valid_queries += 1
        first_hit = int(np.argmax(matches))
        if first_hit < max_rank:
            cmc_sum[first_hit:] += 1
        hits = np.cumsum(matches)
        precision_at_hit = hits[matches == 1] / (np.flatnonzero(matches) + 1)
        aps.append(precision_at_hit.mean())
    if valid_queries == 0:
        raise ValueError("No valid queries: every query lacks a cross-camera gallery match.")
    return cmc_sum / valid_queries, float(np.mean(aps)), valid_queries


# --------------------------------------------------------------------------- feature extraction

class LegacyBGRExtractor(ReidExtractor):
    """Reproduces the old preprocessing: CLAHE, resize, but no BGR->RGB conversion."""

    def preprocess(self, crop_bgr):
        img = self._apply_clahe(crop_bgr) if self.use_clahe else crop_bgr
        return cv2.resize(img, Config.REID_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)


VARIANTS = {
    "current": lambda dev: ReidExtractor(dev, use_clahe=True),
    "legacy_bgr": lambda dev: LegacyBGRExtractor(dev, use_clahe=True),
    "no_clahe": lambda dev: ReidExtractor(dev, use_clahe=False),
}


def extract(extractor, items, batch_size: int):
    feats = []
    for i in range(0, len(items), batch_size):
        crops = [cv2.imread(str(p)) for p, _, _ in items[i:i + batch_size]]
        feats.append(extractor.extract_embeddings_from_crops(crops))
        done = min(i + batch_size, len(items))
        print(f"\r    {done}/{len(items)} images", end="", flush=True)
    print()
    return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Market-1501 root folder (contains query/ and bounding_box_test/)")
    ap.add_argument("--variants", default="current,legacy_bgr,no_clahe",
                    help=f"Comma-separated, from: {', '.join(VARIANTS)}")
    ap.add_argument("--num-ids", type=int, default=0,
                    help="Evaluate on N random identities only (0 = full dataset). Useful on CPU.")
    ap.add_argument("--distractors", type=int, default=2000,
                    help="With --num-ids: number of other gallery images mixed in as distractors.")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.data)
    query = load_split(root / "query")
    gallery = load_split(root / "bounding_box_test")
    if not query or not gallery:
        sys.exit(f"Could not find images under {root}/query and {root}/bounding_box_test")
    if args.num_ids:
        query, gallery = subsample(query, gallery, args.num_ids, args.distractors)
    print(f"Queries: {len(query)}, gallery: {len(gallery)}, "
          f"identities: {len({pid for _, pid, _ in query})}")

    q_pids = np.array([pid for _, pid, _ in query]); q_cams = np.array([c for _, _, c in query])
    g_pids = np.array([pid for _, pid, _ in gallery]); g_cams = np.array([c for _, _, c in gallery])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            sys.exit(f"Unknown variant '{name}'. Choose from: {', '.join(VARIANTS)}")
        print(f"\n[{name}]")
        extractor = VARIANTS[name](device)
        t0 = time.time()
        q_feats = extract(extractor, query, args.batch_size)
        g_feats = extract(extractor, gallery, args.batch_size)
        elapsed = time.time() - t0
        cmc, mAP, n_valid = evaluate(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams)
        ms_per_img = elapsed / (len(query) + len(gallery)) * 1000
        results.append((name, cmc[0], cmc[4], mAP, ms_per_img))
        print(f"    Rank-1 {cmc[0]:.1%}  Rank-5 {cmc[4]:.1%}  mAP {mAP:.1%}  ({n_valid} queries, {ms_per_img:.0f} ms/img)")

    print("\n| Variant | Rank-1 | Rank-5 | mAP | ms/image |")
    print("|---|---|---|---|---|")
    for name, r1, r5, mAP, ms in results:
        print(f"| {name} | {r1:.1%} | {r5:.1%} | {mAP:.1%} | {ms:.0f} |")


if __name__ == "__main__":
    main()
