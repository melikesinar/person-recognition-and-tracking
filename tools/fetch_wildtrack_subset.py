"""
Downloads only a PART of the WILDTRACK dataset (selected cameras, first N frames)
without downloading the whole 6.8 GB zip.

The EPFL server allows reading any byte range of the zip file, so this script
reads the zip's table of contents remotely and fetches only the files it needs:
all annotations and calibration files (a few MB) plus the chosen camera images
(about 2.4 MB per image).

Usage (from the project root):
    python tools/fetch_wildtrack_subset.py --list                        # show sizes, download nothing
    python tools/fetch_wildtrack_subset.py --out C:/datasets/Wildtrack --cams 1,4,6 --frames 150

Rough size: cameras x frames x 2.4 MB (3 cameras x 150 frames = about 1.1 GB).
Already downloaded files are skipped, so it can be stopped and restarted.
"""
import io
import sys
import time
import argparse
import zipfile
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
BLOCK = 4 << 20  # read 4 MB per request (fewer round trips)


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, using HTTP Range requests."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise SystemExit("Server does not support partial downloads.")
        self.pos = 0
        self._cache_start, self._cache = -1, b""
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.downloaded += len(data)
                return data
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while n > 0:
            c0 = self._cache_start
            if not (c0 <= self.pos < c0 + len(self._cache)):
                start = self.pos
                self._cache = self._fetch(start, min(self.size, start + max(BLOCK, n)))
                self._cache_start = start
            off = self.pos - self._cache_start
            chunk = self._cache[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fix_offsets(infos):
    """
    This zip was made on macOS without proper large-file (zip64) support: file
    positions beyond 4 GB are stored "wrapped around" (modulo 2^32), and Python
    then mis-reads all of them. The entries are stored in file order, so the real
    position is the wrapped value plus 2^32 times the number of wrap-arounds so far.
    """
    wraps, prev = 0, -1
    for info in infos:
        low = info.header_offset % (1 << 32)
        if low < prev:
            wraps += 1
        prev = low
        info.header_offset = low + wraps * (1 << 32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="Destination folder")
    ap.add_argument("--cams", default="1,4,6", help="Cameras 1..7")
    ap.add_argument("--frames", type=int, default=150, help="First N annotated frames (max 400)")
    ap.add_argument("--list", action="store_true", help="Only show what is in the zip")
    args = ap.parse_args()

    remote = HttpRangeFile(URL)
    zf = zipfile.ZipFile(remote)
    fix_offsets(zf.infolist())
    # Skip macOS metadata entries (__MACOSX/, ._*, .DS_Store) packed into the zip
    infos = [i for i in zf.infolist()
             if not i.filename.startswith("__MACOSX") and not PurePosixPath(i.filename).name.startswith(".")]
    print(f"zip: {remote.size / 1e9:.2f} GB, {len(infos)} files")

    def parts(info):
        return PurePosixPath(info.filename).parts

    # Find the folder layout (the zip may or may not have a top-level folder)
    img = [i for i in infos if "Image_subsets" in parts(i) and not i.is_dir()
           and PurePosixPath(i.filename).suffix.lower() in (".png", ".jpg")]
    if args.list or not args.out:
        by_dir = {}
        for i in infos:
            if i.is_dir():
                continue
            d = str(PurePosixPath(i.filename).parent)
            n, s = by_dir.get(d, (0, 0))
            by_dir[d] = (n + 1, s + i.file_size)
        for d, (n, s) in sorted(by_dir.items()):
            print(f"  {d:50s} {n:5d} files  {s / 1e6:8.1f} MB")
        if img:
            print(f"average image: {sum(i.file_size for i in img) / len(img) / 1e6:.2f} MB")
        return

    cams = {f"C{int(c)}" for c in args.cams.split(",")}
    frame_names = sorted({PurePosixPath(i.filename).stem for i in img})[:args.frames]
    keep_frames = set(frame_names)

    wanted = []
    for i in infos:
        if i.is_dir():
            continue
        p = parts(i)
        if "annotations_positions" in p:
            if PurePosixPath(i.filename).stem in keep_frames:
                wanted.append(i)
        elif "calibrations" in p:
            wanted.append(i)
        elif "Image_subsets" in p:
            cam = p[p.index("Image_subsets") + 1]
            if cam in cams and PurePosixPath(i.filename).stem in keep_frames:
                wanted.append(i)

    total = sum(i.file_size for i in wanted)
    print(f"downloading {len(wanted)} files, {total / 1e9:.2f} GB "
          f"(cameras {sorted(cams)}, {len(frame_names)} frames)")
    out = Path(args.out)
    done_bytes, t0 = 0, time.time()
    for k, info in enumerate(wanted):
        p = parts(info)
        # Strip any top-level folder so `out` directly contains Image_subsets/ etc.
        for anchor in ("Image_subsets", "annotations_positions", "calibrations"):
            if anchor in p:
                rel = Path(*p[p.index(anchor):])
                break
        dest = out / rel
        if dest.exists() and dest.stat().st_size == info.file_size:
            done_bytes += info.file_size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(str(dest) + ".part", "wb") as dst:
            while True:
                chunk = src.read(BLOCK)
                if not chunk:
                    break
                dst.write(chunk)
        Path(str(dest) + ".part").replace(dest)
        done_bytes += info.file_size
        if (k + 1) % 50 == 0 or k + 1 == len(wanted):
            rate = remote.downloaded / max(1e-6, time.time() - t0) / 1e6
            print(f"  {k + 1}/{len(wanted)} files, {done_bytes / 1e9:.2f}/{total / 1e9:.2f} GB, "
                  f"{rate:.1f} MB/s", flush=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
