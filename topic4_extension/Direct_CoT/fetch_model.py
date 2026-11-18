"""Download and verify the exact Qwen GGUF used in the archived Local1 run."""

import argparse
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from itertools import count
from urllib.request import Request, urlopen

from run_direct_cot import MODEL_HASH, ROOT, sha256


MODEL_SIZE = 1_359_279_776
MODEL_URL = f"https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:{MODEL_HASH}"
MODEL_PATH = ROOT / "_runtime/qwen3-1.7b-Q4_K_M.gguf"
CHUNK_SIZE = 8 * 1024**2


def download_range(folder, start, end, keep_trying):
    path = folder / f"{start:010d}-{end:010d}.part"
    size = end - start + 1
    for attempt in count():
        downloaded = path.stat().st_size if path.exists() else 0
        if downloaded == size:
            return path
        if downloaded > size:
            raise ValueError(f"Oversized download fragment: {path}")
        first = start + downloaded
        req = Request(MODEL_URL, headers={"Range": f"bytes={first}-{end}", "Accept-Encoding": "identity"})
        try:
            with urlopen(req, timeout=20) as response:
                expected_range = f"bytes {first}-{end}/{MODEL_SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise OSError(f"Unexpected response for {path.name}: {response.status}")
                with path.open("ab") as output:
                    remaining = size - downloaded
                    while remaining:
                        block = response.read(min(256 * 1024, remaining))
                        if not block:
                            raise OSError(f"Incomplete range {path.name}")
                        output.write(block)
                        remaining -= len(block)
                return path
        except (OSError, IncompleteRead) as error:
            if not keep_trying and attempt == 9:
                raise RuntimeError(f"Cannot download {path.name}: {error}") from error
            if keep_trying and attempt and attempt % 20 == 0:
                print(f"Still retrying range {path.name} at {downloaded:,}/{size:,} bytes ({type(error).__name__})", flush=True)
            time.sleep(min(0.5 * 2**min(attempt, 4), 6) + random.random())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-trying", action="store_true", help="continue through transient connection failures")
    args = parser.parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        if MODEL_PATH.stat().st_size != MODEL_SIZE or sha256(MODEL_PATH) != MODEL_HASH:
            raise ValueError(f"Existing model failed SHA-256 verification: {MODEL_PATH}")
        print(f"Verified {MODEL_PATH}")
        return

    # Reuse the contiguous prefix from the initial download if available.
    prefix = MODEL_PATH.with_suffix(".gguf.part")
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    if prefix_size > MODEL_SIZE:
        raise ValueError(f"Oversized model prefix: {prefix}")
    chunk_dir = MODEL_PATH.parent / "model_ranges"
    chunk_dir.mkdir(exist_ok=True)
    ranges = [(start, min(start + CHUNK_SIZE, MODEL_SIZE) - 1)
              for start in range(prefix_size, MODEL_SIZE, CHUNK_SIZE)]
    print(f"Resuming from {prefix_size:,} verified-by-final-hash prefix bytes; {len(ranges)} ranges.", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = {pool.submit(download_range, chunk_dir, start, end, args.keep_trying): (start, end)
                   for start, end in ranges}
        fragments = {}
        total = prefix_size
        for count, future in enumerate(as_completed(pending), 1):
            start, end = pending[future]
            fragments[start] = future.result()
            total += end - start + 1
            if count == 1 or count % 5 == 0 or total == MODEL_SIZE:
                print(f"{total / 1e6:.0f}/{MODEL_SIZE / 1e6:.0f} MB ready", flush=True)

    staged = MODEL_PATH.with_suffix(".gguf.assembling")
    with staged.open("wb") as output:
        if prefix_size:
            with prefix.open("rb") as fragment:
                shutil.copyfileobj(fragment, output, length=4 * 1024**2)
        for start in sorted(fragments):
            with fragments[start].open("rb") as fragment:
                shutil.copyfileobj(fragment, output, length=4 * 1024**2)
    if staged.stat().st_size != MODEL_SIZE or sha256(staged) != MODEL_HASH:
        raise ValueError("Combined model failed SHA-256 verification")
    staged.replace(MODEL_PATH)
    for fragment in fragments.values():
        fragment.unlink()
    if prefix.exists():
        prefix.unlink()
    print(f"Verified SHA-256 {MODEL_HASH}\n{MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
