import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv


CSV_PATH = "data_dh-infra.csv"
OUT_DIR = "downloads"
CHECK_ONLY = False
LIMIT = None
WORKERS = 32
MANIFEST_WORKERS = 4


def load_credentials():
    load_dotenv(find_dotenv(usecwd=True))
    user = os.getenv("IIIF_USER")
    pw = os.getenv("IIIF_PASS")
    if not user or not pw:
        sys.exit("IIIF_USER / IIIF_PASS missing in .env")
    return (user, pw)


def image_urls_from_manifest(manifest):
    if "sequences" in manifest:
        canvases = manifest["sequences"][0]["canvases"]
        services = [c["images"][0]["resource"]["service"]["@id"] for c in canvases]
    elif "items" in manifest:
        canvases = manifest["items"]
        services = []
        for canvas in canvases:
            body = canvas["items"][0]["items"][0]["body"]
            service = body["service"][0]
            services.append(service.get("id") or service["@id"])
    else:
        import json
        print("Unexpected manifest structure. Top-level keys:", list(manifest.keys()))
        print(json.dumps(manifest, indent=2)[:2000])
        sys.exit(1)
    return [f"{s}/full/max/0/default.jpg" for s in services]


def fetch_manifest(row, out_root, session, auth):
    issue_dir = out_root / row["anno_id"]
    issue_dir.mkdir(parents=True, exist_ok=True)
    manifest_url = f"https://iiif-auth.onb.ac.at/presentation/ANNO/{row['anno_id']}/manifest/"
    manifest = session.get(manifest_url, auth=auth, timeout=60).json()
    urls = image_urls_from_manifest(manifest)
    paths = [issue_dir / f"p{i:03d}.jpg" for i in range(1, len(urls) + 1)]
    return row["anno_id"], list(zip(urls, paths))


def fetch_page(url, path, session, auth):
    if path.exists() and path.stat().st_size > 0:
        return path.stat().st_size
    r = session.get(url, auth=auth, timeout=120, stream=True)
    r.raise_for_status()
    size = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
            size += len(chunk)
    return size


def main():
    auth = load_credentials()
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    if CHECK_ONLY:
        rows = rows[:1]
    elif LIMIT:
        rows = rows[:LIMIT]

    session = requests.Session()

    t_manifest = time.perf_counter()
    print(f"Fetching {len(rows)} manifests...")
    issues = {}
    with ThreadPoolExecutor(max_workers=MANIFEST_WORKERS) as mpool:
        futures = [mpool.submit(fetch_manifest, row, out_root, session, auth) for row in rows]
        for n, fut in enumerate(as_completed(futures), 1):
            anno_id, pairs = fut.result()
            issues[anno_id] = pairs
            if n % 25 == 0 or n == len(rows):
                print(f"  manifests: {n}/{len(rows)}")

    all_tasks = [(anno_id, url, path) for anno_id, pairs in issues.items() for url, path in pairs]
    total_pages = len(all_tasks)
    print(
        f"Manifests done in {time.perf_counter() - t_manifest:.1f}s. "
        f"{total_pages} pages across {len(issues)} issues."
    )

    t_start = time.perf_counter()
    total_bytes = 0
    done = 0
    batch_bytes = 0
    t_batch = t_start

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_page, url, path, session, auth): anno_id
            for anno_id, url, path in all_tasks
        }
        for fut in as_completed(futures):
            size = fut.result()
            total_bytes += size
            batch_bytes += size
            done += 1
            if done % 50 == 0 or done == total_pages:
                now = time.perf_counter()
                batch_elapsed = now - t_batch
                batch_mb = batch_bytes / 1_000_000
                batch_speed = batch_mb / batch_elapsed if batch_elapsed else 0
                total_mb = total_bytes / 1_000_000
                avg_speed = total_mb / (now - t_start) if now > t_start else 0
                print(
                    f"  [{done}/{total_pages}] {total_mb:.1f} MB total, "
                    f"batch {batch_speed:.2f} MB/s, avg {avg_speed:.2f} MB/s"
                )
                batch_bytes = 0
                t_batch = now

    total_elapsed = time.perf_counter() - t_start
    total_mb = total_bytes / 1_000_000
    avg = total_mb / total_elapsed if total_elapsed else 0
    print(
        f"\nDone: {len(rows)} issues, {total_pages} pages, "
        f"{total_mb:.1f} MB in {total_elapsed:.1f}s ({avg:.2f} MB/s avg)"
    )


if __name__ == "__main__":
    main()
