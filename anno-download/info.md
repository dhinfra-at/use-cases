# IIIF Parallel Download

Bulk-downloads ANNO newspaper issues from the authenticated IIIF endpoint, given a CSV of `anno_id` values.

## Approach

Two-phase pipeline to keep workers saturated:

1. **Manifest phase** — `MANIFEST_WORKERS` threads fetch each issue's IIIF manifest in parallel and extract every page's image URL.
2. **Download phase** — all page URLs from all issues are flattened into one task list, then a single `WORKERS`-sized thread pool downloads them via `as_completed`. No per-issue barrier, so threads never idle waiting for the next manifest.

Existing files (non-zero size) are skipped, so the script is resumable.

## How it was tested

- Ran against a small slice of the CSV (`LIMIT` set to a handful of rows) to confirm manifests parse and pages land under `downloads/<anno_id>/pNNN.jpg`.
- Watched the batch throughput line (`batch X.XX MB/s, avg X.XX MB/s`, every 50 pages) to verify workers stay busy across issue boundaries.
- Re-ran without deleting output to confirm the skip-if-exists path short-circuits already-downloaded pages.

## Config

All configurations are constants at the top of `iiif_download.py` (`CSV_PATH`, `OUT_DIR`, `WORKERS`, `MANIFEST_WORKERS`, `LIMIT`, `CHECK_ONLY`). Credentials come from `.env` as `IIIF_USER` / `IIIF_PASS` (not uploaded here).
