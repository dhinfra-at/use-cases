"""
PageXML OCR pipeline for local XMLs that reference IIIF image URLs.

Per file:
  download IIIF image to scratch  ->  OCR (writes output XML directly)  ->
  delete scratch image

The output XML preserves the original IIIF URL in @imageFilename so results
remain portable and images can be re-fetched on demand.

Layout assumed:
  input_root  / <newspaper> / <year> / <file>.xml
  output_root / <newspaper> / <year> / <file>.xml

Use from Jupyter:
    from ocr_pipeline import run_pipeline, filter_unprocessed
    todo = filter_unprocessed(xml_files, "Layout", "OCR")
    run_pipeline(todo, "Layout", "OCR", ...)
"""

import os
import shutil
import tempfile
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import requests
import tesserocr
from PIL import Image, ImageDraw
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from ocrd_models.ocrd_page import parse, to_xml
from ocrd_models.ocrd_page_generateds import TextEquivType


# ---------------------------------------------------------------------------
# Worker side (subprocess) -- OCR + atomic output write
# ---------------------------------------------------------------------------

_worker_api = None


def _init_worker(tessdata_path: str, model_name: str) -> None:
    global _worker_api
    _worker_api = tesserocr.PyTessBaseAPI(
        path=tessdata_path,
        lang=model_name,
        psm=tesserocr.PSM.RAW_LINE,
    )


def _ocr_local_pagexml(args):
    """OCR one PageXML; write output XML atomically to its final path."""
    local_xml_in, local_image, output_xml = args
    try:
        pcgts = parse(local_xml_in, silence=True)
        page = pcgts.get_Page()

        with Image.open(local_image) as im:
            image = im.convert("RGB")

        for region in page.get_AllRegions(classes=["Text"]):
            for line in region.get_TextLine():
                line.set_TextEquiv([])

                coords_string = line.get_Coords().points
                polygon = [
                    (int(x), int(y))
                    for x, y in (pt.split(",") for pt in coords_string.split())
                ]
                if not polygon:
                    continue

                xs, ys = zip(*polygon)
                left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
                bbox = image.crop((left, top, right, bottom))

                shifted = [(x - left, y - top) for x, y in polygon]
                mask = Image.new("L", bbox.size, 0)
                ImageDraw.Draw(mask).polygon(shifted, outline=255, fill=255)
                white_bg = Image.new("RGB", bbox.size, (255, 255, 255))
                line_image = Image.composite(bbox, white_bg, mask)

                _worker_api.SetImage(line_image)
                text = _worker_api.GetUTF8Text().strip()
                conf = _worker_api.MeanTextConf() / 100.0

                if text:
                    line.add_TextEquiv(TextEquivType(Unicode=text, conf=conf))

        os.makedirs(os.path.dirname(output_xml), exist_ok=True)
        tmp = output_xml + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(to_xml(pcgts))
        os.replace(tmp, output_xml)
        return ("ok", local_xml_in, None)
    except Exception:
        return ("err", local_xml_in, traceback.format_exc())


# ---------------------------------------------------------------------------
# Driver-side: IIIF downloading with per-thread sessions + retry/backoff
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=2.0,  # 0, 2, 4, 8, 16, 32 seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        pool_connections=32,
        pool_maxsize=32,
        max_retries=retry,
    )
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "OCR-Pipeline (DHInfra Uni Graz)"})
    return s


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = _make_session()
        _thread_local.session = sess
    return sess


def _image_url_from_xml(local_xml: str) -> str:
    pcgts = parse(local_xml, silence=True)
    return pcgts.get_Page().get_imageFilename()


def _download_image(args):
    """Download the IIIF image referenced by the XML into scratch."""
    local_xml, scratch_image_path, timeout = args
    try:
        url = _image_url_from_xml(local_xml)
        if not url or not url.startswith(("http://", "https://")):
            return None, f"Not a URL: {url!r}"

        session = _get_session()
        tmp = scratch_image_path + ".tmp"
        with session.get(url, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, scratch_image_path)
        return scratch_image_path, None
    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Resume helper -- relpath-based, so same-stem files in different newspapers
# don't get confused
# ---------------------------------------------------------------------------


def _existing_output_is_valid(path: str, min_size: int = 200) -> bool:
    try:
        return os.path.getsize(path) >= min_size
    except OSError:
        return False


def filter_unprocessed(xml_files, input_root: str, output_root: str):
    """Return only XMLs whose corresponding output doesn't exist yet."""
    input_root_p = Path(input_root).resolve()
    output_root_p = Path(output_root).resolve()
    todo = []
    for src in tqdm(xml_files, desc="Checking resume", unit=" file"):
        try:
            rel = Path(src).resolve().relative_to(input_root_p)
        except ValueError:
            todo.append(src)
            continue
        out = output_root_p / rel
        if not _existing_output_is_valid(str(out)):
            todo.append(src)
    return todo


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    xml_files,
    input_root: str,
    output_root: str,
    tessdata_path: str,
    model_name: str,
    *,
    ocr_workers: int = 100,
    download_workers: int = 16,
    max_in_flight: int = 500,
    scratch_root: str | None = None,
    request_timeout: float = 60.0,
):
    """
    Run OCR over local PageXMLs whose images live behind IIIF URLs.

    input_root / output_root let the pipeline mirror your tree structure:
        input_root  = "Layout"
        output_root = "OCR"
        source      = "Layout/guv/1891/guv18911215_0.xml"
        output      = "OCR/guv/1891/guv18911215_0.xml"

    Tuning:
      ocr_workers      ~100 for a 120-core box (Tesseract loaded once each)
      download_workers 8-24; start at 16, drop if you see repeated 429s
      max_in_flight    caps disk/RAM use; 500 is comfortable on tmpfs
    """
    if not xml_files:
        print("Nothing to do.")
        return []

    input_root_p = Path(input_root).resolve()
    output_root_p = Path(output_root).resolve()
    os.makedirs(output_root_p, exist_ok=True)

    if scratch_root is None:
        base = "/dev/shm" if os.path.isdir("/dev/shm") else None
        scratch_root = tempfile.mkdtemp(prefix="ocr_iiif_", dir=base)
    else:
        os.makedirs(scratch_root, exist_ok=True)
    print(f"Scratch: {scratch_root}")

    total = len(xml_files)
    errors = []
    errors_lock = threading.Lock()
    state_lock = threading.Lock()
    completed_count = [0]
    completed_evt = threading.Event()
    in_flight = threading.Semaphore(max_in_flight)
    pools = {"ocr": None, "dl": None}
    shutting_down = threading.Event()

    pbar = tqdm(total=total, desc="OCR pipeline", unit=" file")

    # Build per-file descriptors. Index-based scratch dir prevents collisions
    # across newspapers with same-stem files.
    items = []
    for idx, source in enumerate(xml_files):
        try:
            rel = Path(source).resolve().relative_to(input_root_p)
        except ValueError:
            rel = Path(Path(source).name)
        item_dir = os.path.join(scratch_root, f"{idx:08d}")
        items.append({
            "source": source,
            "item_dir": item_dir,
            "scratch_image": os.path.join(item_dir, "image"),
            "output": str(output_root_p / rel),
        })

    def mark_done(item, error_kind=None, error_msg=None):
        if error_kind:
            with errors_lock:
                errors.append((item["source"], f"[{error_kind}] {error_msg}"))
        shutil.rmtree(item["item_dir"], ignore_errors=True)
        with state_lock:
            completed_count[0] += 1
            pbar.update(1)
            if completed_count[0] >= total:
                completed_evt.set()
        in_flight.release()

    def submit_safe(pool_key, fn, *args):
        if shutting_down.is_set():
            return None
        try:
            return pools[pool_key].submit(fn, *args)
        except RuntimeError:
            return None

    def download_callback(item):
        def cb(fut):
            try:
                result, err = fut.result()
            except Exception:
                mark_done(item, "download", traceback.format_exc())
                return
            if err or result is None:
                mark_done(item, "download", err or "unknown")
                return
            ocr_fut = submit_safe(
                "ocr", _ocr_local_pagexml,
                (item["source"], result, item["output"]),
            )
            if ocr_fut is None:
                mark_done(item, "download", "pool shutdown before OCR submit")
                return
            ocr_fut.add_done_callback(ocr_callback(item))
        return cb

    def ocr_callback(item):
        def cb(fut):
            try:
                status, _, tb = fut.result()
            except Exception:
                mark_done(item, "ocr", traceback.format_exc())
                return
            if status != "ok":
                mark_done(item, "ocr", tb)
                return
            mark_done(item)
        return cb

    try:
        with ProcessPoolExecutor(
            max_workers=ocr_workers,
            initializer=_init_worker,
            initargs=(tessdata_path, model_name),
        ) as ocr_pool, ThreadPoolExecutor(
            max_workers=download_workers, thread_name_prefix="dl"
        ) as dl_pool:
            pools["ocr"] = ocr_pool
            pools["dl"] = dl_pool

            for item in items:
                in_flight.acquire()
                if shutting_down.is_set():
                    in_flight.release()
                    break
                os.makedirs(item["item_dir"], exist_ok=True)
                dl_fut = dl_pool.submit(
                    _download_image,
                    (item["source"], item["scratch_image"], request_timeout),
                )
                dl_fut.add_done_callback(download_callback(item))

            completed_evt.wait()
    except KeyboardInterrupt:
        shutting_down.set()
        print("\nInterrupted; shutting down...")
        raise
    finally:
        pbar.close()
        shutil.rmtree(scratch_root, ignore_errors=True)

    if errors:
        print(f"\n{len(errors)} error(s). First 10:")
        for src, e in errors[:10]:
            print(f"\n--- {src} ---\n{e[:500]}")
    else:
        print("All files processed successfully.")
    return errors
