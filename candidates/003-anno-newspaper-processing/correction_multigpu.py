"""
Multi-GPU ByT5 post-correction.

Loads one model copy per GPU, distributes work across all of them with
length-balanced chunking. Designed to be imported from a Jupyter notebook.

Usage:
    from correction_multigpu import post_correct_multi_gpu

    corrected = post_correct_multi_gpu(
        df_split["split_text"].tolist(),
        model_path="path/to/your/finetuned/byt5",
        gpu_ids=[0, 1],
        batch_size=130,
        max_input_len=150,
        num_beams=4,
    )
"""

import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Empty

import numpy as np
from tqdm.auto import tqdm

# Worker-local state (set in initializer, used in task)
_model = None
_tokenizer = None
_dtype = None
_progress_queue = None


# --- ByT5 fast tokenize/decode -----------------------------------------------
# ByT5 vocab: 0=<pad>, 1=</s>, 2=<unk>, 3..258 = bytes 0..255, 259+ = sentinels
BYT5_PAD_ID = 0
BYT5_EOS_ID = 1
BYT5_BYTE_OFFSET = 3


def fast_byt5_encode(texts, max_length):
    """
    Drop-in replacement for HuggingFace ByT5Tokenizer for our use case.
    5-10x faster because it skips PreTrainedTokenizer's generic overhead.
    Returns {"input_ids", "attention_mask"} as torch.long tensors, right-padded.
    """
    import torch

    # Truncate to max_length - 1 bytes (last slot is EOS)
    byte_seqs = [t.encode("utf-8")[: max_length - 1] for t in texts]
    lengths = [len(b) + 1 for b in byte_seqs]  # +1 for EOS
    max_len = max(lengths) if lengths else 1
    n = len(texts)

    input_ids = np.zeros((n, max_len), dtype=np.int64)
    attention_mask = np.zeros((n, max_len), dtype=np.int64)

    for i, b in enumerate(byte_seqs):
        L = len(b)
        if L > 0:
            # bytes -> uint8 array -> int64 + offset, all in numpy
            input_ids[i, :L] = (
                np.frombuffer(b, dtype=np.uint8).astype(np.int64) + BYT5_BYTE_OFFSET
            )
        input_ids[i, L] = BYT5_EOS_ID
        attention_mask[i, : L + 1] = 1

    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention_mask),
    }


def fast_byt5_decode(ids_tensor):
    """
    Drop-in replacement for tokenizer.batch_decode(..., skip_special_tokens=True).
    Discards pad/eos/unk/sentinels, keeps only byte tokens.
    """
    if hasattr(ids_tensor, "cpu"):
        ids = ids_tensor.cpu().numpy()
    else:
        ids = ids_tensor

    out = []
    for row in ids:
        # Keep only tokens that map to real bytes
        mask = (row >= BYT5_BYTE_OFFSET) & (row < BYT5_BYTE_OFFSET + 256)
        byte_vals = (row[mask] - BYT5_BYTE_OFFSET).astype(np.uint8)
        out.append(byte_vals.tobytes().decode("utf-8", errors="ignore"))
    return out


def _init_worker(gpu_queue, model_path, dtype_str, progress_queue):
    """Runs once per worker. Pins to a GPU and loads the model."""
    global _model, _tokenizer, _dtype, _progress_queue

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _progress_queue = progress_queue

    # Import torch ONLY after setting CUDA_VISIBLE_DEVICES
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    _dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=_dtype
    ).to("cuda")
    _model.eval()


def _process_chunk(texts, batch_size, max_input_len, num_beams, length_buffer):
    """
    Run correction on one chunk with CPU/GPU pipelining.

    Three stages run concurrently:
      - Producer thread: tokenize next batch
      - Main thread:     run model.generate on current batch
      - Consumer thread: decode previous output

    Pinned memory + non-blocking transfers overlap H2D copies with compute,
    so the GPU is not starved between batches.
    """
    import threading
    from queue import Queue

    import torch

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    if not batches:
        return []

    n = len(batches)
    # Small queues = backpressure (avoid building huge prep backlog)
    tokenize_q = Queue(maxsize=3)
    decode_q = Queue(maxsize=3)
    results = [None] * n

    def producer():
        for idx, batch in enumerate(batches):
            enc = fast_byt5_encode(batch, max_input_len)
            # Pin so the H2D copy in the main thread can be async
            enc = {k: v.pin_memory() for k, v in enc.items()}
            tokenize_q.put((idx, enc))
        tokenize_q.put(None)

    def consumer():
        while True:
            item = decode_q.get()
            if item is None:
                break
            idx, out_cpu = item
            decoded = fast_byt5_decode(out_cpu)
            results[idx] = [t.strip() for t in decoded]
            if _progress_queue is not None:
                try:
                    _progress_queue.put(len(decoded))
                except Exception:
                    pass

    prod = threading.Thread(target=producer, daemon=True)
    cons = threading.Thread(target=consumer, daemon=True)
    prod.start()
    cons.start()

    try:
        with torch.inference_mode():
            while True:
                item = tokenize_q.get()
                if item is None:
                    break
                idx, enc = item
                enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
                out = _model.generate(
                    **enc,
                    max_new_tokens=max_input_len + length_buffer,
                    num_beams=num_beams,
                    do_sample=False,
		    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=True,
                )
                # Move to CPU here (blocking sync with the GPU stream).
                # Doing it in the main thread avoids cross-thread CUDA
                # stream ordering surprises.
                decode_q.put((idx, out.cpu()))
    finally:
        decode_q.put(None)

    prod.join()
    cons.join()

    return [t for batch_res in results for t in batch_res]


def post_correct_multi_gpu(
    texts,
    model_path,
    gpu_ids=(0, 1),
    batch_size=130,
    max_input_len=150,
    num_beams=4,
    length_buffer=8,
    n_chunks=None,
    dtype="bfloat16",
):
    """
    Post-correct a list of texts using all GPUs in `gpu_ids`.

    Returns a list of corrected strings in the same order as `texts`.
    """
    if n_chunks is None:
        n_chunks = 20 * len(gpu_ids)

    # Sort by length for efficient batching; remember inverse permutation
    order = np.argsort([len(t) for t in texts])
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    sorted_texts = [texts[i] for i in order]

    # Stripe sorted texts across chunks. Each chunk gets a uniform sample of
    # the length distribution, so both per-chunk memory peak and per-chunk
    # total work are balanced. Within a chunk, texts remain length-sorted,
    # so batch padding stays efficient.
    chunks = [sorted_texts[i::n_chunks] for i in range(n_chunks)]
    chunks = [c for c in chunks if c]

    # Pass GPU ids to workers via a queue (one ID consumed per worker init)
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Manager().Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    # Progress queue: workers push batch sizes; reader thread updates tqdm
    progress_queue = ctx.Manager().Queue()
    total_items = sum(len(c) for c in chunks)
    pbar = tqdm(total=total_items, desc="Correcting", unit=" text")
    stop_reader = threading.Event()

    def _reader():
        while not stop_reader.is_set():
            try:
                n = progress_queue.get(timeout=0.5)
            except Empty:
                continue
            if n is None:
                return
            pbar.update(n)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    # Map each chunk index to which one will be striped back together
    chunk_results = [None] * len(chunks)
    try:
        with ProcessPoolExecutor(
            max_workers=len(gpu_ids),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(gpu_queue, model_path, dtype, progress_queue),
        ) as pool:
            futures = {
                pool.submit(
                    _process_chunk, c, batch_size, max_input_len, num_beams, length_buffer
                ): i
                for i, c in enumerate(chunks)
            }
            for fut in as_completed(futures):
                chunk_results[futures[fut]] = fut.result()
    finally:
        stop_reader.set()
        progress_queue.put(None)
        reader.join(timeout=5.0)
        pbar.close()

    # Un-stripe: chunk i contained sorted_texts[i::n_chunks]; results in same order
    sorted_results = [None] * len(sorted_texts)
    for chunk_idx, chunk_out in enumerate(chunk_results):
        sorted_results[chunk_idx::len(chunks)] = chunk_out

    # Restore original order
    return [sorted_results[i] for i in inv_order]
