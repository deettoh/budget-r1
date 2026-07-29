"""Assemble the Search-R1 retrieval index and corpus in place.

Rebuilds e5_Flat.index from its split parts by renaming part_aa then
appending part_ab, so peak extra disk stays under the ~33gb quota a
plain cat would blow. Gunzips the corpus after. Idempotent and
resumable from a partial append.

Typical usage example:

  python3 scripts/assemble_retrieval.py --data_dir retrieval_data
"""

import argparse
import gzip
import os
import shutil

# upstream part sizes for validation + idempotency, no state file
_PART_AA_BYTES = 42_949_672_960
_PART_AB_BYTES = 21_609_402_413
_INDEX_TOTAL_BYTES = _PART_AA_BYTES + _PART_AB_BYTES  # 64,559,075,373

_COPY_BUF_BYTES = 64 * 1024 * 1024  # 64 MiB streaming chunks.
_INDEX_OUT = "e5_Flat.index"
_PART_AA = "part_aa"
_PART_AB = "part_ab"
_CORPUS_GZ = "wiki-18.jsonl.gz"
_CORPUS_OUT = "wiki-18.jsonl"


def _assemble_index_inplace(data_dir: str) -> None:
    """Build e5_Flat.index from the parts without 65 GB headroom.

    Renaming part_aa onto the index path costs no copy, so only
    part_ab is streamed in. A full-size index short-circuits and a
    partial append is detected and re-done.

    Args:
        data_dir: Directory holding the index parts and the output.

    Raises:
        FileNotFoundError: A required part is absent.
        OSError: A part or the assembled index is the wrong size,
            refusing to clobber a file it cannot account for.
    """
    out = os.path.join(data_dir, _INDEX_OUT)
    pa = os.path.join(data_dir, _PART_AA)
    pb = os.path.join(data_dir, _PART_AB)

    if os.path.exists(out):
        size = os.path.getsize(out)
        if size == _INDEX_TOTAL_BYTES:
            print(f"[skip] {out} already complete ({size:,} bytes)")
            return
        if _PART_AA_BYTES <= size < _INDEX_TOTAL_BYTES and os.path.exists(pb):
            # mid-assembly rename done but append didn't finish
            print(
                f"[resume] {out} is {size:,} bytes; truncating to "
                f"{_PART_AA_BYTES:,} and re-appending {pb}"
            )
            with open(out, "r+b") as f:
                f.truncate(_PART_AA_BYTES)
        else:
            raise OSError(
                f"{out} exists at unexpected size {size:,}; refusing "
                "to clobber. Inspect manually before re-running."
            )
    else:
        if not (os.path.exists(pa) and os.path.exists(pb)):
            raise FileNotFoundError(
                f"Need both {pa} and {pb} to build {out}."
            )
        actual_aa = os.path.getsize(pa)
        if actual_aa != _PART_AA_BYTES:
            raise OSError(
                f"{pa} is {actual_aa:,} bytes, expected "
                f"{_PART_AA_BYTES:,}; aborting."
            )
        actual_ab = os.path.getsize(pb)
        if actual_ab != _PART_AB_BYTES:
            raise OSError(
                f"{pb} is {actual_ab:,} bytes, expected "
                f"{_PART_AB_BYTES:,}; aborting."
            )
        # same-fs rename is metadata-only and atomic, no copy
        print(f"[rename] {pa} -> {out} (atomic, no copy)")
        os.rename(pa, out)

    print(f"[append] {pb} -> {out} (+{_PART_AB_BYTES:,} bytes)")
    with open(out, "ab") as dst, open(pb, "rb") as src:
        shutil.copyfileobj(src, dst, _COPY_BUF_BYTES)

    final = os.path.getsize(out)
    if final != _INDEX_TOTAL_BYTES:
        raise OSError(
            f"{out} ended at {final:,} bytes, expected "
            f"{_INDEX_TOTAL_BYTES:,}. Leaving {pb} in place."
        )

    print(f"[delete] {pb} ({_PART_AB_BYTES:,} bytes freed)")
    os.remove(pb)
    print(f"[index] done: {final:,} bytes")


def _decompress_corpus(data_dir: str) -> None:
    """Decompress the gzip corpus into a plain JSONL file."""
    gz_path = os.path.join(data_dir, _CORPUS_GZ)
    out_path = os.path.join(data_dir, _CORPUS_OUT)
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"Missing corpus archive: {gz_path}")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] {out_path} already present")
        return
    print(f"[corpus] decompressing {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as out:
        shutil.copyfileobj(src, out, _COPY_BUF_BYTES)
    print(f"[corpus] done: {os.path.getsize(out_path):,} bytes")


def main() -> None:
    """Assemble the index in place and decompress the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="retrieval_data",
        help="Directory holding the index parts and corpus archive.",
    )
    args = parser.parse_args()
    _assemble_index_inplace(args.data_dir)
    _decompress_corpus(args.data_dir)
    print("[ok] retrieval data assembled")


if __name__ == "__main__":
    main()
