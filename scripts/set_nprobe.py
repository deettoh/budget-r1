"""Bake a frozen nprobe into an IVF index in place (updated_plan §4b).

Calibration picked nprobe=512 (recall@3 0.995 of the exhaustive-IVF
ceiling). Persisting it on the shared index freezes the operating
point across every condition without touching the divergent HPC
retrieval code.

Typical usage example:

  python3 scripts/set_nprobe.py --nprobe 512 \
    --index retrieval_data/e5_IVF.index
"""

import argparse

import faiss


def main() -> None:
    """Set the IVF nprobe on disk, skipping a no-op rewrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="print the persisted nprobe and exit")
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    ivf = faiss.extract_index_ivf(index)
    print(f"current nprobe: {ivf.nprobe}")
    if args.dry_run:
        print("dry run, no rewrite")
        return
    if ivf.nprobe == args.nprobe:
        print("already frozen at target, no rewrite")
        return
    ivf.nprobe = args.nprobe
    faiss.write_index(index, args.index)
    print(f"froze nprobe -> {args.nprobe} in {args.index}")


if __name__ == "__main__":
    main()
