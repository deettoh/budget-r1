"""Bake a frozen nprobe into an IVF index in place (updated_plan §4b).

Calibration picked nprobe=512 (recall@3 0.995 of the exhaustive-IVF
ceiling). Persisting it on the shared index freezes the operating
point across every condition without touching the divergent HPC
retrieval code.
"""

import argparse

import faiss


def main() -> None:
    """Set the IVF nprobe on disk, skipping a no-op rewrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    ivf = faiss.extract_index_ivf(index)
    print(f"current nprobe: {ivf.nprobe}")
    if ivf.nprobe == args.nprobe:
        print("already frozen at target, no rewrite")
        return
    ivf.nprobe = args.nprobe
    faiss.write_index(index, args.index)
    print(f"froze nprobe -> {args.nprobe} in {args.index}")


if __name__ == "__main__":
    main()
