"""Download vanilla Qwen2.5-3B-Instruct to the HPC.

The no-RAG and naive-RAG baselines use the untrained instruct model
(NOT the Search-R1 GRPO checkpoint sr1-3b-it) to measure parametric
and single-shot-retrieval ability without agentic-search training.
Run once; safe to re-run (snapshot is resumable and content-addressed).
"""

import os

from huggingface_hub import snapshot_download

CACHE_DIR = "/media/D1/ait_users/staff18/hf_cache"
REPOS = {
    "Qwen/Qwen2.5-3B-Instruct": "models/qwen2.5-3b-it",
}


def main() -> None:
    """Download every repo in REPOS into its destination dir."""
    for repo_id, dest in REPOS.items():
        os.makedirs(dest, exist_ok=True)
        print(f"[download] {repo_id} -> {dest}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            cache_dir=CACHE_DIR,
            max_workers=4,
        )
        print(f"[download] DONE {repo_id}", flush=True)


if __name__ == "__main__":
    main()
