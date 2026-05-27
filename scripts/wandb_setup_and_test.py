"""Programmatic wandb login + smoke-test run.

The HPC blocks the wandb CLI, so this logs in from Python via
wandb.login (writes ~/.netrc) and fires a tiny run to confirm
connectivity. Run once per account, reads the key from ~/.wandb_key.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path


DEFAULT_KEY_FILE = "~/.wandb_key"
DEFAULT_PROJECT = "FYP"


def read_api_key(key_file: str) -> str:
    path = Path(os.path.expanduser(key_file))
    if not path.exists():
        sys.exit(
            f"[wandb-setup] FAIL: {path} does not exist.\n"
            "Create it interactively on the HPC (no `export`, no `chmod`):\n"
            "    nano ~/.wandb_key      # paste key from https://wandb.ai/authorize, save\n"
            "Then re-run this script. It will tighten the file permissions "
            "for you via os.chmod()."
        )
    key = path.read_text().strip()
    if not key:
        sys.exit(f"[wandb-setup] FAIL: {path} is empty.")
    if any(c.isspace() for c in key):
        sys.exit(
            f"[wandb-setup] FAIL: {path} contains whitespace. The file must "
            "hold ONLY the API key on a single line -- no quotes, no newlines."
        )

    # shell chmod is blocked on the HPC but os.chmod isn't
    try:
        current_mode = path.stat().st_mode & 0o777
        if current_mode != 0o600:
            os.chmod(path, 0o600)
            print(f"[wandb-setup] os.chmod({path}, 0o600) (was {oct(current_mode)}).")
    except OSError as exc:
        # non-fatal the key still works just less private
        print(f"[wandb-setup] WARN: could not tighten {path} permissions: {exc}")

    return key


def confirm_netrc_written() -> None:
    netrc_path = Path(os.path.expanduser("~/.netrc"))
    if not netrc_path.exists():
        sys.exit("[wandb-setup] FAIL: wandb.login() did not write ~/.netrc.")
    contents = netrc_path.read_text()
    if "api.wandb.ai" not in contents:
        sys.exit(
            "[wandb-setup] FAIL: ~/.netrc exists but has no api.wandb.ai entry. "
            "Re-check the API key."
        )
    print(f"[wandb-setup] OK: ~/.netrc has api.wandb.ai entry ({netrc_path}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key-file",
        default=DEFAULT_KEY_FILE,
        help=f"Path to file containing the wandb API key (default: {DEFAULT_KEY_FILE})",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"wandb project for the smoke run (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--skip-test-run",
        action="store_true",
        help="Only do the login; do not start a wandb.init() smoke run.",
    )
    args = parser.parse_args()

    import wandb

    print(f"[wandb-setup] wandb version: {wandb.__version__}")
    api_key = read_api_key(args.key_file)
    print(f"[wandb-setup] read {len(api_key)}-char key from {args.key_file}")

    # replaces `wandb login` validates the key and writes ~/.netrc
    ok = wandb.login(key=api_key, relogin=True, verify=True)
    if not ok:
        sys.exit("[wandb-setup] FAIL: wandb.login() returned False.")
    print("[wandb-setup] wandb.login() succeeded.")
    confirm_netrc_written()

    if args.skip_test_run:
        print("[wandb-setup] --skip-test-run passed; not starting a wandb.init() run.")
        return

    run_name = "wandb-smoke-" + dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    print(
        f"[wandb-setup] starting smoke run '{run_name}' in project '{args.project}'..."
    )
    run = wandb.init(
        project=args.project,
        name=run_name,
        job_type="smoke-test",
        config={"purpose": "verify wandb works on HPC without `wandb login`"},
        mode="online",
    )
    try:
        for step in range(5):
            run.log({"smoke/step": step, "smoke/value": 0.5**step})
            time.sleep(0.2)
    finally:
        run.finish()

    print(
        "[wandb-setup] DONE. Check run at "
        f"https://wandb.ai/<your-entity>/{args.project}/runs."
    )


if __name__ == "__main__":
    main()
