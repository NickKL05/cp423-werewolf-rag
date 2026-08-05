"""Reproduce every experimental result and table reported in the write-up.

    python run_all.py

Runs, in order:

    1. Preprocess   rebuild documents and chunks from the committed snapshot
    2. Index        embed the chunks and cache the dense index
    3. Diagnostic   closed-book corpus suitability check (report section 2)
    4. Evaluate     all four systems against the gold evaluation set

The network crawl is not part of this command. The corpus snapshot is committed
to the repository so that results are reproducible even as the wiki changes.
Pass --crawl to refetch it, which will produce a different snapshot and
therefore different chunk IDs.

All random seeds are fixed in src/wolfrag/config.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from wolfrag import config, generate  # noqa: E402


def run_step(name: str, module: str, args: list[str] | None = None) -> None:
    print("\n" + "=" * 72)
    print(f"STEP: {name}")
    print("=" * 72, flush=True)

    started = time.time()
    env_path = str(SRC)
    command = [sys.executable, "-m", module] + (args or [])
    result = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent),
        env={**dict(__import__("os").environ), "PYTHONPATH": env_path},
    )
    if result.returncode != 0:
        raise SystemExit(f"\nStep {name!r} failed with exit code {result.returncode}.")
    print(f"\n[{name} finished in {time.time() - started:.1f}s]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Refetch the corpus from the wiki instead of using the snapshot.",
    )
    parser.add_argument(
        "--skip-diagnostic",
        action="store_true",
        help="Skip the closed-book diagnostic.",
    )
    args = parser.parse_args()

    if not config.RAW_CORPUS_PATH.exists() and not args.crawl:
        raise SystemExit(
            f"No corpus snapshot at {config.RAW_CORPUS_PATH}. "
            "Run 'python run_all.py --crawl' to build one."
        )

    if not generate.check_server():
        raise SystemExit(
            f"Ollama is not reachable at {config.OLLAMA_HOST}, or the model "
            f"{config.GENERATION_MODEL} is not installed.\n"
            f"Start Ollama, then run: ollama pull {config.GENERATION_MODEL}"
        )

    if not config.GOLD_QUESTIONS_PATH.exists():
        raise SystemExit(
            f"No gold question set at {config.GOLD_QUESTIONS_PATH}. "
            "Run 'python -m wolfrag.make_gold_template' and fill it in."
        )

    started = time.time()

    if args.crawl:
        run_step("Crawl corpus from wiki", "wolfrag.crawl")

    run_step("Preprocess and chunk", "wolfrag.preprocess")
    run_step("Build dense index", "wolfrag.retrieval")

    if not args.skip_diagnostic:
        run_step("Closed-book diagnostic", "wolfrag.diagnostic")

    run_step("Evaluate all systems", "wolfrag.evaluate")

    print("\n" + "=" * 72)
    print(f"All steps completed in {time.time() - started:.1f}s.")
    print("=" * 72)
    print(f"\nResults are in {config.RESULTS_DIR}:")
    for path in sorted(config.RESULTS_DIR.glob("*")):
        print(f"  {path.name}")
    print("\nThe report tables are in results/tables.md")


if __name__ == "__main__":
    main()
