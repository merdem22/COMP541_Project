"""
Create lightweight artifacts for training loops (token lists, quick stats).
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nuscenes_dataset import NuScenesDetectionDataset
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nuScenes-mini metadata.")
    parser.add_argument("--data-root", default="data/nuscenes", type=Path)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", default="data/nuscenes/cache/sample_tokens.json", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    args = parser.parse_args()

    logger = setup_logging(name="prepare_data")
    dataset = NuScenesDetectionDataset(
        data_root=str(args.data_root),
        version=args.version,
        load_annotations=False,
        use_images=False,
        load_lidar=False,
    )

    tokens = dataset.sample_tokens
    if args.limit:
        tokens = tokens[: args.limit]
    logger.info("Found %d samples for %s", len(tokens), args.version)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"version": args.version, "tokens": tokens}, f, indent=2)
    logger.info("Wrote token list to %s", args.output)


if __name__ == "__main__":
    main()
