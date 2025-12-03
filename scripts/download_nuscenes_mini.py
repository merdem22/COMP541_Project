"""
Helper to validate nuScenes-mini download location.

nuScenes terms prohibit redistribution, so this script only checks whether the
expected folders are present and prints manual download links.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="nuScenes-mini download helper")
    parser.add_argument("--data-root", default="data/nuscenes", type=Path)
    args = parser.parse_args()

    data_root: Path = args.data_root
    expected = ["samples", "sweeps", "maps", "v1.0-mini"]

    if all((data_root / name).exists() for name in expected):
        print(f"[ok] Found nuScenes-mini at {data_root}")
        return

    print("[info] nuScenes-mini not fully present.")
    print("Manual download required (license restricted): https://www.nuscenes.org/download")
    print("Place the extracted contents so you have:")
    for name in expected:
        print(f"  - {data_root}/{name}")


if __name__ == "__main__":
    main()
