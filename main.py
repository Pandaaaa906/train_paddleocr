"""Training launcher for fine-tuning PP-DocLayoutV3 with PaddleX.

Usage:
    python main.py --mode train --device gpu:0
    python main.py --mode train --device gpu:0 --resume output/ppdoclayoutv3_ft/epoch_10
    python main.py --mode check_dataset
    python main.py --mode eval
    python main.py --mode export
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("./configs/PP-DocLayoutV3.yaml")


def run_engine(config_path: Path, mode: str, overrides: list[str]) -> int:
    """Run PaddleX Engine with the given config and mode."""
    original_argv = sys.argv
    try:
        sys.argv = [
            "paddlex",
            "-c",
            str(config_path.absolute()),
            "-o",
            f"Global.mode={mode}",
        ]
        for o in overrides:
            sys.argv.extend(["-o", o])

        from paddlex.engine import Engine

        engine = Engine()
        engine.run()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune PP-DocLayoutV3 for chemical structure detection."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to PaddleX config YAML",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "export", "check_dataset"],
        default="train",
        help="PaddleX pipeline mode",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Device string (e.g. gpu:0, cpu)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Override config key (can be used multiple times)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    overrides = list(args.override)
    overrides.append(f"Global.device={args.device}")

    if args.resume and args.mode == "train":
        overrides.append(f"Train.resume_path={args.resume}")

    mode_map = {
        "train": "train",
        "eval": "evaluate",
        "export": "export",
        "check_dataset": "check_dataset",
        "pdparams2safetensors": "pdparams2safetensors",
    }
    mode = mode_map[args.mode]

    ret = run_engine(args.config, mode, overrides)

    return ret


if __name__ == "__main__":
    sys.exit(main())
