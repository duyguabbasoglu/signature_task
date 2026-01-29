import argparse
import logging
import sys
from pathlib import Path

from bbox_detector.detector import (
    DetectionResult,
    analyze,
    process_folder,
)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def print_result(name: str, result):
    punct_str = "N/A" if result.is_punctuation is None else (
        "Yes" if result.is_punctuation else "No"
    )
    empty_str = "Yes" if result.is_empty else "No"

    print(f"{name:<30} | {result.result.value:<15} | {empty_str:<5} | "
          f"{punct_str:<5} | {result.total_area:>6} px²")


def analyze_single(path: str):
    result = analyze(path)
    print(f"\nFile: {path}")
    print(f"Result: {result.result.value}")
    print(f"Empty: {result.is_empty}")
    print(f"Punctuation: {result.is_punctuation}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Area: {result.total_area} px²")
    print(f"Components: {result.component_count}")


def analyze_folder(folder: str, limit: int = None):
    print("=" * 75)
    print("BOUNDING BOX CONTENT DETECTION")
    print("=" * 75)
    print()

    results = process_folder(folder)

    if not results:
        print(f"No images found in '{folder}'")
        return

    if limit:
        results = results[:limit]

    print(f"{'FILE':<30} | {'RESULT':<15} | {'EMPTY':<5} | {'PUNCT':<5} | {'AREA'}")
    print("-" * 75)

    summary = {
        DetectionResult.EMPTY: 0,
        DetectionResult.FILLED_PUNCT: 0,
        DetectionResult.FILLED_OTHER: 0
    }

    for item in results:
        res = item["analysis"]
        summary[res.result] += 1
        print_result(item["name"], res)

    print("-" * 75)
    total = len(results)
    print(f"\nSummary ({total} files):")
    for result_type, count in summary.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {result_type.value:<15}: {count:>4} ({pct:5.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="bbox-detect",
        description="Bounding Box Content Detection - Detect punctuation vs other content"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data",
        help="Image file or folder to analyze (default: data/)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit number of files to process"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    path = Path(args.path)

    try:
        if path.is_file():
            analyze_single(str(path))
        elif path.is_dir():
            analyze_folder(str(path), args.limit)
        else:
            print(f"Error: '{path}' not found", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
