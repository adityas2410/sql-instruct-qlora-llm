"""Generate linked synthetic insurance investigation records as JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.data_generation import SyntheticDataConfig, SyntheticInsuranceDataGenerator


def parse_args() -> argparse.Namespace:
    """Parse generator command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2410, help="Deterministic random seed.")
    parser.add_argument("--customers", type=int, default=200, help="Number of customers.")
    parser.add_argument("--repair-shops", type=int, default=30, help="Number of repair shops.")
    parser.add_argument("--normal-claims", type=int, default=500, help="Number of normal claims.")
    parser.add_argument(
        "--fraudulent-claims",
        type=int,
        default=80,
        help="Number of historically fraudulent claims.",
    )
    parser.add_argument(
        "--coordinated-groups",
        type=int,
        default=6,
        help="Number of coordinated claim groups.",
    )
    parser.add_argument(
        "--coordinated-group-size",
        type=int,
        default=8,
        help="Claims per coordinated group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory where table JSON files are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate synthetic data and write JSON tables."""
    args = parse_args()
    config = SyntheticDataConfig(
        seed=args.seed,
        customer_count=args.customers,
        repair_shop_count=args.repair_shops,
        normal_claim_count=args.normal_claims,
        fraudulent_claim_count=args.fraudulent_claims,
        coordinated_group_count=args.coordinated_groups,
        coordinated_group_size=args.coordinated_group_size,
        output_dir=args.output_dir,
    )
    generator = SyntheticInsuranceDataGenerator(config)
    dataset = generator.generate()
    generator.write(dataset)

    table_counts = ", ".join(f"{name}={len(records)}" for name, records in dataset.items())
    print(f"Generated synthetic insurance data in {config.output_dir}: {table_counts}")


if __name__ == "__main__":
    main()
