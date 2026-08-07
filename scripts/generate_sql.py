"""Generate SQL for one investigator request using the Falcon SQL adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.schema import render_schema_for_prompt
from sql_model.inference import FalconSQLGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="Natural-language investigation request.")
    return parser.parse_args()


def main() -> None:
    """Generate SQL and print it to stdout."""
    args = parse_args()
    generator = FalconSQLGenerator()
    result = generator.generate_sql(args.instruction, schema=render_schema_for_prompt())
    print(result.sql)


if __name__ == "__main__":
    main()
