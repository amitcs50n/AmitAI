"""Print a fresh inference-session credential without storing it."""

import secrets
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        raise SystemExit("Usage: python -m runtime.inference_token (no arguments)")
    print(secrets.token_urlsafe(32))


if __name__ == "__main__":
    main()
