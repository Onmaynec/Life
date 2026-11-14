"""Start Life, or validate an installed distribution with --smoke-test."""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        from life_lab.smoke import run_smoke
        run_smoke()
        return
    if len(sys.argv) > 1:
        raise SystemExit("Использование: Life.exe [--smoke-test]")
    from life_lab.app import LifeLabApp
    LifeLabApp().run()


if __name__ == "__main__":
    main()
