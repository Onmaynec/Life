"""Start Life, or validate an installed distribution with smoke checks."""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        from life_lab.smoke import run_smoke
        run_smoke()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--ui-smoke-test":
        if len(sys.argv) > 3:
            raise SystemExit("Использование: Life.exe --ui-smoke-test [папка отчёта]")
        from life_lab.ui_smoke import run_ui_smoke
        run_ui_smoke(sys.argv[2] if len(sys.argv) == 3 else None)
        return
    if len(sys.argv) > 1:
        raise SystemExit("Использование: Life.exe [--smoke-test | --ui-smoke-test [папка отчёта]]")
    from life_lab.app import LifeLabApp
    LifeLabApp().run()


if __name__ == "__main__":
    main()
