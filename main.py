"""Start Life: install requirements, then run python main.py."""
from life_lab.app import LifeLabApp


def main() -> None:
    LifeLabApp().run()


if __name__ == "__main__":
    main()
