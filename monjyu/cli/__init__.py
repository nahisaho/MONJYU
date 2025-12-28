# MONJYU CLI
"""
FEAT-008: CLI (Command Line Interface)
メインエントリーポイント
"""

from monjyu.cli.main import app, console

__all__ = ["app", "console", "main"]


def main():
    """CLI entry point for pyproject.toml"""
    app()


if __name__ == "__main__":
    main()
