from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT = "python_code_bundle.txt"
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    ".venv314",
    "venv",
    "env",
    "bin",
    "Scripts",
    "Include",
    "Lib",
    "lib",
    "etc",
    "share",
}


def should_skip(path: Path) -> bool:
    """Return True if file is inside a directory that should be ignored."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def read_text_with_fallback(file_path: Path) -> str:
    """Read text robustly with UTF-8 first, then fallback encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "cp1252", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # Last-resort fallback preserving bytes.
    return file_path.read_text(encoding="utf-8", errors="replace")


def gather_python_files(project_root: Path) -> list[Path]:
    """Collect all .py files from project root excluding known environment dirs."""
    files: list[Path] = []
    for py_file in project_root.rglob("*.py"):
        if should_skip(py_file.relative_to(project_root)):
            continue
        files.append(py_file)
    return sorted(files, key=lambda p: p.relative_to(project_root).as_posix().lower())


def build_bundle(project_root: Path, files: list[Path]) -> str:
    """Build combined text output with section headers for each Python file."""
    sections: list[str] = []
    separator = "=" * 100

    for file_path in files:
        relative_path = file_path.relative_to(project_root).as_posix()
        content = read_text_with_fallback(file_path)
        section = (
            f"{separator}\n"
            f"FILE: {relative_path}\n"
            f"{separator}\n"
            f"{content.rstrip()}\n"
        )
        sections.append(section)

    return "\n\n".join(sections) + ("\n" if sections else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all .py files from a project and save them into one .txt file "
            "with sections containing relative path and code."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output .txt file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    project_root = args.root.resolve()
    output_path = args.output if args.output.is_absolute() else (project_root / args.output)

    py_files = gather_python_files(project_root)
    bundled_text = build_bundle(project_root, py_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundled_text, encoding="utf-8")

    print(f"Project root: {project_root}")
    print(f"Python files bundled: {len(py_files)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
