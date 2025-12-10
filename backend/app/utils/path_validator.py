"""
🛡️ Path Validation Utilities

Provides security and validation for file paths to prevent creating
directories in unexpected locations.
"""

from pathlib import Path

ALLOWED_ROOT_DIRS = {
    "backend",
    "frontend",
    "docs",
    "scripts",
    ".git",
    ".vscode",
    "node_modules",
}

PROJECT_ROOT = Path.cwd()


def validate_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        try:
            relative_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            raise ValueError(f"The absolute path must be within the project directory: {path}")
    else:
        relative_path = path

    parts = relative_path.parts
    if parts and parts[0] not in ALLOWED_ROOT_DIRS:
        if not parts[0].startswith("."):
            allowed_dirs = ", ".join(ALLOWED_ROOT_DIRS)
            raise ValueError(
                f"Creating directories in the project root directory is not allowed '{parts[0]}'. Allowed root directories: {allowed_dirs}"
            )

    return path


def safe_mkdir(path: str | Path, parents: bool = True, exist_ok: bool = True) -> Path:
    validated_path = validate_path(path)
    validated_path.mkdir(parents=parents, exist_ok=exist_ok)
    return validated_path


def safe_resolve_path(path: str | Path, base_dir: str = "backend") -> Path:
    path = Path(path)

    if path.is_absolute() or (path.parts and path.parts[0] == base_dir):
        return validate_path(path)

    safe_path = Path(base_dir) / path
    return validate_path(safe_path)


def get_project_relative_path(path: str | Path) -> str:
    path = Path(path)

    if path.is_absolute():
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    return str(path)
