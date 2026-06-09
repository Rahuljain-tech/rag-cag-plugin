import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

from app.core.config import settings
from app.rag.chunking import chunk_text

DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    "venv",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    ".mypy_cache",
    ".cursor",
}

DEFAULT_IGNORE_GLOBS: Tuple[str, ...] = (
    "*.pyc",
    "*.db",
    ".env*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
)

DEFAULT_TEXT_EXTENSIONS: Tuple[str, ...] = (
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".rs",
    ".go",
    ".java",
    ".html",
    ".css",
    ".sql",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
)


def normalize_query(text: str) -> str:
    return " ".join(text.strip().split())


def _load_gitignore_patterns(root: Path) -> List[str]:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []

    patterns: List[str] = []
    for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _matches_gitignore(relative_path: str, patterns: List[str]) -> bool:
    path = relative_path.replace("\\", "/")
    name = Path(path).name

    for pattern in patterns:
        normalized = pattern.rstrip("/")
        if normalized.startswith("/"):
            if fnmatch.fnmatch(path, normalized[1:]) or fnmatch.fnmatch(path, normalized[1:] + "/*"):
                return True
            continue
        if fnmatch.fnmatch(name, normalized) or fnmatch.fnmatch(path, normalized):
            return True
        if fnmatch.fnmatch(path, f"*/{normalized}"):
            return True
    return False


def _should_skip_path(path: Path, root: Path, gitignore_patterns: List[str]) -> bool:
    relative = path.relative_to(root).as_posix()

    for part in path.relative_to(root).parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True

    for pattern in DEFAULT_IGNORE_GLOBS:
        if fnmatch.fnmatch(path.name, pattern):
            return True

    if _matches_gitignore(relative, gitignore_patterns):
        return True

    return False


def _is_text_extension(path: Path, extensions: Optional[List[str]]) -> bool:
    allowed = tuple(extensions) if extensions else DEFAULT_TEXT_EXTENSIONS
    return path.suffix.lower() in allowed


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def collect_repo_files(
    root_path: str,
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Directory not found: {root_path}")

    gitignore_patterns = _load_gitignore_patterns(root)
    files: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        dirnames[:] = [
            name
            for name in dirnames
            if not _should_skip_path(current / name, root, gitignore_patterns)
        ]

        for filename in filenames:
            file_path = current / filename
            if _should_skip_path(file_path, root, gitignore_patterns):
                continue
            if not _is_text_extension(file_path, extensions):
                continue
            if file_path.stat().st_size > settings.INGEST_MAX_FILE_BYTES:
                continue
            files.append(file_path)

    return sorted(files)


def build_chunks_from_file(path: Path, root: Optional[Path] = None) -> List[str]:
    content = _read_text_file(path)
    if not content:
        return []

    label = path.name if root is None else path.relative_to(root).as_posix()
    header = f"Source: {label}\n\n"
    return [
        f"{header}{chunk}"
        for chunk in chunk_text(
            content,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
        )
    ]


def build_chunks_from_text(content: str) -> List[str]:
    return chunk_text(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
