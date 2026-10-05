"""Validate untrusted archive names before copying ordinary file contents."""

import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

_WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
}


def _safe_component(component: str) -> bool:
    """Check portable names, including Windows device and stream aliases."""
    return (
        bool(component)
        and component not in (".", "..")
        and not component.endswith((".", " "))
        and not any(ord(char) < 32 or char in '<>:"\\|?*\x7f' for char in component)
        and component.split(".", 1)[0].rstrip(" ").upper() not in _WINDOWS_DEVICES
    )


def validate_remote_filename(filename: str) -> None:
    """Reject server-derived filenames that are paths or filesystem aliases.

    Explicit caller-supplied file names are a separate, trusted path input and
    keep their existing precedence. This check applies only to response headers
    and decoded URL basenames, before creating a download destination.
    """
    if "/" in filename or not _safe_component(filename):
        raise ValueError(f"Unsafe remote filename: {filename!r}")


def _relative_path(name: str, is_directory: bool) -> Path:
    """Validate an archive member without silently normalizing traversal."""
    if is_directory and name in (".", "./"):
        return Path(".")
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part == ".." for part in name.split("/"))
        or (name.endswith("/") and not is_directory)
    ):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    parts = PurePosixPath(name).parts
    if not parts or not all(_safe_component(part) for part in parts):
        raise ValueError(f"Unsafe archive member name: {name!r}")
    return Path(*parts)


def _linked(path: Path) -> bool:
    """Detect symlinks and Windows reparse points on all supported runtimes."""
    status = path.lstat()
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _check_directory_chain(path: Path) -> None:
    """Reject existing links or non-directory ancestors before writes."""
    for current in (path, *path.parents):
        try:
            current.lstat()
        except FileNotFoundError:
            continue
        # lstat also finds a junction whose target does not yet exist.
        if _linked(current) or not current.is_dir():
            raise ValueError(f"Unsafe archive destination directory: {current}")


@dataclass(frozen=True)
class _Member:
    """An ordinary archive entry with a reviewed relative output path."""

    relative: Path
    is_directory: bool
    source: tarfile.TarInfo | zipfile.ZipInfo


def _preflight(members: list[_Member], output: Path, archive_path: Path) -> None:
    """Validate the entire archive and existing output tree before extraction."""
    _check_directory_chain(output)
    root = output.resolve()
    archive = archive_path.resolve()
    names: dict[tuple[str, ...], bool] = {}
    for member in members:
        key = tuple(part.casefold() for part in member.relative.parts)
        if key in names:
            raise ValueError(f"Unsafe duplicate archive member: {member.relative}")
        names[key] = member.is_directory
    for key in names:
        if any(names.get(key[:index]) is False for index in range(1, len(key))):
            raise ValueError("Unsafe archive file/directory collision")
    for member in members:
        target = output / member.relative
        _check_directory_chain(target if member.is_directory else target.parent)
        resolved = target.resolve()
        if not resolved.is_relative_to(root) or resolved == archive:
            raise ValueError(f"Unsafe archive destination: {target}")
        try:
            status = target.lstat()
        except FileNotFoundError:
            continue
        # A dangling Windows junction exists for lstat even when exists() and
        # is_symlink() are both false, including targets within the output root.
        if _linked(target):
            raise ValueError(f"Unsafe linked archive destination: {target}")
        if not member.is_directory and (
            not stat.S_ISREG(status.st_mode) or status.st_nlink > 1
        ):
            raise ValueError(f"Unsafe archive file destination: {target}")


def _tar_members(archive: tarfile.TarFile) -> list[_Member]:
    """Accept TAR directories and regular files, excluding links and specials."""
    members = []
    for item in archive.getmembers():
        directory = item.type == tarfile.DIRTYPE
        if not directory and item.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise ValueError(f"Unsafe TAR member type: {item.name!r}")
        members.append(_Member(_relative_path(item.name, directory), directory, item))
    return members


def _zip_members(archive: zipfile.ZipFile) -> list[_Member]:
    """Reject ZIP links and special UNIX modes instead of copying their payload."""
    members = []
    for item in archive.infolist():
        directory = item.is_dir()
        mode = stat.S_IFMT(item.external_attr >> 16) if item.create_system == 3 else 0
        if mode not in (0, stat.S_IFDIR if directory else stat.S_IFREG):
            raise ValueError(f"Unsafe ZIP member type: {item.filename!r}")
        if item.orig_filename != item.filename:
            raise ValueError(
                f"Unsafe truncated ZIP member name: {item.orig_filename!r}"
            )
        members.append(
            _Member(_relative_path(item.filename, directory), directory, item)
        )
    return members


def extract_archive(archive_path: Path, output: Path) -> list[Path]:
    """Copy reviewed TAR/ZIP files and return their safe relative paths.

    Every member and existing destination is checked before the first write.
    Only directories and regular file bytes are materialized; archive ownership,
    permissions, links and special metadata are deliberately not restored. The
    implementation works on Python 3.10 without extraction-filter support and
    never falls back to ``extractall``. Callers must keep the destination private
    from concurrent filesystem changes; validation is not a filesystem sandbox.

    Args:
        archive_path: The downloaded TAR or ZIP archive to retain during copying.
        output: The caller-selected extraction directory.

    Returns:
        Safe relative paths for all ordinary files, excluding directories.

    Raises:
        ValueError: The archive type, entries or destination are unsafe.
        OSError: A file cannot be read or written; the original archive remains.
    """
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            members = _tar_members(archive)
            _preflight(members, output, archive_path)
            output.mkdir(parents=True, exist_ok=True)
            for member in members:
                target = output / member.relative
                if member.is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Only regular members reached this point, so extractfile cannot
                # dereference an archive link or create a special filesystem node.
                source = archive.extractfile(cast(tarfile.TarInfo, member.source))
                if source is None:
                    raise ValueError(f"Unsafe unreadable TAR member: {member.relative}")
                with source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_archive:
            members = _zip_members(zip_archive)
            _preflight(members, output, archive_path)
            output.mkdir(parents=True, exist_ok=True)
            for member in members:
                target = output / member.relative
                if member.is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_archive.open(
                    cast(zipfile.ZipInfo, member.source)
                ) as zip_source:
                    with target.open("wb") as destination:
                        shutil.copyfileobj(zip_source, destination)
    else:
        raise ValueError(f"Not a valid TAR/ZIP archive: {archive_path}")
    return [member.relative for member in members if not member.is_directory]
