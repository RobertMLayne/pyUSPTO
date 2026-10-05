"""Offline regressions for archive and remote filename trust boundaries."""

import io
import os
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from pyUSPTO.clients.base import BaseUSPTOClient
from pyUSPTO.http_config import HTTPConfig


def _client() -> BaseUSPTOClient:
    """Use only local file methods without creating a session or reading a key."""
    client = BaseUSPTOClient.__new__(BaseUSPTOClient)
    client.http_config = HTTPConfig()
    return client


def _tar(path: Path, members: list[tuple[str, bytes | None]]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


def _zip(path: Path, members: list[tuple[str, bytes | None]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            archive.writestr(name, data or b"")
    return path


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_traversal_rejected_before_any_member_is_written(
    tmp_path: Path, kind: str
) -> None:
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}",
        [("safe.txt", b"safe"), ("../sentinel.txt", b"escaped")],
    )
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, output, remove_archive=True)
    assert sentinel.read_bytes() == b"unchanged"
    assert not (output / "safe.txt").exists()
    assert archive.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize(
    "name",
    [
        "nested/../../sentinel.txt",
        "C:/escape.txt",
        "C:escape.txt",
        r"..\sentinel.txt",
        r"\\server\share\file.txt",
        "safe:stream.txt",
        "CON.txt",
        "CON .txt",
        "nested/Lpt9.txt",
        "trailing. ",
        "safe/../nested.txt",
    ],
)
def test_nonportable_or_escaping_member_names_are_rejected(
    tmp_path: Path, kind: str, name: str
) -> None:
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [(name, b"bad")]
    )
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, tmp_path / "output")
    assert archive.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_absolute_member_stays_inside_disposable_fixture_on_rejection(
    tmp_path: Path, kind: str
) -> None:
    sentinel = tmp_path / "absolute-sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [(sentinel.as_posix(), b"bad")]
    )
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, tmp_path / "output")
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_safe_nested_single_file_preserves_return_path_and_removes_archive(
    tmp_path: Path, kind: str
) -> None:
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}",
        [("nested/", None), ("./nested/safe.txt", b"expected")],
    )
    output = tmp_path / "output"
    result = _client()._extract_archive(archive, output, remove_archive=True)
    assert result == str(output / "nested" / "safe.txt")
    assert Path(result).read_bytes() == b"expected"
    assert not archive.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize("count", [0, 2])
def test_empty_and_multiple_regular_files_return_directory(
    tmp_path: Path, kind: str, count: int
) -> None:
    members = [(f"nested/{index}.txt", b"content") for index in range(count)]
    archive = (_tar if kind == "tar" else _zip)(tmp_path / f"input.{kind}", members)
    output = tmp_path / "output"
    assert _client()._extract_archive(archive, output) == str(output)
    assert output.is_dir()
    assert archive.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize("names", [("file", "file/child"), ("safe.txt", "SAFE.txt")])
def test_member_collisions_are_preflighted(
    tmp_path: Path, kind: str, names: tuple[str, str]
) -> None:
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [(name, b"content") for name in names]
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_tar_links_and_special_members_are_rejected(
    tmp_path: Path, member_type: bytes
) -> None:
    archive_path = tmp_path / "input.tar"
    with tarfile.open(archive_path, "w") as archive:
        safe = tarfile.TarInfo("safe.txt")
        safe.size = 4
        archive.addfile(safe, io.BytesIO(b"safe"))
        special = tarfile.TarInfo("special")
        special.type = member_type
        special.linkname = "../sentinel.txt"
        archive.addfile(special)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive_path, output, remove_archive=True)
    assert not output.exists()
    assert archive_path.exists()


@pytest.mark.parametrize("member_type", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_zip_links_and_special_members_are_rejected(
    tmp_path: Path, member_type: int
) -> None:
    archive_path = tmp_path / "input.zip"
    info = zipfile.ZipInfo("special")
    info.create_system = 3
    info.external_attr = (member_type | 0o644) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", b"safe")
        archive.writestr(info, b"../sentinel.txt")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive_path, output, remove_archive=True)
    assert not output.exists()
    assert archive_path.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_existing_hardlinked_target_is_not_overwritten(
    tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    os.link(sentinel, output / "target.txt")
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [("target.txt", b"overwrite")]
    )
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, output)
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize("target_exists", [True, False])
def test_existing_directory_link_is_not_followed(
    tmp_path: Path, kind: str, target_exists: bool
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    if target_exists:
        outside.mkdir()
    link = output / "nested"
    if os.name == "nt":
        # Junction creation is local to this disposable fixture and needs no admin.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [("nested/target.txt", b"escaped")]
    )
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, output)
    assert not (outside / "target.txt").exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_dangling_internal_link_target_is_preflighted_before_writes(
    tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    missing_target = output / "missing-target"
    link = output / "linked-target"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(missing_target)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert not link.exists()
        assert not link.is_symlink()
        assert link.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    else:
        link.symlink_to(missing_target, target_is_directory=True)
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}",
        [("safe-early.txt", b"safe"), ("linked-target", b"content")],
    )
    with pytest.raises(ValueError, match="Unsafe linked archive destination"):
        _client()._extract_archive(archive, output, remove_archive=True)
    assert not (output / "safe-early.txt").exists()
    assert not missing_target.exists()
    assert archive.exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_archive_cannot_overwrite_itself(tmp_path: Path, kind: str) -> None:
    archive = (_tar if kind == "tar" else _zip)(
        tmp_path / f"input.{kind}", [(f"input.{kind}", b"overwrite")]
    )
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._extract_archive(archive, tmp_path, remove_archive=True)
    assert archive.read_bytes() == original


def test_safe_extraction_does_not_depend_on_newer_tar_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    monkeypatch.setattr(
        tarfile.TarFile,
        "extractall",
        lambda *args, **kwargs: pytest.fail("Unsafe extraction fallback used"),
    )
    archive = _tar(tmp_path / "input.tar", [("nested/safe.txt", b"safe")])
    result = _client()._extract_archive(archive, tmp_path / "output")
    assert Path(result).read_bytes() == b"safe"


@pytest.mark.parametrize(
    "header,url",
    [
        ('attachment; filename="../sentinel.txt"', "https://example.test/file"),
        ("attachment; filename*=UTF-8''..%2Fsentinel.txt", "https://example.test/file"),
        ("", "https://example.test/..%2Fsentinel.txt"),
        ('attachment; filename="CON.txt"', "https://example.test/file"),
        ('attachment; filename="safe:stream.txt"', "https://example.test/file"),
    ],
)
def test_remote_filename_rejected_before_destination_creation(
    tmp_path: Path, header: str, url: str
) -> None:
    response = MagicMock(spec=requests.Response)
    response.headers = {"Content-Disposition": header}
    response.url = url
    response.iter_content.return_value = [b"overwrite"]
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe"):
        _client()._save_response_to_file(response, str(output), overwrite=True)
    assert sentinel.read_bytes() == b"unchanged"
    assert not output.exists()
    response.iter_content.assert_not_called()


def test_explicit_caller_filename_keeps_precedence_and_relative_paths(
    tmp_path: Path,
) -> None:
    response = MagicMock(spec=requests.Response)
    response.headers = {"Content-Disposition": 'attachment; filename="../bad.txt"'}
    response.url = "https://example.test/..%2Fbad.txt"
    response.iter_content.return_value = [b"expected"]
    (tmp_path / "nested").mkdir()
    result = _client()._save_response_to_file(
        response, str(tmp_path), file_name="nested/chosen.txt"
    )
    assert result == str(tmp_path / "nested" / "chosen.txt")
    assert Path(result).read_bytes() == b"expected"


def test_copy_failure_retains_original_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _tar(tmp_path / "input.tar", [("safe.txt", b"safe")])

    def fail_read(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic read failure")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", fail_read)
    with pytest.raises(OSError, match="synthetic read failure"):
        _client()._extract_archive(archive, tmp_path / "output", remove_archive=True)
    assert archive.exists()
