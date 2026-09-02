"""Strict bounded V1 ciphertext ZIP container. Never extract archive-supplied paths."""

from __future__ import annotations

import json
import os
import re
import struct
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from backend.asset_crypto import ASSET_OVERHEAD, MAX_ASSET_CIPHERTEXT_BYTES
from backend.asset_storage import ASSET_ID_PATTERN

from .backup_files import copy_and_hash, file_fingerprint, regular_file, stage_copy

FORMAT = "amitai-backup"
VERSION = 1
MAX_ASSETS = 10_000
MAX_DATABASE_BYTES = 1024**3
MAX_ENVELOPE_BYTES = 32 * 1024
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_ARCHIVE_BYTES = 2 * 1024**3 - 1
MAX_MEMBERS = MAX_ASSETS + 3
MAX_DIRECTORY_BYTES = MAX_MEMBERS * 128
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
DATABASE = "database.bin"
ENVELOPE = "database-key-envelope.json"
MANIFEST = "manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Entry:
    size: int
    sha256: str

    @property
    def fingerprint(self) -> tuple[int, str]:
        return self.size, self.sha256

    def document(self) -> dict[str, int | str]:
        return {"size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class Manifest:
    backup_id: str
    entries: dict[str, Entry]

    def encode(self) -> bytes:
        assets = [
            {"id": name[7:-6], **entry.document()}
            for name, entry in sorted(self.entries.items())
            if name.startswith("assets/")
        ]
        return json.dumps(
            {
                "format": FORMAT,
                "version": VERSION,
                "backup_id": self.backup_id,
                "database": self.entries[DATABASE].document(),
                "key_envelope": self.entries[ENVELOPE].document(),
                "asset_count": len(assets),
                "assets": assets,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate metadata field")
            result[key] = value
        return result

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise TypeError("Invalid metadata object")
    return result


def canonical_uuid(value: object) -> str:
    if not isinstance(value, str) or ASSET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Invalid backup identifier")
    if str(UUID(value)) != value:
        raise ValueError("Invalid backup identifier")
    return value


def _entry(value: object, maximum: int, minimum: int = 1) -> Entry:
    if not isinstance(value, dict) or set(value) != {"size", "sha256"}:
        raise ValueError("Invalid backup entry")
    size, digest = value["size"], value["sha256"]
    if type(size) is not int or not minimum <= size <= maximum:
        raise ValueError("Invalid backup size")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        raise ValueError("Invalid backup hash")
    return Entry(size, digest)


def parse_manifest(raw: bytes) -> Manifest:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds V1 limits")
    value = strict_json(raw)
    if set(value) != {
        "format",
        "version",
        "backup_id",
        "database",
        "key_envelope",
        "asset_count",
        "assets",
    }:
        raise ValueError("Invalid backup manifest")
    if (
        value["format"] != FORMAT
        or type(value["version"]) is not int
        or value["version"] != VERSION
    ):
        raise ValueError("Unsupported backup format")
    backup_id = canonical_uuid(value["backup_id"])
    assets = value["assets"]
    count = value["asset_count"]
    if (
        type(count) is not int
        or not 0 <= count <= MAX_ASSETS
        or not isinstance(assets, list)
        or count != len(assets)
    ):
        raise ValueError("Invalid backup asset count")
    entries = {
        DATABASE: _entry(value["database"], MAX_DATABASE_BYTES),
        ENVELOPE: _entry(value["key_envelope"], MAX_ENVELOPE_BYTES),
    }
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {"id", "size", "sha256"}:
            raise ValueError("Invalid backup asset")
        name = f"assets/{canonical_uuid(asset['id'])}.asset"
        if name in entries:
            raise ValueError("Duplicate backup asset")
        entries[name] = _entry(
            {"size": asset["size"], "sha256": asset["sha256"]},
            MAX_ASSET_CIPHERTEXT_BYTES,
            ASSET_OVERHEAD + 1,
        )
    if sum(entry.size for entry in entries.values()) > MAX_ARCHIVE_BYTES:
        raise ValueError("Backup exceeds V1 limits")
    return Manifest(backup_id, entries)


def _directory_bounds(source, size: int) -> tuple[int, int]:
    # Bound the central directory BEFORE ZipFile allocates/parses it. V1 excludes
    # ZIP64, split archives, comments, preambles and trailing data entirely.
    if size < 22:
        raise ValueError("Invalid backup container")
    source.seek(size - 22)
    signature, disk, start_disk, disk_count, count, length, offset, comment = struct.unpack(
        "<4s4H2IH", source.read(22)
    )
    if (
        signature != b"PK\x05\x06"
        or disk
        or start_disk
        or comment
        or disk_count != count
        or not 3 <= count <= MAX_MEMBERS
        or not 0 < length <= MAX_DIRECTORY_BYTES
        or offset + length + 22 != size
    ):
        raise ValueError("Invalid backup directory")
    # Python's ZIP reader follows ZIP64 locators even with allowZip64=False
    # (that switch only controls writing). Reject before it can trust ZIP64 sizes.
    if size >= 42:
        source.seek(size - 42)
        if source.read(4) == b"PK\x06\x07":
            raise ValueError("ZIP64 is not supported in backup V1")
    return offset, count


def _layout(source, archive: zipfile.ZipFile, directory: int, count: int) -> None:
    infos = archive.infolist()
    if len(infos) != count or len({info.filename for info in infos}) != count:
        raise ValueError("Duplicate or invalid backup members")
    offset = 0
    for info in infos:
        name = info.filename
        if name not in {MANIFEST, DATABASE, ENVELOPE}:
            if not name.startswith("assets/") or not name.endswith(".asset"):
                raise ValueError("Unexpected backup member")
            canonical_uuid(name[7:-6])
        if (
            info.orig_filename != name
            or info.header_offset != offset
            or info.compress_type != zipfile.ZIP_STORED
            or info.flag_bits != 0
            or info.compress_size != info.file_size
            or info.file_size < 1
            or info.file_size > MAX_DATABASE_BYTES
            or info.create_system != 0
            or info.external_attr != 0x20
            or info.volume != 0
            or info.internal_attr
            or info.extra
            or info.comment
            or info.date_time != FIXED_DATE
            or info.extract_version != 20
            or info.create_version != 20
        ):
            raise ValueError("Unsupported backup member attributes")
        source.seek(offset)
        header = source.read(30)
        signature, version, flags, method, time, date, crc, packed, size, names, extra = (
            struct.unpack("<4s5H3I2H", header)
        )
        if (
            signature != b"PK\x03\x04"
            or version != 20
            or flags
            or method
            or time != 0
            or date != 33
            or crc != info.CRC
            or packed != size
            or size != info.file_size
            or names != len(name)
            or extra
            or source.read(names) != name.encode("ascii")
        ):
            raise ValueError("Invalid backup local header")
        offset += 30 + names + size
    if offset != directory:
        raise ValueError("Invalid backup member layout")


@contextmanager
def validated_archive(path: Path) -> Iterator[tuple[zipfile.ZipFile, Manifest]]:
    size = regular_file(path, MAX_ARCHIVE_BYTES)
    with path.open("rb") as source:
        directory, count = _directory_bounds(source, size)
        with zipfile.ZipFile(source, "r", allowZip64=False) as archive:
            _layout(source, archive, directory, count)
            info = archive.getinfo(MANIFEST)
            if info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("Manifest exceeds V1 limits")
            manifest = parse_manifest(archive.read(info))
            if set(archive.namelist()) != {MANIFEST, *manifest.entries}:
                raise ValueError("Backup members do not match manifest")
            for name, entry in manifest.entries.items():
                if archive.getinfo(name).file_size != entry.size:
                    raise ValueError("Backup member size mismatch")
                with archive.open(name) as member:
                    if copy_and_hash(member, None, entry.size) != entry.fingerprint:
                        raise ValueError("Backup member hash mismatch")
            yield archive, manifest


def extract_verified(archive: zipfile.ZipFile, manifest: Manifest, stage: Path) -> None:
    # Never ZipFile.extract()/extractall(). All names have one exact supported form.
    for name, entry in manifest.entries.items():
        with archive.open(name) as source:
            if stage_copy(source, stage / name, entry.size) != entry.fingerprint:
                raise ValueError("Backup changed during staging")


def write_archive(path: Path, stage: Path, manifest: Manifest) -> None:
    # ZipInfo is deliberately NOT constructed from filesystem paths/stat metadata.
    parse_manifest(manifest.encode())
    with path.open("r+b") as output:
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_STORED, allowZip64=False
        ) as archive:
            for name in [MANIFEST, *sorted(manifest.entries)]:
                info = zipfile.ZipInfo(name, FIXED_DATE)
                info.create_system = 0
                info.external_attr = 0x20  # regular DOS archive file, no source ACL/mode
                if name == MANIFEST:
                    archive.writestr(info, manifest.encode())
                else:
                    entry = manifest.entries[name]
                    info.file_size = entry.size
                    if file_fingerprint(stage / name, entry.size) != entry.fingerprint:
                        raise ValueError("Backup staging changed")
                    with (stage / name).open("rb") as source, archive.open(info, "w") as target:
                        if copy_and_hash(source, target, entry.size) != entry.fingerprint:
                            raise ValueError("Backup staging changed")
        output.flush()
        os.fsync(output.fileno())
