"""Untrusted archive rejection before extraction, prompting, or installation."""

import hashlib
import io
import json
import stat
import struct
import warnings
import zipfile
from uuid import uuid4

import pytest

from runtime import backup, backup_archive
from runtime.backup_archive import DATABASE, ENVELOPE, FIXED_DATE, MANIFEST, Entry, Manifest
from runtime.backup_files import copy_and_hash, new_private_file
from runtime.key_store import KeyStore, KeyStorePolicy


def zip_bytes(values, mutate=None):
    output = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(output, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)  # intentionally duplicate hostile entries
        for name, content in values:
            info = zipfile.ZipInfo(name, FIXED_DATE)
            info.create_system, info.external_attr = 0, 0x20
            if mutate:
                mutate(info)
            archive.writestr(info, content)
    return output.getvalue()


@pytest.fixture
def container(tmp_path):
    values = {
        DATABASE: b"D" * 100,
        ENVELOPE: b'{"test":"envelope"}',
        f"assets/{uuid4()}.asset": b"A" * 100,
    }
    manifest = Manifest(
        str(uuid4()),
        {
            name: Entry(len(value), hashlib.sha256(value).hexdigest())
            for name, value in values.items()
        },
    )
    values[MANIFEST] = manifest.encode()
    path = tmp_path / "private" / "untrusted.amitai-backup"
    new_private_file(path, zip_bytes(values.items()))
    with backup_archive.validated_archive(path) as (_, parsed):
        assert parsed == manifest
    return path, values


def rejected_before_extraction(container, monkeypatch):
    path, _ = container
    target = path.parent / "new" / "restored.db"
    store = KeyStore(
        path.parent / "secrets" / "database-key.json", policy=KeyStorePolicy.for_tests()
    )
    assets = path.parent / "new-assets"
    monkeypatch.setattr(backup, "default_asset_directory", lambda _: assets)
    monkeypatch.setattr(
        backup, "extract_verified", lambda *_: pytest.fail("Extracted untrusted archive")
    )
    with pytest.raises(backup.BackupError) as error:
        backup.restore_backup(
            path, target, store, prompt=lambda _: pytest.fail("Prompted before rejection")
        )
    assert (
        str(error.value) == "Restore failed; if interrupted, retry the same command with --resume"
    )
    assert not target.exists() and not store.key_file.exists() and not assets.exists()
    assert not list(path.parent.rglob(".amitai-backup-*"))


@pytest.mark.parametrize(
    "name",
    [
        "unexpected.txt",
        "../outside.asset",
        "assets/../outside.asset",
        "/absolute.asset",
        "C:/absolute.asset",
        "C:\\absolute.asset",
        "\\\\server\\share\\secret",
        "assets/",
        "assets/not-a-uuid.asset",
        "assets/00000000-0000-0000-0000-00000000000A.asset",
        "assets/00000000-0000-0000-0000-000000000001.png",
        "./database.bin",
        "assets/00000000-0000-0000-0000-000000000001.asset/extra",
        "assets/00000000-0000-0000-0000-000000000001.asset\x00ignored",
        DATABASE,
        MANIFEST,
    ],
)
def test_reject_unknown_unsafe_plaintext_and_duplicate_members(container, monkeypatch, name):
    path, values = container
    path.write_bytes(zip_bytes([*values.items(), (name, b"UNTRUSTED_RAW_CANARY")]))
    rejected_before_extraction(container, monkeypatch)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("external_attr", (stat.S_IFLNK | 0o777) << 16),
        ("create_system", 3),
        ("compress_type", zipfile.ZIP_DEFLATED),
        ("extra", b"\x01\x00\x00\x00"),
        ("comment", b"SOURCE_PATH_CANARY"),
        ("date_time", (2026, 9, 2, 1, 2, 4)),
        ("internal_attr", 1),
        ("create_version", 45),
        ("extract_version", 45),
    ],
)
def test_reject_symlinks_compression_and_noncanonical_metadata(
    container, monkeypatch, attribute, value
):
    path, values = container
    path.write_bytes(zip_bytes(values.items(), lambda info: setattr(info, attribute, value)))
    rejected_before_extraction(container, monkeypatch)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"\xff",
        b"[]",
        b"null",
        b'{"version":1,"version":1}',
        b'{"format":"amitai-backup","__secret__":"RAW_PRIVATE_CANARY"}',
    ],
)
def test_malformed_manifest_rejected(container, monkeypatch, raw):
    path, values = container
    values[MANIFEST] = raw
    path.write_bytes(zip_bytes(values.items()))
    rejected_before_extraction(container, monkeypatch)


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "bool-version",
        "format",
        "id",
        "unknown",
        "negative-size",
        "huge-size",
        "bool-size",
        "hash",
        "entry-extra",
        "asset-extra",
        "count",
        "bool-count",
        "duplicate-id",
    ],
)
def test_manifest_schema_bounds_are_exact(container, monkeypatch, mutation):
    path, values = container
    document = json.loads(values[MANIFEST])
    if mutation in {"version", "bool-version", "format", "id", "unknown"}:
        key, value = {
            "version": ("version", 2),
            "bool-version": ("version", True),
            "format": ("format", "other"),
            "id": ("backup_id", "../escape"),
            "unknown": ("source_path", "PRIVATE_PATH"),
        }[mutation]
        document[key] = value
    elif mutation in {"negative-size", "huge-size", "bool-size"}:
        document["database"]["size"] = {"negative-size": -1, "huge-size": 2**63, "bool-size": True}[
            mutation
        ]
    elif mutation == "hash":
        document["database"]["sha256"] = "A" * 64
    elif mutation == "entry-extra":
        document["key_envelope"]["extra"] = "PRIVATE_CANARY"
    elif mutation == "asset-extra":
        document["assets"][0]["filename"] = "PRIVATE_CLIENT_FILENAME"
    elif mutation == "count":
        document["asset_count"] = 10_001
    elif mutation == "bool-count":
        document["asset_count"] = True
    else:
        document["assets"].append(document["assets"][0])
        document["asset_count"] += 1
    values[MANIFEST] = json.dumps(document).encode()
    path.write_bytes(zip_bytes(values.items()))
    rejected_before_extraction(container, monkeypatch)


@pytest.mark.parametrize(
    "limit",
    [
        "MAX_ARCHIVE_BYTES",
        "MAX_DATABASE_BYTES",
        "MAX_ENVELOPE_BYTES",
        "MAX_MANIFEST_BYTES",
        "MAX_ASSET_CIPHERTEXT_BYTES",
        "MAX_ASSETS",
        "MAX_MEMBERS",
        "MAX_DIRECTORY_BYTES",
    ],
)
def test_archive_limits_checked_before_extraction(container, monkeypatch, limit):
    monkeypatch.setattr(backup_archive, limit, 0 if limit == "MAX_ASSETS" else 1)
    rejected_before_extraction(container, monkeypatch)


def test_manifest_aggregate_size_is_bounded(container, monkeypatch):
    _, values = container
    monkeypatch.setattr(backup_archive, "MAX_ARCHIVE_BYTES", 150)
    with pytest.raises(ValueError, match="exceeds V1"):
        backup_archive.parse_manifest(values[MANIFEST])


@pytest.mark.parametrize(
    "mutation",
    ["trailing", "preamble", "truncated", "header", "crc", "count", "directory", "split-entry"],
)
def test_container_layout_is_strict(container, monkeypatch, mutation):
    path, _ = container
    data = bytearray(path.read_bytes())
    if mutation == "trailing":
        data += b"RAW_CANARY"
    elif mutation == "preamble":
        data[:0] = b"PREAMBLE"
    elif mutation == "truncated":
        del data[-30:]
    elif mutation == "header":
        data[6] = 8  # data descriptor flag in local header only
    elif mutation == "crc":
        data[14] ^= 1
    elif mutation == "count":
        struct.pack_into("<H", data, len(data) - 12, 65_535)
    elif mutation == "split-entry":
        struct.pack_into("<H", data, data.index(b"PK\x01\x02") + 34, 1)
    else:
        struct.pack_into("<I", data, len(data) - 10, 2**32 - 1)
    path.write_bytes(data)
    rejected_before_extraction(container, monkeypatch)


def test_zip64_locator_rejected_before_zipfile_can_allocate(container, monkeypatch):
    path, _ = container
    data = bytearray(path.read_bytes())
    data[-42:-38] = b"PK\x06\x07"
    path.write_bytes(data)
    monkeypatch.setattr(
        backup_archive.zipfile, "ZipFile", lambda *_a, **_k: pytest.fail("Parsed ZIP64")
    )
    rejected_before_extraction(container, monkeypatch)


def test_bounded_copy_never_writes_over_limit():
    destination = io.BytesIO()
    with pytest.raises(ValueError, match="exceeds"):
        copy_and_hash(io.BytesIO(b"A" * 101), destination, 100)
    assert len(destination.getvalue()) <= 100


def test_staging_checks_archive_again_if_ciphertext_changes(container, tmp_path):
    path, _ = container
    with backup_archive.validated_archive(path) as (archive, manifest):
        # Even a forged reader after validation must not install changed bytes.
        archive.open = lambda *_: io.BytesIO(b"changed payload")
        with pytest.raises(ValueError, match="changed during staging"):
            backup_archive.extract_verified(archive, manifest, tmp_path / "stage")
