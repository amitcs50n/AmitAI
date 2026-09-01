"""Shared private key-store and ephemeral runtime-token path handling."""

from __future__ import annotations

import os
import secrets
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

MAX_PRIVATE_FILE_BYTES = 64 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PrivatePathError(RuntimeError):
    """A sanitized private-path or permission failure."""


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def default_key_file(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "win32":
        root = values.get("LOCALAPPDATA")
        if not root:
            raise PrivatePathError("LOCALAPPDATA is required for secure key storage")
        return _resolved(Path(root) / "AmitAI" / "secrets" / "database-key.json")
    if selected_platform == "darwin":
        return _resolved(
            Path.home()
            / "Library"
            / "Application Support"
            / "AmitAI"
            / "secrets"
            / "database-key.json"
        )
    data_home = values.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return _resolved(root / "amitai" / "secrets" / "database-key.json")


def default_runtime_token_file(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    override = values.get("AMITAI_LOCAL_API_TOKEN_FILE")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            raise PrivatePathError("Runtime token file override must be absolute")
        return _resolved(override_path)
    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "win32":
        root = values.get("LOCALAPPDATA")
        if not root:
            raise PrivatePathError("LOCALAPPDATA is required for secure runtime storage")
        return _resolved(Path(root) / "AmitAI" / "runtime" / "local-api-token")
    if selected_platform == "darwin":
        return _resolved(
            Path.home()
            / "Library"
            / "Application Support"
            / "AmitAI"
            / "runtime"
            / "local-api-token"
        )
    runtime_home = values.get("XDG_RUNTIME_DIR")
    if runtime_home:
        return _resolved(Path(runtime_home) / "amitai" / "local-api-token")
    state_home = values.get("XDG_STATE_HOME")
    root = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return _resolved(root / "amitai" / "runtime" / "local-api-token")


def rotation_journal_path(key_file: Path) -> Path:
    return key_file.with_name(f"{key_file.name}.rotation")


def rotation_candidate_path(database_path: Path) -> Path:
    return database_path.with_name(f".{database_path.name}.amitai-rotating")


def validate_key_file_path(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> Path:
    candidate = _resolved(path)
    repository = _resolved(repository_root)
    if candidate == repository or repository in candidate.parents:
        raise PrivatePathError("Wrapped key file must be outside the AmitAI repository")

    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "win32":
        values = os.environ if environ is None else environ
        for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            root = values.get(name)
            if not root:
                continue
            one_drive_root = _resolved(Path(root))
            if candidate == one_drive_root or one_drive_root in candidate.parents:
                raise PrivatePathError("Wrapped key file must not be stored in OneDrive")
    return candidate


def _windows_security_attributes():
    try:
        import pywintypes
        import win32api
        import win32con
        import win32security
    except ImportError:
        raise PrivatePathError("Windows owner-only ACL support is unavailable") from None

    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32con.TOKEN_QUERY,
    )
    user_sid = win32security.GetTokenInformation(
        token,
        win32security.TokenUser,
    )[0]
    sid_text = win32security.ConvertSidToStringSid(user_sid)
    descriptor = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
        f"D:P(A;;FA;;;{sid_text})",
        win32security.SDDL_REVISION_1,
    )
    attributes = pywintypes.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = descriptor
    return attributes


def _assert_windows_owner_only(path: Path) -> None:
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security
    except ImportError:
        raise PrivatePathError("Windows owner-only ACL support is unavailable") from None

    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32con.TOKEN_QUERY,
    )
    user_sid = win32security.GetTokenInformation(
        token,
        win32security.TokenUser,
    )[0]
    descriptor = win32security.GetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION,
    )
    if descriptor.GetSecurityDescriptorOwner() != user_sid:
        raise PrivatePathError("Private path is not owned by the current user")
    control, _ = descriptor.GetSecurityDescriptorControl()
    if not control & win32security.SE_DACL_PROTECTED:
        raise PrivatePathError("Private path ACL inheritance is not disabled")
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None or dacl.GetAceCount() != 1:
        raise PrivatePathError("Private path does not have an owner-only ACL")
    ace = dacl.GetAce(0)
    ace_type = ace[0][0]
    mask = ace[1]
    sid = ace[2]
    if (
        ace_type != win32security.ACCESS_ALLOWED_ACE_TYPE
        or sid != user_sid
        or mask & ntsecuritycon.FILE_ALL_ACCESS != ntsecuritycon.FILE_ALL_ACCESS
    ):
        raise PrivatePathError("Private path grants access beyond the current user")


def _assert_posix_owner_only(path: Path, *, directory: bool) -> None:
    details = path.stat()
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        raise PrivatePathError("Private path is not owned by the current user")
    expected = 0o700 if directory else 0o600
    if stat.S_IMODE(details.st_mode) != expected:
        raise PrivatePathError("Private path permissions are not owner-only")


def assert_owner_only(path: Path, *, directory: bool) -> None:
    try:
        if os.name == "nt":
            _assert_windows_owner_only(path)
        else:
            _assert_posix_owner_only(path, directory=directory)
    except PrivatePathError:
        raise
    except OSError:
        raise PrivatePathError("Private path permissions could not be verified") from None
    except Exception:  # noqa: BLE001 - platform security APIs vary by provider
        raise PrivatePathError("Private path permissions could not be verified") from None


def ensure_private_directory(path: Path) -> None:
    target = _resolved(path)
    if target.exists():
        if not target.is_dir():
            raise PrivatePathError("Private directory path is invalid")
        assert_owner_only(target, directory=True)
        return

    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    try:
        for directory in reversed(missing):
            if os.name == "nt":
                import win32file

                win32file.CreateDirectory(
                    str(directory),
                    _windows_security_attributes(),
                )
            else:
                os.mkdir(directory, 0o700)
            assert_owner_only(directory, directory=True)
    except Exception:  # noqa: BLE001 - normalize platform-specific failures
        raise PrivatePathError("Private directory could not be created securely") from None


def _write_windows_private_file(path: Path, data: bytes) -> None:
    try:
        import win32con
        import win32file

        handle = win32file.CreateFile(
            str(path),
            win32con.GENERIC_WRITE,
            0,
            _windows_security_attributes(),
            win32con.CREATE_NEW,
            win32con.FILE_ATTRIBUTE_NORMAL,
            None,
        )
        try:
            win32file.WriteFile(handle, data)
            win32file.FlushFileBuffers(handle)
        finally:
            handle.Close()
    except PrivatePathError:
        raise
    except Exception:  # noqa: BLE001 - pywin32 exposes multiple exception types
        raise PrivatePathError("Private file could not be written securely") from None


def _write_posix_private_file(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_private(path: Path, data: bytes) -> None:
    if len(data) > MAX_PRIVATE_FILE_BYTES:
        raise PrivatePathError("Private file content is too large")
    target = _resolved(path)
    ensure_private_directory(target.parent)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise PrivatePathError("Private file path is invalid")
        assert_owner_only(target, directory=False)
    candidate = target.with_name(f".{target.name}.tmp-{secrets.token_hex(8)}")
    try:
        if os.name == "nt":
            _write_windows_private_file(candidate, data)
        else:
            _write_posix_private_file(candidate, data)
        assert_owner_only(candidate, directory=False)
        os.replace(candidate, target)
        assert_owner_only(target, directory=False)
        _fsync_directory(target.parent)
    except PrivatePathError:
        try:
            candidate.unlink()
        except OSError:
            pass
        raise
    except OSError:
        try:
            candidate.unlink()
        except OSError:
            pass
        raise PrivatePathError("Private file could not be written securely") from None


def read_private(path: Path) -> bytes:
    target = _resolved(path)
    assert_owner_only(target.parent, directory=True)
    if target.is_symlink() or not target.is_file():
        raise PrivatePathError("Private file path is invalid")
    assert_owner_only(target, directory=False)
    try:
        size = target.stat().st_size
        if size > MAX_PRIVATE_FILE_BYTES:
            raise PrivatePathError("Private file content is too large")
        return target.read_bytes()
    except PrivatePathError:
        raise
    except OSError:
        raise PrivatePathError("Private file could not be read") from None


def remove_private(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        raise PrivatePathError("Private file could not be removed") from None
