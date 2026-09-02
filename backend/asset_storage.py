"""UUID-only, owner-only ciphertext store. Plaintext exists only in memory."""

from __future__ import annotations

import errno
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from .asset_crypto import (
    ASSET_MAGIC,
    MAX_ASSET_BYTES,
    MAX_ASSET_CIPHERTEXT_BYTES,
    PNG_SIGNATURE,
    decrypt_asset,
    encrypt_asset,
)
from .secure_memory import SecretHandle

__all__ = ["MAX_ASSET_BYTES", "AssetStorage", "AssetStorageError", "default_asset_directory"]

ASSET_ID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_OWNED_FILE = re.compile(
    r"(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.(?:png|asset)|"
    r"\.[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.(?:png|asset)\.tmp-[0-9a-f]{16})\Z"
)


class AssetStorageError(RuntimeError):
    """No filesystem details may leave this boundary."""


P = ParamSpec("P")
T = TypeVar("T")


def _safe_storage(method: Callable[P, T]) -> Callable[P, T]:
    @wraps(method)
    def safe(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - filesystem and ACL errors contain paths
            raise AssetStorageError("Local asset storage is unavailable") from None

    return safe


def default_asset_directory(namespace: str) -> Path:
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            raise AssetStorageError("Private asset storage is unavailable")
        base = Path(root) / "AmitAI"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "AmitAI"
    else:
        base = (
            Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "amitai"
        )
    return base / "assets" / namespace


def _windows_permissions():
    import pywintypes
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    sid_text = win32security.ConvertSidToStringSid(sid)
    attributes = pywintypes.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = (
        win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
            f"D:P(A;;FA;;;{sid_text})",
            win32security.SDDL_REVISION_1,
        )
    )
    return attributes, sid


def _assert_private(path: Path, *, directory: bool) -> None:
    if os.name == "nt":
        import win32security

        _, owner = _windows_permissions()
        descriptor = win32security.GetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
        )
        acl = descriptor.GetSecurityDescriptorDacl()
        if (
            descriptor.GetSecurityDescriptorOwner() != owner
            or acl is None
            or acl.GetAceCount() != 1
            or not descriptor.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_PROTECTED
        ):
            raise AssetStorageError("Asset permissions are not private")
        ace = acl.GetAce(0)
        if ace[0][0] != win32security.ACCESS_ALLOWED_ACE_TYPE or ace[2] != owner:
            raise AssetStorageError("Asset permissions are not private")
    else:
        details = path.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != (
            0o700 if directory else 0o600
        ):
            raise AssetStorageError("Asset permissions are not private")


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        if not path.parent.exists():
            _ensure_directory(path.parent)
        if os.name == "nt":
            import win32file

            win32file.CreateDirectory(str(path), _windows_permissions()[0])
        else:
            path.mkdir(mode=0o700)
    _assert_private(path, directory=True)


def _fsync_directory(path: Path) -> None:
    # Windows has no supported equivalent here; file FlushFileBuffers is mandatory.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
    finally:
        os.close(descriptor)


class AssetStorage:
    def __init__(self, root: Path) -> None:
        # Do not resolve links first: a reparse point must fail, not redirect us.
        self.root = root.absolute()
        self._key: SecretHandle | None = None

    def bind_key(self, key: SecretHandle) -> None:
        if self._key is not None:
            raise AssetStorageError("Asset key is already initialized")
        self._key = key

    def close(self) -> None:
        if self._key is not None:
            self._key.close()
            self._key = None

    def _key_bytes(self) -> bytes:
        if self._key is None:
            raise AssetStorageError("Asset key is unavailable")
        return self._key.copy_bytes()

    def _check_root(self, *, create: bool = False) -> None:
        for path in (self.root, *self.root.parents):
            try:
                details = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(details.st_mode) or (
                getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise AssetStorageError("Asset storage links are not allowed")
        if create:
            _ensure_directory(self.root)
        if self.root.exists():
            _assert_private(self.root, directory=True)

    def _path(self, asset_id: str, *, legacy: bool = False) -> Path:
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise AssetStorageError("Invalid asset ID")
        self._check_root()
        path = self.root / f"{asset_id}.{'png' if legacy else 'asset'}"
        self._check_file(path)
        return path

    @staticmethod
    def _check_file(path: Path) -> None:
        if path.exists() or path.is_symlink():
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or (
                getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise AssetStorageError("Invalid asset storage entry")
            _assert_private(path, directory=False)

    @_safe_storage
    def write(self, asset_id: str, content: bytes) -> None:
        self._check_root(create=True)
        path = self._path(asset_id)
        if path.exists() or self._path(asset_id, legacy=True).exists():
            raise AssetStorageError("Asset already exists")
        encrypted = encrypt_asset(asset_id, content, self._key_bytes())
        self._atomic_ciphertext(path, encrypted)

    def _atomic_ciphertext(
        self,
        path: Path,
        encrypted: bytes,
        *,
        phase_hook: Callable[[str], None] | None = None,
    ) -> None:
        candidate = path.with_name(f".{path.stem}.asset.tmp-{secrets.token_hex(8)}")
        created = False
        try:
            if os.name == "nt":
                import win32con
                import win32file

                handle = win32file.CreateFile(
                    str(candidate),
                    win32con.GENERIC_WRITE,
                    0,
                    _windows_permissions()[0],
                    win32con.CREATE_NEW,
                    win32con.FILE_ATTRIBUTE_NORMAL,
                    None,
                )
                created = True
                try:
                    win32file.WriteFile(handle, encrypted)
                    win32file.FlushFileBuffers(handle)
                finally:
                    handle.Close()
            else:
                descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
                with os.fdopen(descriptor, "wb") as output:
                    output.write(encrypted)
                    output.flush()
                    os.fsync(output.fileno())
            _assert_private(candidate, directory=False)
            if phase_hook is not None:
                phase_hook("after_temp_fsync")
            self._check_root()
            self._check_file(path)
            os.replace(candidate, path)
            _assert_private(path, directory=False)
            _fsync_directory(self.root)
        except Exception:
            # A process crash may leave ciphertext only. Startup removes such temps.
            if created:
                candidate.unlink(missing_ok=True)
            raise

    def _read_bounded(self, path: Path, *, limit: int = MAX_ASSET_CIPHERTEXT_BYTES) -> bytes:
        self._check_root()
        self._check_file(path)
        with path.open("rb") as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise AssetStorageError("Asset too large")
        return data

    @_safe_storage
    def read(self, asset_id: str) -> bytes:
        return decrypt_asset(asset_id, self._read_bounded(self._path(asset_id)), self._key_bytes())

    @_safe_storage
    def delete(self, asset_id: str) -> None:
        self._path(asset_id).unlink(missing_ok=True)
        if self.root.exists():
            _fsync_directory(self.root)

    @_safe_storage
    def owned_files(self) -> list[Path]:
        self._check_root()
        if not self.root.exists():
            return []
        files = sorted(path for path in self.root.iterdir() if _OWNED_FILE.fullmatch(path.name))
        for path in files:
            self._check_file(path)
        return files

    @_safe_storage
    def assert_key_creation_safe(self) -> None:
        for path in self.owned_files():
            # Even partial/unknown .asset artifacts imply a previously established key.
            if ".asset" in path.name:
                raise AssetStorageError("Asset key is unavailable")
            with path.open("rb") as source:
                if source.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                    raise AssetStorageError("Asset key is unavailable")

    @_safe_storage
    def migrate_asset(
        self,
        asset_id: str,
        validate: Callable[[bytes], None],
        *,
        phase_hook: Callable[[str], None] | None = None,
    ) -> None:
        """Offline: replace plaintext at its original name BEFORE the final rename."""
        target = self._path(asset_id)
        legacy = self._path(asset_id, legacy=True)
        if target.exists():
            if legacy.exists():
                raise AssetStorageError("Conflicting asset representations")
            validate(self.read(asset_id))
            return
        if not legacy.exists():
            return  # Missing metadata-backed assets remain inaccessible; never recreate.
        data = self._read_bounded(legacy)
        if data.startswith(ASSET_MAGIC):
            validate(decrypt_asset(asset_id, data, self._key_bytes()))
        elif data.startswith(PNG_SIGNATURE):
            validate(data)
            encrypted = encrypt_asset(asset_id, data, self._key_bytes())
            self._atomic_ciphertext(legacy, encrypted, phase_hook=phase_hook)
            if phase_hook is not None:
                phase_hook("after_legacy_replace")
        else:
            raise AssetStorageError("Unknown asset representation")
        self._check_root()
        self._check_file(legacy)
        if self._path(asset_id).exists():
            raise AssetStorageError("Conflicting asset representations")
        os.replace(legacy, target)
        _assert_private(target, directory=False)
        _fsync_directory(self.root)
        if phase_hook is not None:
            phase_hook("after_legacy_rename")

    @_safe_storage
    def clean_interrupted_temps(self) -> None:
        # Startup only, before serving, with every other backend stopped. No live uploads.
        for path in self.owned_files():
            if path.name.startswith("."):
                path.unlink()
        if self.root.exists():
            _fsync_directory(self.root)

    @_safe_storage
    def clean_orphans(self, active_ids: set[str], *, older_than: float) -> int:
        self._check_root()
        if not self.root.exists():
            return 0
        count = 0
        # Only this store's generated names, flat directory, never recurse/follow.
        for path in self.root.iterdir():
            if not _OWNED_FILE.fullmatch(path.name):
                continue
            if path.stem in active_ids:
                continue
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_mtime >= older_than:
                continue
            if getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            _assert_private(path, directory=False)
            path.unlink()
            count += 1
        _fsync_directory(self.root)
        return count
