"""Private normalized images addressed exclusively by generated UUIDs.

This separate local store is NOT encrypted by SQLCipher. No browser/model input
can select its root or supply a path. Only a trusted application factory can.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

MAX_ASSET_BYTES = 20 * 1024 * 1024
ASSET_ID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_OWNED_FILE = re.compile(
    r"(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.png|"
    r"\.[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.png\.tmp-[0-9a-f]{16})\Z"
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


class AssetStorage:
    def __init__(self, root: Path) -> None:
        # Do not resolve links first: a reparse point must fail, not redirect us.
        self.root = root.absolute()

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

    def _path(self, asset_id: str) -> Path:
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise AssetStorageError("Invalid asset ID")
        self._check_root()
        path = self.root / f"{asset_id}.png"
        if path.exists() or path.is_symlink():
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or (
                getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise AssetStorageError("Invalid asset storage entry")
        return path

    @_safe_storage
    def write(self, asset_id: str, content: bytes) -> None:
        self._check_root(create=True)
        path = self._path(asset_id)
        if path.exists():
            raise AssetStorageError("Asset already exists")
        candidate = path.with_name(f".{path.name}.tmp-{secrets.token_hex(8)}")
        try:
            if len(content) > MAX_ASSET_BYTES:
                raise AssetStorageError("Asset too large")
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
                try:
                    win32file.WriteFile(handle, content)
                    win32file.FlushFileBuffers(handle)
                finally:
                    handle.Close()
            else:
                descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
            _assert_private(candidate, directory=False)
            os.replace(candidate, path)
        finally:
            candidate.unlink(missing_ok=True)

    @_safe_storage
    def read(self, asset_id: str) -> bytes:
        path = self._path(asset_id)
        _assert_private(path, directory=False)
        with path.open("rb") as source:
            data = source.read(MAX_ASSET_BYTES + 1)
        if len(data) > MAX_ASSET_BYTES:
            raise AssetStorageError("Asset too large")
        return data

    @_safe_storage
    def delete(self, asset_id: str) -> None:
        self._path(asset_id).unlink(missing_ok=True)

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
            if path.name.removesuffix(".png") in active_ids:
                continue
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_mtime >= older_than:
                continue
            if getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            path.unlink()
            count += 1
        return count
