"""Request-local authorization for one explicitly selected remote vision image."""

from dataclasses import dataclass, field
from threading import Event
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, repr=False)
class RemoteVisionGrant:
    asset_id: str
    explicit_consent: bool
    purpose: Literal["vision"] = "vision"
    cancel_event: Event | None = field(default=None, compare=False)
    _revoked: Event = field(default_factory=Event, init=False, compare=False)

    def __post_init__(self) -> None:
        if str(UUID(self.asset_id)) != self.asset_id:
            raise ValueError("Invalid vision grant")
        self.require(self.asset_id)

    def require(self, asset_id: str | None = None) -> None:
        if (
            self.explicit_consent is not True
            or self.purpose != "vision"
            or self._revoked.is_set()
            or (self.cancel_event is not None and self.cancel_event.is_set())
            or (asset_id is not None and asset_id != self.asset_id)
        ):
            raise PermissionError("Remote vision requires current request consent")

    def revoke(self) -> None:
        self._revoked.set()


def require_remote_vision_grant(
    grant: RemoteVisionGrant | None, asset_id: str | None = None
) -> None:
    if not isinstance(grant, RemoteVisionGrant):
        raise PermissionError("Remote vision requires current request consent")
    grant.require(asset_id)
