"""Platform-aware process hardening for canonical secure startup."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass


class ProcessHardeningError(RuntimeError):
    """A sanitized mandatory hardening failure."""


@dataclass(frozen=True)
class ProcessHardeningState:
    core_dumps_disabled: bool
    process_dumpable_disabled: bool


def apply_process_hardening() -> ProcessHardeningState:
    """Apply mandatory controls supported by this operating system."""

    if os.name == "nt":
        return ProcessHardeningState(
            core_dumps_disabled=False,
            process_dumpable_disabled=False,
        )

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        raise ProcessHardeningError("Core-dump protection could not be enabled") from None

    dumpable_disabled = False
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = getattr(libc, "prctl", None)
        if prctl is None:
            raise ProcessHardeningError(
                "Linux process-dump protection is unavailable"
            )
        prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        prctl.restype = ctypes.c_int
        if prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE = 4
            raise ProcessHardeningError(
                "Linux process-dump protection could not be enabled"
            )
        dumpable_disabled = True

    return ProcessHardeningState(
        core_dumps_disabled=True,
        process_dumpable_disabled=dumpable_disabled,
    )
