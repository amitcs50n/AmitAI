"""Operator convenience only. No model imports, downloads, or application tools."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import io
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

import httpx

from runtime import inference_token
from runtime.inference_auth import validate_inference_token
from runtime.remote_transport import (
    RemoteTransportPolicy,
    create_remote_ssl_context,
    resolve_addresses,
)

ROOT = Path(__file__).resolve().parents[1]


class StartupError(RuntimeError):
    """Operator-safe failure; never include response bodies or credentials."""


def remote_policy(url: str) -> RemoteTransportPolicy:
    return RemoteTransportPolicy.from_config(url, [url])


def public_url(explicit: str | None, pod_id: str | None) -> str:
    if explicit:
        origin = remote_policy(explicit).origin
        if origin.loopback:
            raise StartupError("The public RunPod URL must be an HTTPS hostname origin")
        return origin.url
    if not pod_id or not re.fullmatch(r"[a-zA-Z0-9-]+", pod_id):
        raise StartupError(
            "RUNPOD_POD_ID is missing; pass --public-url with the pod's HTTPS origin"
        )
    return remote_policy(f"https://{pod_id}-8000.proxy.runpod.net").origin.url


def require_free_ports(*ports: int) -> None:
    for port in ports:
        with socket.socket() as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind(("0.0.0.0", port))
            except OSError:
                raise StartupError(
                    f"Port {port} is occupied. Stop the existing service first."
                ) from None


@contextlib.contextmanager
def runpod_lock():
    """Lifetime lock; no token/PID file and no stale-lock deletion or PID guessing."""
    import fcntl
    import stat

    path = Path("/tmp") / f"aevon-inference-{os.getuid()}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise StartupError("Inference launcher lock has unsafe ownership or permissions")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StartupError("An Aevon inference launcher is already running") from None
        yield
    finally:
        os.close(descriptor)


def wait_health(client, alive, timeout: float, stop: threading.Event | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive() or (stop is not None and stop.is_set()):
            raise StartupError("Service exited before startup completed")
        try:
            response = client.get("/health")
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
        except (httpx.HTTPError, ValueError):
            pass
        if stop is not None:
            stop.wait(0.25)
        else:
            time.sleep(0.25)
    raise StartupError("Liveness wait expired; inspect the service log")


def preload_and_wait(client, token: str, alive, timeout: float) -> None:
    headers = {"Authorization": f"Bearer {validate_inference_token(token)}"}
    response = client.post("/preload", headers=headers)
    if response.status_code not in {200, 202}:
        raise StartupError("Preload rejected; check inference configuration")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive():
            raise StartupError("Inference process exited while loading")
        response = client.get("/ready", headers=headers)
        state = response.json().get("state")
        if response.status_code == 200 and state == "ready":
            return
        if response.status_code != 503 or state not in {"unloaded", "loading", "failed"}:
            raise StartupError("Readiness rejected or returned an invalid state")
        if state == "failed":
            raise StartupError("Model initialization failed; inspect the log before retrying")
        time.sleep(1)
    raise StartupError("Model readiness wait expired")


def loopback_client(port: int = 8000):
    return httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False, follow_redirects=False
    )


def private_log(label: str):
    from runtime.paths import atomic_write_private

    # atomic_write_private also creates the parent with the V1 owner-only ACL.
    # tempfile.mkdtemp inherits Windows ACLs that the secure path helper rejects.
    directory = Path(tempfile.gettempdir()) / f"aevon-{label}-{secrets.token_hex(16)}"
    path = directory / "server.log"
    atomic_write_private(path, b"")
    return path, path.open("ab", buffering=0)


def runpod(args) -> None:
    if not sys.platform.startswith("linux"):
        raise StartupError("RunPod startup requires Linux")
    url = public_url(args.public_url, os.environ.get("RUNPOD_POD_ID"))
    with runpod_lock():
        require_free_ports(8000)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            inference_token.main([])
        token = validate_inference_token(captured.getvalue().rstrip("\r\n"))
        environment = {
            **os.environ,
            "HF_HOME": "/workspace/hf",
            "HUGGINGFACE_HUB_CACHE": "/workspace/hf/hub",
            "HF_HUB_CACHE": "/workspace/hf/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": "/tmp",
            "AMITAI_INFERENCE_AUTH_TOKEN": token,
            "AMITAI_ENABLE_DEV_DOCS": "0",
        }
        path, log = private_log("runpod")
        with log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "runtime.inference_app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--workers",
                    "1",
                    "--no-access-log",
                ],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            print(f"Inference PID: {process.pid}; log: {path}", flush=True)
            try:
                with loopback_client() as client:

                    def alive():
                        return process.poll() is None

                    wait_health(client, alive, min(args.timeout, 60))
                    print("Liveness: OK. Loading the shared model over loopback...", flush=True)
                    preload_and_wait(client, token, alive, args.timeout)
                if process.poll() is not None:
                    raise StartupError("Inference process exited before startup completed")
                print(f"READY\nPublic URL: {url}\nInference token: {token}", flush=True)
                print(
                    "Keep this terminal open. Ctrl+C stops this inference process safely.",
                    flush=True,
                )
                if process.wait() != 0:
                    raise StartupError("Inference process exited; inspect its log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    print(
                        "Waiting for inference shutdown; an active model load cannot be forced safely.",
                        flush=True,
                    )
                    while True:
                        try:
                            process.wait()
                            break
                        except KeyboardInterrupt:
                            print("Still waiting for graceful inference shutdown.", flush=True)


def frontend_environment(environment: dict[str, str]) -> dict[str, str]:
    result = {
        key: value
        for key, value in environment.items()
        if not key.startswith("AMITAI_")
        and key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}
    }
    result["AMITAI_API_ORIGIN"] = "http://127.0.0.1:8000"
    if environment.get("AMITAI_LOCAL_API_TOKEN_FILE"):
        result["AMITAI_LOCAL_API_TOKEN_FILE"] = environment["AMITAI_LOCAL_API_TOKEN_FILE"]
    return result


class WindowsFrontend:
    """Hidden Node process in an owned kill-on-close job, including Next's workers."""

    def __init__(self, command, environment, log):
        import msvcrt

        import win32api
        import win32con
        import win32job
        import win32process

        self.job = win32job.CreateJobObject(None, "")
        self.process = None
        try:
            settings = win32job.QueryInformationJobObject(
                self.job, win32job.JobObjectExtendedLimitInformation
            )
            settings["BasicLimitInformation"]["LimitFlags"] = (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                self.job, win32job.JobObjectExtendedLimitInformation, settings
            )
        except BaseException:
            self.close()
            raise
        startup = win32process.STARTUPINFO()
        startup.dwFlags = win32con.STARTF_USESTDHANDLES | win32con.STARTF_USESHOWWINDOW
        startup.wShowWindow = win32con.SW_HIDE
        with open(os.devnull, "rb") as null:
            handles = [msvcrt.get_osfhandle(null.fileno()), msvcrt.get_osfhandle(log.fileno())]
            startup.hStdInput, startup.hStdOutput = handles
            startup.hStdError = handles[1]
            thread = None
            try:
                for handle in handles:
                    os.set_handle_inheritable(handle, True)
                self.process, thread, self.pid, _ = win32process.CreateProcess(
                    command[0],
                    subprocess.list2cmdline(command),
                    None,
                    None,
                    True,
                    win32con.CREATE_SUSPENDED | win32con.CREATE_NO_WINDOW,
                    environment,
                    str(ROOT / "frontend"),
                    startup,
                )
                win32job.AssignProcessToJobObject(self.job, self.process)
                win32process.ResumeThread(thread)
            except BaseException:
                if self.process is not None:
                    win32process.TerminateProcess(
                        self.process, 1
                    )  # Still suspended if assignment failed.
                self.close()
                raise
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
                if thread is not None:
                    win32api.CloseHandle(thread)

    def poll(self):
        import win32con
        import win32process

        code = win32process.GetExitCodeProcess(self.process)
        return None if code == win32con.STILL_ACTIVE else code

    def close(self):
        import win32api
        import win32event

        if self.job is not None:
            win32api.CloseHandle(self.job)
            self.job = None
        if self.process is not None:
            win32event.WaitForSingleObject(self.process, 15000)
            win32api.CloseHandle(self.process)
            self.process = None


def windows(args) -> None:
    if os.name != "nt":
        raise StartupError("Windows startup requires Windows")
    require_free_ports(8000, 3000)
    node = shutil.which("node")
    next_cli = ROOT / "frontend/node_modules/next/dist/bin/next"
    if not node or not next_cli.is_file():
        raise StartupError(
            "Node.js and the existing frontend dependencies are required (npm ci in frontend)"
        )
    policy = remote_policy(input("RunPod inference URL: ").strip())
    token = validate_inference_token(getpass.getpass("RunPod inference token (hidden): "))
    policy.validate_dns(resolve_addresses)
    with httpx.Client(
        verify=create_remote_ssl_context(), timeout=10, trust_env=False, follow_redirects=False
    ) as client:
        response = client.get(
            policy.origin.url + "/ready", headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code != 200 or response.json() != {"state": "ready"}:
            raise StartupError(
                "Remote inference is not ready or authentication was rejected; run the pod launcher first"
            )

    previous = dict(os.environ)
    old_cwd = Path.cwd()
    stop = threading.Event()
    failures = []
    frontend = []
    path, log = private_log("frontend")
    os.environ.update(
        {
            "AMITAI_INFERENCE_PROVIDER": "remote",
            "AMITAI_REMOTE_INFERENCE_URL": policy.origin.url,
            "AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS": policy.origin.url,
            "AMITAI_REMOTE_INFERENCE_TOKEN": token,
            "AMITAI_HOST": "127.0.0.1",
            "AMITAI_PORT": "8000",
            "AMITAI_ALLOW_LAN": "0",
            "AMITAI_ENABLE_DEV_DOCS": "0",
        }
    )
    os.environ.pop("AMITAI_ENCRYPT_EXISTING_DB", None)

    def start_ui():
        try:
            with loopback_client() as client:
                # The local backend uses /api/health, not the inference /health route.
                deadline = time.monotonic() + args.timeout
                while not stop.is_set():
                    try:
                        response = client.get("/api/health")
                        if response.status_code == 200 and response.json() == {"status": "ok"}:
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise StartupError("Backend startup/unlock wait expired")
                    stop.wait(0.25)
            if stop.is_set():
                return
            child = WindowsFrontend(
                [node, str(next_cli), "dev", "-H", "127.0.0.1", "-p", "3000"],
                frontend_environment(dict(os.environ)),
                log,
            )
            frontend.append(child)
            with loopback_client(3000) as client:
                deadline = time.monotonic() + 120
                while not stop.is_set():
                    if child.poll() is not None:
                        raise StartupError("Frontend exited; inspect its log")
                    try:
                        if client.get("/").status_code == 200:
                            print(
                                f"Aevon UI: http://127.0.0.1:3000 (frontend log: {path})",
                                flush=True,
                            )
                            webbrowser.open("http://127.0.0.1:3000")
                            return
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise StartupError("Frontend startup wait expired; inspect its log")
                    stop.wait(0.5)
        except Exception:  # noqa: BLE001 - do not print dependency/HTTP errors containing secrets.
            failures.append(True)
            print(
                f"UI startup failed. Frontend log: {path}. Stop with Ctrl+C and inspect setup.",
                flush=True,
            )

    worker = threading.Thread(target=start_ui, name="aevon-ui-startup")
    try:
        os.chdir(ROOT)
        worker.start()
        print(
            "Unlock the existing encrypted database below. Ctrl+C stops backend and frontend.",
            flush=True,
        )
        from runtime.serve import main as serve

        serve([])
    finally:
        stop.set()
        worker.join()
        for child in frontend:
            child.close()
        log.close()
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(previous)
    if failures:
        raise StartupError("Frontend startup did not complete")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the existing Aevon runtime with session-only credentials"
    )
    parser.add_argument("platform", choices=("runpod", "windows"))
    parser.add_argument(
        "--public-url", help="RunPod HTTPS origin when RUNPOD_POD_ID is unavailable"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="Startup/readiness wait in seconds (default: 900)"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    previous = {}
    if args.platform == "runpod":

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                previous[signum] = signal.signal(signum, interrupted)
    try:
        (runpod if args.platform == "runpod" else windows)(args)
        return 0
    except KeyboardInterrupt:
        print("Aevon stopped.")
        return 0
    except StartupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - the operator boundary must redact unknown exception text.
        print(
            "Startup failed. Check dependencies, configuration and the printed service log.",
            file=sys.stderr,
        )
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
