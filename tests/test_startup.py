"""Launcher orchestration tests: fake HTTP/model services and CPU-only child processes."""

import contextlib
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from scripts import startup

TOKEN = "LAUNCHER_TEST_TOKEN_0123456789abcdef"
CANARY = "hf_SECRET_CANARY /workspace/hf/private prompt credential"


def client_for(handler):
    return httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler))


def test_public_url_uses_pod_id_or_canonical_exact_origin():
    assert startup.public_url(None, "pod123") == "https://pod123-8000.proxy.runpod.net"
    assert startup.public_url("https://EXAMPLE.com:443/", None) == "https://example.com"
    with pytest.raises(startup.StartupError):
        startup.public_url(None, None)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://example.com/path",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "http://127.0.0.1:8000",
        "https://127.0.0.1",
    ],
)
def test_public_url_rejects_unsafe_or_nonpublic_origins(url):
    with pytest.raises((ValueError, startup.StartupError)):
        startup.public_url(url, None)


def test_occupied_port_is_refused_without_touching_its_owner():
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen()
        port = owner.getsockname()[1]
        with pytest.raises(startup.StartupError, match="occupied"):
            startup.require_free_ports(port)
        assert owner.getsockname()[1] == port


@pytest.mark.parametrize("states", [["unloaded", "loading", "ready"], ["ready"]])
def test_liveness_then_explicit_preload_and_authenticated_ready(monkeypatch, states):
    requests = []
    remaining = iter(states)

    def handle(request):
        requests.append((request.method, request.url.path))
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/health":
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"status": "ok"})
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        if request.url.path == "/preload":
            assert request.content == b""
            return httpx.Response(202, json={"status": "accepted"})
        state = next(remaining)
        return httpx.Response(200 if state == "ready" else 503, json={"state": state})

    monkeypatch.setattr(startup.time, "sleep", lambda _: None)
    with client_for(handle) as client:
        startup.wait_health(client, lambda: True, 1)
        assert requests == [("GET", "/health")]
        startup.preload_and_wait(client, TOKEN, lambda: True, 1)
    assert requests == [("GET", "/health"), ("POST", "/preload")] + [("GET", "/ready")] * len(
        states
    )


@pytest.mark.parametrize(
    "status,body",
    [
        (503, {"state": "failed", "exception": CANARY}),
        (401, {"detail": CANARY}),
        (200, {"state": "loading"}),
    ],
)
def test_failed_readiness_never_retries_load_or_exposes_response(status, body):
    requests = []

    def handle(request):
        requests.append(request.url.path)
        return (
            httpx.Response(202) if request.method == "POST" else httpx.Response(status, json=body)
        )

    with client_for(handle) as client, pytest.raises(startup.StartupError) as error:
        startup.preload_and_wait(client, TOKEN, lambda: True, 1)
    assert requests == ["/preload", "/ready"]
    assert CANARY not in str(error.value) and TOKEN not in str(error.value)


def test_preload_rejection_does_not_poll_or_print_response():
    with (
        client_for(lambda _: httpx.Response(403, text=CANARY)) as client,
        pytest.raises(startup.StartupError, match="Preload rejected") as error,
    ):
        startup.preload_and_wait(client, TOKEN, lambda: True, 1)
    assert CANARY not in str(error.value)


def test_loading_deadline_and_dead_process_are_bounded(monkeypatch):
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(startup.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(startup.time, "sleep", lambda _: None)
    with (
        client_for(lambda _: httpx.Response(503, json={"state": "loading"})) as client,
        pytest.raises(startup.StartupError, match="Liveness wait expired"),
    ):
        startup.wait_health(client, lambda: True, 1)
    ticks = iter([0, 0, 2])
    with (
        client_for(
            lambda r: (
                httpx.Response(202)
                if r.method == "POST"
                else httpx.Response(503, json={"state": "loading"})
            )
        ) as client,
        pytest.raises(startup.StartupError, match="readiness wait expired"),
    ):
        startup.preload_and_wait(client, TOKEN, lambda: True, 1)
    ticks = iter([0, 0])
    with (
        client_for(lambda _: httpx.Response(202)) as client,
        pytest.raises(startup.StartupError, match="process exited"),
    ):
        startup.preload_and_wait(client, TOKEN, lambda: False, 1)


@pytest.mark.parametrize("failure", [False, True])
def test_runpod_supervises_exactly_one_server_and_only_prints_token_after_ready(
    monkeypatch,
    tmp_path,
    capsys,
    failure,
):
    calls = []
    process = SimpleNamespace(pid=123, running=True)

    def terminate():
        calls.append("terminate")
        process.running = False

    def wait():
        calls.append("wait")
        process.running = False
        return 0

    process.poll = lambda: None if process.running else 0
    process.wait, process.terminate = wait, terminate

    def spawn(command, **kwargs):
        calls.append("spawn")
        assert command[1:] == [
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
        ]
        assert kwargs["start_new_session"] and kwargs["cwd"] == startup.ROOT
        env = kwargs["env"]
        assert env["AMITAI_INFERENCE_AUTH_TOKEN"] == TOKEN
        assert env["HF_HOME"] == "/workspace/hf"
        assert env["HF_HUB_CACHE"] == env["HUGGINGFACE_HUB_CACHE"] == "/workspace/hf/hub"
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["TMPDIR"] == "/tmp"
        return process

    def preload(*args):
        calls.append("preload")
        assert TOKEN not in capsys.readouterr().out
        if failure:
            raise startup.StartupError("Model initialization failed")

    monkeypatch.setattr(startup.sys, "platform", "linux")
    monkeypatch.setenv("RUNPOD_POD_ID", "testpod")
    monkeypatch.setattr(startup, "runpod_lock", contextlib.nullcontext)
    monkeypatch.setattr(startup, "require_free_ports", lambda *args: calls.append("port"))
    monkeypatch.setattr(startup.inference_token, "main", lambda _: print(TOKEN))
    monkeypatch.setattr(startup.subprocess, "Popen", spawn)
    path = tmp_path / "server.log"
    monkeypatch.setattr(startup, "private_log", lambda _: (path, path.open("ab")))
    monkeypatch.setattr(startup, "loopback_client", lambda: contextlib.nullcontext(object()))
    monkeypatch.setattr(startup, "wait_health", lambda *args: calls.append("health"))
    monkeypatch.setattr(startup, "preload_and_wait", preload)
    args = SimpleNamespace(public_url=None, timeout=30)
    if failure:
        with pytest.raises(startup.StartupError):
            startup.runpod(args)
        assert "terminate" in calls
    else:
        startup.runpod(args)
    output = capsys.readouterr().out
    assert (TOKEN in output) is not failure
    assert calls[:4] == ["port", "spawn", "health", "preload"]
    assert calls.count("spawn") == 1 and calls[-1] == "wait"
    assert TOKEN not in path.read_text()


def test_frontend_does_not_inherit_inference_or_database_secrets():
    env = startup.frontend_environment(
        {
            "PATH": "path",
            "AMITAI_REMOTE_INFERENCE_TOKEN": TOKEN,
            "HF_TOKEN": CANARY,
            "AMITAI_INFERENCE_AUTH_TOKEN": TOKEN,
            "AMITAI_DB_KEY": CANARY,
            "AMITAI_ENCRYPT_EXISTING_DB": "1",
            "AMITAI_LOCAL_API_TOKEN_FILE": "private-token-file",
        }
    )
    assert env == {
        "PATH": "path",
        "AMITAI_API_ORIGIN": "http://127.0.0.1:8000",
        "AMITAI_LOCAL_API_TOKEN_FILE": "private-token-file",
    }


@pytest.mark.parametrize("attempt", range(2))
def test_launcher_log_is_owner_only(attempt):
    from runtime.paths import assert_owner_only

    path, log = startup.private_log("cpu-test")
    try:
        with log:
            log.write(b"CPU-only log check\n")
        assert_owner_only(path, directory=False)
        assert_owner_only(path.parent, directory=True)
        assert path.read_text() == "CPU-only log check\n"
        if os.name == "nt":
            assert path.parent.parent == Path(os.environ["LOCALAPPDATA"]) / "AmitAI/runtime"
    finally:
        log.close()
        path.unlink(missing_ok=True)
        path.parent.rmdir()


def test_launcher_log_does_not_bypass_private_path_failure(monkeypatch):
    from runtime import paths

    def denied(*args):
        raise paths.PrivatePathError("ACL denied")

    monkeypatch.setattr(paths, "atomic_write_private", denied)
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("Insecure log fallback"))
    with pytest.raises(paths.PrivatePathError, match="ACL denied"):
        startup.private_log("cpu-test")


@pytest.mark.skipif(os.name != "nt", reason="Windows secure log location")
def test_windows_log_requires_localappdata_without_temp_fallback(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(
        startup.tempfile, "gettempdir", lambda: pytest.fail("Insecure temp fallback")
    )
    with pytest.raises(startup.StartupError, match="LOCALAPPDATA"):
        startup.private_log("cpu-test")


WINDOWS_SECURE_MODULES = (
    "win32api",
    "win32con",
    "win32security",
    "ntsecuritycon",
    "pywintypes",
    "win32job",
    "win32process",
    "win32event",
    "argon2.exceptions",
    "argon2.low_level",
    "sqlcipher3.dbapi2",
)
REPAIR_COMMAND = r'.venv\Scripts\python.exe -m pip install -e ".[secure-runtime,encrypted-storage]"'


@pytest.fixture
def windows_dependencies(monkeypatch):
    # Exercise the real codec verifier with a fake driver; no database file or native DLL required.
    cursor = Mock()
    cursor.execute.return_value.fetchone.return_value = ("4.12.0",)
    connection = Mock()
    connection.cursor.return_value = cursor
    driver = SimpleNamespace(connect=Mock(return_value=connection), Error=sqlite3.Error)
    argon2 = SimpleNamespace(
        hash_secret_raw=Mock(return_value=b"x" * 32),
        Type=SimpleNamespace(ID=2),
        ARGON2_VERSION=19,
    )
    modules = {name: SimpleNamespace() for name in WINDOWS_SECURE_MODULES}
    modules["argon2.low_level"] = argon2
    modules["sqlcipher3.dbapi2"] = driver
    imported = []
    real_import = startup.importlib.import_module

    def load(name, *args, **kwargs):
        if name not in modules:
            return real_import(name, *args, **kwargs)
        imported.append(name)
        module = modules[name]
        if isinstance(module, Exception):
            raise module
        return module

    monkeypatch.setattr(startup.importlib, "import_module", load)
    return SimpleNamespace(
        modules=modules,
        imported=imported,
        argon2=argon2,
        driver=driver,
        cursor=cursor,
        connection=connection,
    )


def test_windows_secure_runtime_preflight_imports_all_required_modules(windows_dependencies):
    startup.require_windows_secure_runtime()
    assert windows_dependencies.imported == list(WINDOWS_SECURE_MODULES)
    windows_dependencies.argon2.hash_secret_raw.assert_called_once()
    windows_dependencies.driver.connect.assert_called_once_with(":memory:")
    windows_dependencies.cursor.execute.assert_called_once_with("PRAGMA cipher_version")
    windows_dependencies.cursor.close.assert_called_once()
    windows_dependencies.connection.close.assert_called_once()


@pytest.mark.parametrize("missing", WINDOWS_SECURE_MODULES)
@pytest.mark.parametrize("error", [ModuleNotFoundError, ImportError, OSError, RuntimeError])
def test_windows_secure_runtime_preflight_has_safe_repair_message(
    windows_dependencies, missing, error
):
    windows_dependencies.modules[missing] = error(CANARY + TOKEN)
    with pytest.raises(startup.StartupError) as failure:
        startup.require_windows_secure_runtime()
    message = str(failure.value)
    assert "Windows secure runtime or encrypted-storage dependencies" in message
    assert REPAIR_COMMAND in message
    assert "--upgrade --force-reinstall pywin32==311" in message
    assert CANARY not in message and TOKEN not in message


@pytest.mark.parametrize(
    "failure", [*WINDOWS_SECURE_MODULES, "argon2-hash", "sqlcipher-connect", "plaintext-driver"]
)
def test_windows_missing_dependencies_stop_before_workflow(
    monkeypatch, capsys, windows_dependencies, failure
):
    if failure == "argon2-hash":
        windows_dependencies.argon2.hash_secret_raw.side_effect = RuntimeError(CANARY + TOKEN)
    elif failure == "sqlcipher-connect":
        windows_dependencies.driver.connect.side_effect = sqlite3.DatabaseError(CANARY + TOKEN)
    elif failure == "plaintext-driver":
        windows_dependencies.modules["sqlcipher3.dbapi2"] = sqlite3
    else:
        windows_dependencies.modules[failure] = ImportError(CANARY + TOKEN)

    def forbidden(*args, **kwargs):
        pytest.fail("Workflow ran before dependency preflight")

    monkeypatch.setattr(startup, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(startup, "require_free_ports", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(startup.getpass, "getpass", forbidden)
    monkeypatch.setattr(startup.httpx, "Client", forbidden)
    monkeypatch.setattr(startup, "resolve_addresses", forbidden)
    monkeypatch.setattr(startup, "private_log", forbidden)
    monkeypatch.setattr(startup.subprocess, "Popen", forbidden)
    monkeypatch.setattr(startup, "WindowsFrontend", forbidden)
    monkeypatch.setattr(startup.threading, "Thread", forbidden)
    previous = dict(os.environ)
    assert startup.main(["windows"]) == 1
    assert dict(os.environ) == previous
    output = capsys.readouterr()
    message = output.out + output.err
    assert "Windows secure runtime or encrypted-storage dependencies" in message
    assert REPAIR_COMMAND in message
    assert CANARY not in message and TOKEN not in message


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_windows_uses_normal_unlock_flow_and_restores_environment(
    monkeypatch, tmp_path, windows_dependencies
):
    import runtime.serve

    opened = threading.Event()
    calls = []
    monkeypatch.setenv("AMITAI_ENCRYPT_EXISTING_DB", "1")
    previous = dict(os.environ)
    old_cwd = Path.cwd()
    monkeypatch.setattr(startup, "require_free_ports", lambda *args: None)
    monkeypatch.setattr(startup.shutil, "which", lambda _: "node.exe")
    monkeypatch.setattr(Path, "is_file", lambda _: True)

    def prompt(_):
        assert windows_dependencies.imported == list(WINDOWS_SECURE_MODULES)
        windows_dependencies.argon2.hash_secret_raw.assert_called_once()
        windows_dependencies.cursor.execute.assert_called_once_with("PRAGMA cipher_version")
        return "https://EXAMPLE.com/"

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(startup.getpass, "getpass", lambda _: TOKEN)
    monkeypatch.setattr(startup, "resolve_addresses", lambda *_: ["93.184.216.34"])
    real_client = httpx.Client

    def remote(request):
        assert request.method == "GET" and str(request.url) == "https://example.com/ready"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        calls.append("remote-ready")
        return httpx.Response(200, json={"state": "ready"})

    def remote_client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(remote))

    monkeypatch.setattr(startup.httpx, "Client", remote_client)
    monkeypatch.setattr(
        startup,
        "loopback_client",
        lambda port=8000: real_client(
            base_url=f"http://127.0.0.1:{port}",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "ok"})),
        ),
    )
    path = tmp_path / "frontend.log"
    monkeypatch.setattr(startup, "private_log", lambda _: (path, path.open("ab")))

    def frontend(command, env, log):
        assert command[-5:] == ["dev", "-H", "127.0.0.1", "-p", "3000"]
        assert "AMITAI_REMOTE_INFERENCE_TOKEN" not in env
        return SimpleNamespace(poll=lambda: None, close=lambda: calls.append("frontend-closed"))

    monkeypatch.setattr(startup, "WindowsFrontend", frontend)
    monkeypatch.setattr(startup.webbrowser, "open", lambda url: opened.set())

    def serve(argv):
        assert argv == [] and Path.cwd() == startup.ROOT
        assert os.environ["AMITAI_INFERENCE_PROVIDER"] == "remote"
        assert os.environ["AMITAI_REMOTE_INFERENCE_URL"] == "https://example.com"
        assert os.environ["AMITAI_REMOTE_INFERENCE_ALLOWED_ORIGINS"] == "https://example.com"
        assert os.environ["AMITAI_REMOTE_INFERENCE_TOKEN"] == TOKEN
        assert "AMITAI_ENCRYPT_EXISTING_DB" not in os.environ
        assert os.environ["AMITAI_HOST"] == "127.0.0.1"
        assert os.environ["AMITAI_ALLOW_LAN"] == "0"
        assert opened.wait(5)
        calls.append("normal-serve")

    monkeypatch.setattr(runtime.serve, "main", serve)
    startup.windows(SimpleNamespace(timeout=5))
    assert calls == ["remote-ready", "normal-serve", "frontend-closed"]
    assert dict(os.environ) == previous and Path.cwd() == old_cwd
    assert TOKEN not in path.read_text()


@pytest.mark.skipif(os.name != "nt", reason="Native Windows job ownership")
def test_windows_job_stops_only_owned_cpu_process_tree(tmp_path):
    import win32api
    import win32con
    import win32event

    child_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with (tmp_path / "cpu.log").open("ab", buffering=0) as log:
        process = startup.WindowsFrontend([sys.executable, "-c", code], dict(os.environ), log)
        child_handle = None
        try:
            deadline = time.monotonic() + 10
            while not child_file.exists() and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.05)
            child_handle = win32api.OpenProcess(
                win32con.SYNCHRONIZE, False, int(child_file.read_text())
            )
            assert process.poll() is None
        finally:
            process.close()
        try:
            assert win32event.WaitForSingleObject(child_handle, 5000) == win32event.WAIT_OBJECT_0
        finally:
            win32api.CloseHandle(child_handle)


def test_unexpected_startup_errors_are_redacted(monkeypatch, capsys):
    def fail(_):
        raise RuntimeError(CANARY + TOKEN)

    monkeypatch.setattr(startup, "windows", fail)
    assert startup.main(["windows"]) == 1
    output = capsys.readouterr()
    assert "Startup failed" in output.err
    assert CANARY not in output.err and TOKEN not in output.err


def test_import_cannot_load_models_or_spawn_services():
    code = """
import importlib.abc, os, subprocess, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('model import attempted')
def no_spawn(*args, **kwargs):
    raise AssertionError('process spawn attempted')
sys.meta_path.insert(0, NoModels())
subprocess.Popen = no_spawn
before = dict(os.environ)
import scripts.startup
assert dict(os.environ) == before
"""
    subprocess.run([sys.executable, "-c", code], cwd=startup.ROOT, check=True, timeout=15)


@pytest.mark.skipif(os.name != "nt", reason="Real Windows secure dependencies")
def test_real_windows_preflight_cannot_load_models_download_or_spawn():
    code = """
import importlib.abc, platform, socket, subprocess, sys
# Cache stdlib OS discovery: Python 3.11 may run `ver` when Argon2 asks platform.machine().
platform.uname()
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'huggingface_hub', 'datasets',
            'accelerate', 'bitsandbytes', 'unsloth', 'peft', 'trl',
        }:
            raise AssertionError('model or download import attempted')
def forbidden(*args, **kwargs):
    raise AssertionError('network, prompt or process attempted')
class NoSpawn(subprocess.Popen):
    # asyncio subclasses Popen while importing on Windows; reject construction, not definition.
    __init__ = forbidden
sys.meta_path.insert(0, NoModels())
subprocess.Popen = NoSpawn
socket.socket.connect = forbidden
socket.create_connection = forbidden
import builtins, getpass
builtins.input = getpass.getpass = forbidden
from scripts.startup import require_windows_secure_runtime
require_windows_secure_runtime()
assert 'runtime.serve' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=startup.ROOT, check=True, timeout=15)


@pytest.mark.parametrize("setup", [False, True])
@pytest.mark.parametrize("selection", ["override", "activated", "repository"])
def test_bash_wrapper_reuses_venv_and_offline_cache_without_running_python(
    tmp_path, setup, selection
):
    bash = (
        str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if os.name == "nt"
        else shutil.which("bash")
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is unavailable")
    root = tmp_path / "checkout with spaces"
    script = root / "scripts/runpod/start.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes((startup.ROOT / "scripts/runpod/start.sh").read_bytes())
    venv = root / (".venv" if selection == "repository" else "existing venv")
    fake = venv / "bin/python"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "CALL:$*" "HF_HOME=$HF_HOME" "HF_HUB_CACHE=$HF_HUB_CACHE" \\
    "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \\
    "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE" "TMPDIR=$TMPDIR" >> "$AEVON_TEST_CAPTURE"
""",
        encoding="utf-8",
        newline="\n",
    )
    fake.chmod(0o700)
    capture = root / "calls.txt"
    env = {
        **os.environ,
        "AEVON_TEST_CAPTURE": capture.as_posix(),
    }
    env.pop("AEVON_RUNPOD_VENV", None)
    env.pop("VIRTUAL_ENV", None)
    if selection == "override":
        env["AEVON_RUNPOD_VENV"] = venv.as_posix()
        env["VIRTUAL_ENV"] = (root / "unused activated venv").as_posix()
    elif selection == "activated":
        env["VIRTUAL_ENV"] = venv.as_posix()
    subprocess.run([bash, "-n", script.as_posix()], check=True, timeout=15)
    subprocess.run(
        [bash, script.as_posix(), *(["--setup"] if setup else []), "--timeout", "7"],
        env=env,
        cwd=tmp_path,
        check=True,
        timeout=15,
        capture_output=True,
    )
    output = capture.read_text()
    assert ("CALL:-m pip install -e .[runtime]" in output) is setup
    assert "CALL:-m scripts.startup runpod --timeout 7" in output
    for setting in [
        "HF_HOME=/workspace/hf",
        "HF_HUB_CACHE=/workspace/hf/hub",
        "HUGGINGFACE_HUB_CACHE=/workspace/hf/hub",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TMPDIR=/tmp",
    ]:
        assert setting in output


@pytest.mark.parametrize("dependencies_available", [False, True])
def test_bash_fresh_default_uses_pod_disk_then_reuses_without_installing(
    tmp_path, dependencies_available
):
    bash = (
        str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if os.name == "nt"
        else shutil.which("bash")
    )
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is unavailable")
    root = tmp_path / "checkout with spaces"
    script = root / "scripts/runpod/start.sh"
    script.parent.mkdir(parents=True)
    source = (startup.ROOT / "scripts/runpod/start.sh").read_text()
    assert "venv=/root/amitai-venv" in source
    assert "/workspace/venv" not in source
    # Rebase only the absolute default into a fake filesystem; never write /root or /workspace.
    script.write_text(
        source.replace("venv=/root/amitai-venv", 'venv="$AEVON_TEST_DISK/root/amitai-venv"'),
        encoding="utf-8",
        newline="\n",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    bootstrap = fake_bin / "python3"
    bootstrap.write_text(
        """#!/usr/bin/env bash
set -eu
printf 'CREATE:%s\\n' "$*" >> "$AEVON_TEST_CAPTURE"
[[ "$*" == "-m venv --system-site-packages $AEVON_TEST_DISK/root/amitai-venv" ]] || exit 90
mkdir -p "$4/bin"
cp "$AEVON_TEST_PYTHON" "$4/bin/python"
chmod +x "$4/bin/python"
""",
        encoding="utf-8",
        newline="\n",
    )
    fake_python = fake_bin / "venv-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf 'CALL:%s\\n' "$*" >> "$AEVON_TEST_CAPTURE"
if [[ "$1" == -c ]]; then exit "$AEVON_TEST_DEPENDENCY_STATUS"; fi
""",
        encoding="utf-8",
        newline="\n",
    )
    bootstrap.chmod(0o700)
    fake_python.chmod(0o700)
    capture = tmp_path / "calls.txt"
    env = {
        **os.environ,
        "AEVON_TEST_DISK": tmp_path.as_posix(),
        "AEVON_TEST_CAPTURE": capture.as_posix(),
        "AEVON_TEST_PYTHON": fake_python.as_posix(),
        "AEVON_TEST_DEPENDENCY_STATUS": "0",
    }
    env.pop("AEVON_RUNPOD_VENV", None)
    env.pop("VIRTUAL_ENV", None)
    # Set PATH in Bash so Git Bash's Windows path conversion cannot split a drive letter.
    command = [
        bash,
        "-c",
        'export PATH="$(cd -- "$1" && pwd):$PATH"; exec bash "$2" --timeout 7',
        "test",
        fake_bin.as_posix(),
        script.as_posix(),
    ]
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=15)
    output = capture.read_text()
    assert f"CREATE:-m venv --system-site-packages {tmp_path.as_posix()}/root/amitai-venv" in output
    assert output.count("CALL:-m pip install -e .[runtime]") == 1
    assert "CALL:-m scripts.startup runpod --timeout 7" in output
    assert not (tmp_path / "workspace").exists()

    capture.write_text("")
    env["AEVON_TEST_DEPENDENCY_STATUS"] = "0" if dependencies_available else "1"
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=15, check=False
    )
    output = capture.read_text()
    assert "CREATE:" not in output and "pip install" not in output
    assert ("CALL:-m scripts.startup runpod" in output) is dependencies_available
    assert result.returncode == (0 if dependencies_available else 1)
    if not dependencies_available:
        assert "--setup" in result.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux lifetime lock")
def test_runpod_lock_refuses_duplicate_then_releases():
    with (
        startup.runpod_lock(),
        pytest.raises(startup.StartupError, match="already running"),
        startup.runpod_lock(),
    ):
        pytest.fail("Duplicate launcher acquired the lock")
    with startup.runpod_lock():
        pass
