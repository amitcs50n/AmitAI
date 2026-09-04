"""Loopback-only UI integration fixture: real orchestration, fake CPU engine.

Invoked by frontend's test:integration, never by the production launcher.
"""

import json
import socket
import sys
from collections import Counter
from pathlib import Path

import uvicorn
from PIL import Image

from backend.app import create_app
from evaluation.hf_backend import GenerationOutput
from runtime.config import load_runtime_config
from runtime.generator import TransformersChatGenerator


def report(event):
    print(json.dumps({"event": event}), flush=True)


class FakeEngine:
    supports_vision = True

    def __init__(self):
        self.attempts = Counter()

    def generate_detailed_stream(self, messages, config, *, cancel_event):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        image = isinstance(user, list)
        if image:
            pixels = next(item["image"] for item in user if item["type"] == "image")
            assert pixels.mode == "RGB" and pixels.getpixel((0, 0)) == (255, 0, 0)
            user = next(item["text"] for item in user if item["type"] == "text")
            report("image-decoded")
        self.attempts[user] += 1
        if user.startswith("Pause") and self.attempts[user] == 1:
            yield "Partial answer "
            assert cancel_event.wait(10), "UI did not cancel the real backend producer"
            report("vision-cancelled" if image else "text-cancelled")
            return
        if user == "Fail image" and self.attempts[user] == 1:
            raise RuntimeError("PRIVATE_ENGINE_ERROR_CANARY")
        if user == "What is 17 * 83?":
            if messages[-1]["role"] == "system" and messages[-1]["content"].startswith("<tool_result>"):
                assert "1411" in messages[-1]["content"]
                report("calculator-result-used")
                output = "The answer is 1411."
            else:
                output = '<tool_call>{"name":"calculator","arguments":{"expression":"17 * 83"}}</tool_call>'
        elif user == "What is my ui.theme?":
            assert any("BLUE_CANARY" in str(m["content"]) for m in messages if m["role"] == "system")
            report("memory-used")
            output = "Your theme is BLUE_CANARY."
        else:
            output = "Red square shown." if image else "Hello from the CPU fixture."
        first, _, rest = output.partition(" ")
        yield first
        if cancel_event.wait(0.15):
            return
        if rest:
            yield " " + rest
        yield GenerationOutput(output, 100, 8)


def main():
    directory = Path(sys.argv[1]).resolve(strict=True)
    token = (directory / "local-api-token").read_text().strip()
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    image.save(directory / "red.png")
    image.close()
    engine = FakeEngine()

    def factory(config, seed):
        return engine

    factory.supports_vision = True
    generator = TransformersChatGenerator(load_runtime_config(), engine_factory=factory)
    app = create_app(
        f"sqlite:///{(directory / 'test.db').as_posix()}",
        generator=generator,
        encrypted_storage=False,  # Temporary test DB only; asset encryption stays real.
        asset_directory=directory / "assets",
        local_api_token=token,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        print(json.dumps({"port": port}), flush=True)
        server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
