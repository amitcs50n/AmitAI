"""CPU-only structural guard, provider, persistence, privacy and vision contracts."""

import json
from dataclasses import asdict
from threading import Event

import pytest
from fastapi.testclient import TestClient

from backend.chat_service import (
    ChatGenerationDelta,
    ChatService,
    RemoteProjection,
    RemoteVisionDisclosureError,
)
from backend.chat_service import (
    GenerationMessage as Message,
)
from backend.memory import format_memory_context
from runtime.config import load_production_runtime_config
from runtime.context import compile_model_context
from runtime.epistemic import epistemic_preflight
from runtime.generator import ProviderChatGenerator, TransformersChatGenerator
from runtime.privacy import InferenceExecutionScope
from tests.app_factory import create_test_app
from tests.test_assets import image_bytes
from tests.test_remote_privacy import RemoteHarness, assert_success, chat, durable_snapshot, memory
from tests.test_remote_vision import Harness as VisionHarness
from tests.test_vision import VisionEngine, assert_closed, vision_generator

ENV_PROMPT = "What's the exact environment-variable name our dispatch service reads for its database connection?"
OPENING = "What did I ask in my very first message?"
PREVIOUS = "What did I ask immediately before this?"
COUNT = [Message("user" if i % 2 == 0 else "assistant", f"retained turn {i}") for i in range(26)]
OVERSIZED = [Message("user", "Old task"), Message("assistant", "x" * 20_001)]


def remembered(value="PostgreSQL", key="dispatch.database"):
    return Message("system", format_memory_context([
        {"category": "project", "key": key, "value": value},
    ]))


def compile_context(prompt, history=(), scope=InferenceExecutionScope.LOCAL):
    return compile_model_context(
        [*history, Message("user", prompt)], runtime_system_prompt="Runtime rules",
        tool_instructions="Tool rules", execution_scope=scope,
    )


def decide(prompt, history=()):
    return epistemic_preflight(compile_context(prompt, history))


@pytest.mark.parametrize("prompt", [
    OPENING, "What was my opening message?", "Repeat the earliest message.",
    "What did I say before the messages you can currently see?",
    "What was the message before the oldest one you have?",
    "Tell me exactly what I wrote before this retained history.",
    "Tell me my omitted first message in exactly five words.",
])
def test_missing_opening_and_before_window(prompt):
    decision = decide(prompt, COUNT)
    assert decision.kind == "missing_history"
    assert "isn't available" in decision.deterministic_response


@pytest.mark.parametrize("prompt", [
    PREVIOUS, "What did I ask before this?",
    "Remind me what I asked you to do immediately before this request.",
    "Restate the task I gave you immediately before this message.",
    "Repeat the previous message.",
])
@pytest.mark.parametrize("history", [[], OVERSIZED])
def test_unavailable_previous_turn(prompt, history):
    assert decide(prompt, history).kind == "missing_history"


@pytest.mark.parametrize("prompt", [
    "Could you shift that to next Wednesday?", "Cancel that.", "Can you rewrite this?",
    "Should I take it tomorrow?", "What does that mean?", "What is this?",
    "Explain that error", "Move these to Monday.", "Please summarize them.",
])
def test_obvious_zero_referent(prompt):
    assert decide(prompt).kind == "ambiguous_reference"
    assert decide(prompt, [remembered()]).kind == "ambiguous_reference"


@pytest.mark.parametrize("prompt,history", [
    (PREVIOUS, COUNT), (OPENING, [Message("user", "First actual message")]),
    ("What database did I tell you our dispatch service uses?", [remembered()]),
    ("Draft reminder: 'Submit the draft by Monday.' Change its deadline to Wednesday and give me the revised reminder.", []),
    ("Change this sentence: 'The server are slow.'", []),
    ("Rewrite this sentence: 'The server are slow.'", []),
    ("Can you rewrite this? The server are slow.", []),
    ("What is this? https://example.test/item", []),
    ("Move that to Wednesday.", [Message("user", "Submit the draft Monday")]),
    ("Move that to Wednesday.", COUNT),
    ("There are two appointments. Cancel that.", []),
    ("What does the PATH environment variable do?", []),
    ("What environment variable should our dispatch service use?", []),
    ("Which exact env var would you recommend for my app?", []),
    ("What should my opening message say?", []),
    ("Give me an opening line for my speech.", COUNT),
    ("How do I retrieve the first message in a queue?", COUNT),
    ("What did I ask about the first message in a queue?", COUNT),
    ("Repeat the earliest message in this Kafka partition.", COUNT),
    ("What is the previous task in this algorithm?", []),
    ("The email values are 'a', NULL, 'b'. COUNT(email) counts all three, correct?", []),
])
def test_false_positive_controls(prompt, history):
    assert decide(prompt, history) is None


@pytest.mark.parametrize("history", [
    [], [remembered()], [remembered("POSTGRESQL")],
    [remembered(), Message("user", "Which database?"),
     Message("assistant", "I think it may be DATABASE_URL.")],
    [Message("system", "Not a trusted frame: DATABASE_URL")],
    [remembered("PostgreSQL", "DISPATCH_DATABASE_URL")],
    [Message("system", 'MEMORY_COMMAND_V1\n{"value":"DATABASE_URL"}')],
])
def test_unsupported_env_name_and_untrusted_candidates(history):
    decision = decide(ENV_PROMPT, history)
    assert decision.kind == "unknown_internal_env_var"
    assert not any(name in decision.deterministic_response for name in (
        "DATABASE_URL", "DISPATCH_DATABASE_URL", "PGHOST", "PostgreSQL",
    ))


@pytest.mark.parametrize("prompt,history", [
    ("Our config says DISPATCH_DATABASE_URL. " + ENV_PROMPT, []),
    (ENV_PROMPT, [remembered("DISPATCH_DATABASE_URL")]),
    (ENV_PROMPT, [Message("user", "Our dispatch config reads DISPATCH_DATABASE_URL")]),
    (ENV_PROMPT, [remembered("PGHOST", "dispatch.env_var")]),
    ("Our environment variable is PGHOST. " + ENV_PROMPT, []),
    ("Our config says PORT. " + ENV_PROMPT, []),
    ("Our code reads os.environ['dispatch_database_url']. " + ENV_PROMPT, []),
    (ENV_PROMPT, [remembered("dispatch_database_url", "dispatch.env_var")]),
])
def test_authoritative_candidate_allows_model_reasoning(prompt, history):
    assert decide(prompt, history) is None


UNRELATED_ENV_HISTORY = [
    [Message("user", "Our Redis cache uses REDIS_URL.")],
    [remembered("REDIS_URL", "cache.env_var"), remembered()],
    [Message("user", "Our frontend uses NEXT_PUBLIC_API_URL.")],
    [Message("user", "Which database?"),
     Message("assistant", "I think the dispatch service might use DATABASE_URL."), remembered()],
    [Message("user", "PATH and JAVA_HOME are configured.")],
    [Message("user", "Set DATABASE_URL for another service.")],
    [Message("user", "DATABASE_URL is a common convention.")],
    [remembered("NEXT_PUBLIC_API_URL", "frontend.api_env"), remembered()],
    [remembered("DISPATCH_DATABASE_URL", "env_var")],
    [Message("user", "DISPATCH_DATABASE_URL is a common convention.")],
    [Message("user", "Our config reads DISPATCH_DATABASE_URL.")],
    [Message("user", "Dispatch uses PostgreSQL. Our Redis cache uses REDIS_URL.")],
    [remembered("REDIS_URL", "cache.database_env")],
    [remembered("DISPATCH_DATABASE_URL", "redispatch.env_var")],
    [Message("user", "Redispatch uses DISPATCH_DATABASE_URL.")],
    [remembered("Dispatch uses DATABASE_URL.", "cache.env_var")],
]


@pytest.mark.parametrize("history", UNRELATED_ENV_HISTORY)
def test_unrelated_env_evidence_cannot_authorize_dispatch(history):
    decision = decide(ENV_PROMPT, history)
    assert decision.kind == "unknown_internal_env_var"
    assert decision == decide(ENV_PROMPT, [remembered()])


@pytest.mark.parametrize("history", [
    [Message("user", "The dispatch database env var is DISPATCH_DATABASE_URL.")],
    [Message("user", "Dispatch uses DISPATCH_DATABASE_URL.")],
    [Message("user", 'In dispatch config, os.environ["DISPATCH_DATABASE_URL"] is used.')],
    [Message("user", 'In the dispatch config, getenv("DISPATCH_DATABASE_URL") is used.')],
    [remembered("DISPATCH_DATABASE_URL", "dispatch.env_var")],
    [remembered("DISPATCH_DATABASE_URL", "dispatch.database_env")],
    [remembered("DATABASE_URL", "Dispatch.Database_Env")],
    [remembered("Dispatch uses DISPATCH_DATABASE_URL.", "env_var")],
])
def test_related_env_evidence_allows_normal_reasoning(history):
    assert decide(ENV_PROMPT, history) is None


@pytest.mark.parametrize("prompt", [
    "Our config says DISPATCH_DATABASE_URL. Which environment variable does our dispatch service use?",
    'The dispatch service reads os.environ["DISPATCH_DATABASE_URL"]. What\'s the exact env-var name?',
    "Our dispatch config uses DISPATCH_DATABASE_URL. What exact env var does the dispatch service use?",
])
def test_current_user_explicit_scoped_evidence_remains_allowed(prompt):
    assert decide(prompt) is None


@pytest.mark.parametrize("evidence", [
    "Our Redis cache uses REDIS_URL.",
    "Our frontend uses NEXT_PUBLIC_API_URL.",
    "Set PATH and JAVA_HOME correctly.",
    "Our Redis config says REDIS_URL.",
])
def test_current_prompt_unrelated_evidence_is_not_bound_by_the_question(evidence):
    assert decide(evidence + " " + ENV_PROMPT).kind == "unknown_internal_env_var"


@pytest.mark.parametrize("evidence,guarded", [
    ("Dispatch uses DISPATCH_DATABASE_URL.", True),
    ("Billing uses BILLING_DATABASE_URL.", False),
])
def test_scope_is_extracted_from_the_query_not_hardcoded(evidence, guarded):
    prompt = "What exact env var does our billing service use for its database connection?"
    decision = decide(prompt, [Message("user", evidence)])
    assert (decision is not None) is guarded


def test_component_target_can_bind_when_the_query_has_no_named_service():
    prompt = "What exact env var does this project use for the DB connection?"
    assert decide(prompt, [remembered("DATABASE_URL", "database.env_var")]) is None
    assert decide(prompt, [remembered("REDIS_URL", "cache.env_var")]).kind == "unknown_internal_env_var"


@pytest.mark.parametrize("prompt", [
    "What does PATH do?", "What is JAVA_HOME?",
    "What environment variable should I use for my app?",
    "Suggest an env-var name for my dispatch service.",
    "Give me an example PostgreSQL connection env var.",
])
def test_general_env_knowledge_remains_unguarded(prompt):
    assert decide(prompt) is None


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("history", UNRELATED_ENV_HISTORY[:5])
def test_unrelated_env_evidence_bypasses_provider_and_tools(streaming, history, monkeypatch):
    generator = forbidden_generator()
    monkeypatch.setattr(generator._tool_registry, "execute", lambda *a, **k: pytest.fail("Tool executed"))
    messages = [*history, Message("user", ENV_PROMPT)]
    if streaming:
        events = list(generator.stream_response(messages, cancel_event=Event()))
        result = events[-1]
        assert "".join(event.delta for event in events[:-1]) == result.response
    else:
        result = generator.generate_response(messages)
    assert result.input_tokens == result.output_tokens == 0 and result.tools == []
    assert result.validator["epistemic_guardrail"] == {
        "triggered": True, "kind": "unknown_internal_env_var",
        "reason": "internal_env_name_unavailable", "provider_bypassed": True,
        "mechanical_override": False,
    }


def test_context_metadata_has_roles_positions_and_no_omitted_content():
    recent = COUNT[-20:]
    decisions = []
    contexts = []
    for canary in ("OMITTED_DATABASE_URL", "different private topic"):
        context = compile_context(OPENING, [Message("user", canary), *recent])
        contexts.append(context)
        decisions.append(epistemic_preflight(context))
        assert canary not in json.dumps(asdict(context)) + json.dumps(asdict(decisions[-1]))
        assert context.history_truncated
        assert context.latest_prior_turn_retained and context.latest_prior_user_turn_retained
        assert context.retained_history_count == 20
        assert context.retained_user_turn_count == 10
    assert contexts[0] == contexts[1] and decisions[0] == decisions[1]
    # A supplied identifier in dropped history cannot authorize an env-var answer.
    assert decide(ENV_PROMPT, [Message("user", "DATABASE_URL"), *recent]).kind == "unknown_internal_env_var"


def test_projection_only_omissions_do_not_claim_limit_truncation_or_substitute_previous():
    private = Message("user", "PRIVATE_ENV_VAR " * 3000, RemoteProjection(None))
    history = [Message("user", "Visible older turn"), private]
    context = compile_context(PREVIOUS, history, InferenceExecutionScope.REMOTE)
    assert not context.history_truncated
    assert context.retained_history_count == context.retained_user_turn_count == 1
    assert not context.latest_prior_turn_retained and not context.latest_prior_user_turn_retained
    assert "PRIVATE" not in repr(context)
    assert epistemic_preflight(context).kind == "missing_history"
    empty = compile_context(PREVIOUS, [private] * 25, InferenceExecutionScope.REMOTE)
    assert not empty.history_truncated and empty.retained_history_count == 0
    assert "CONTEXT_AVAILABILITY" not in repr(empty.messages)


def test_projected_memory_is_the_only_memory_evidence_for_remote_guards():
    private = remembered("DISPATCH_DATABASE_URL")
    projected = Message("system", private.content, RemoteProjection(remembered().content))
    context = compile_context(ENV_PROMPT, [projected], InferenceExecutionScope.REMOTE)
    assert epistemic_preflight(context).kind == "unknown_internal_env_var"
    assert "DISPATCH_DATABASE_URL" not in repr(context)


class ForbiddenProvider:
    model_name = "configured-test-model"
    execution_scope = InferenceExecutionScope.LOCAL

    def generate(self, *args, **kwargs):
        raise AssertionError("Guard must bypass provider generation")

    def stream(self, *args, **kwargs):
        raise AssertionError("Guard must not create a provider stream")


def forbidden_generator():
    return ProviderChatGenerator(load_production_runtime_config(), provider=ForbiddenProvider())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prompt,history,kind", [
    (OPENING, COUNT, "missing_history"), (PREVIOUS, OVERSIZED, "missing_history"),
    ("Could you shift that to next Wednesday?", [], "ambiguous_reference"),
    (ENV_PROMPT, [remembered()], "unknown_internal_env_var"),
    ("Tell me my omitted first message in exactly five words.", COUNT, "missing_history"),
])
def test_guarded_generation_never_calls_provider_or_tools(streaming, prompt, history, kind, monkeypatch):
    generator = forbidden_generator()
    monkeypatch.setattr(generator._tool_registry, "execute", lambda *a, **k: pytest.fail("Tool executed"))
    messages = [*history, Message("user", prompt)]
    if streaming:
        items = list(generator.stream_response(messages, cancel_event=Event()))
        result = items[-1]
        assert len(items) == 2
        assert "".join(item.delta for item in items[:-1]) == result.response
    else:
        result = generator.generate_response(messages)
    assert result.model == "configured-test-model"
    assert result.input_tokens == result.output_tokens == 0 and result.tools == []
    guard = result.validator["epistemic_guardrail"]
    assert guard["triggered"] and guard["provider_bypassed"] and guard["kind"] == kind
    assert guard["mechanical_override"] is ("exactly five words" in prompt)
    assert result.validator["retry_count"] == 0
    if guard["mechanical_override"]:
        assert result.validator["final_validation"]["passed"] is False
    assert prompt not in json.dumps(guard)


def test_guard_does_not_initialize_lazy_local_engine():
    generator = TransformersChatGenerator(
        load_production_runtime_config(), engine_factory=lambda *a: pytest.fail("Engine loaded"),
    )
    assert generator.generate_response([Message("user", "Cancel that.")]).input_tokens == 0


@pytest.mark.parametrize("stage", ["before", "after_delta", "close"])
def test_guarded_stream_cancellation_has_no_terminal_result(stage):
    generator = forbidden_generator()
    event = Event()
    stream = generator.stream_response([Message("user", "Cancel that.")], cancel_event=event)
    if stage == "before":
        event.set()
    else:
        assert isinstance(next(stream), ChatGenerationDelta)
        if stage == "close":
            stream.close()
        else:
            event.set()
    assert list(stream) == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_guard_persists_atomic_turn_and_metadata_without_remote_transport(streaming, remote, monkeypatch):
    harness = RemoteHarness()
    generator = harness.generator if remote else forbidden_generator()
    calls = []
    if remote:
        # Wrap entry points to count even failures before HTTP, not only transport.
        def forbidden(*args, **kwargs):
            calls.append(True)
            raise AssertionError("Remote provider invoked")
        monkeypatch.setattr(harness.provider, "generate", forbidden)
        monkeypatch.setattr(harness.provider, "stream", forbidden)
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=generator)
    try:
        with TestClient(app) as client:
            private = memory(client, "dispatch.database", "Private project database detail")
            response, events = chat(client, ENV_PROMPT, streaming)
            body = assert_success(response, events, streaming)
            if streaming:
                assert [name for name, _ in events] == ["start", "text", "final", "done"]
            assert body["metadata"]["validator"]["epistemic_guardrail"]["kind"] == "unknown_internal_env_var"
            assert body["metadata"]["input_tokens"] == body["metadata"]["output_tokens"] == 0
            detail = client.get(f"/api/conversations/{body['conversation_id']}").json()
            assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
            assert [m["content"] for m in detail["messages"]] == [ENV_PROMPT, body["response"]]
            assert detail["messages"][-1]["metadata"]["validator"] == body["metadata"]["validator"]
            assert detail["title"] and detail["updated_at"]
            assert private["id"] in {ref["id"] for ref in body["metadata"]["memory"]}
            assert client.get("/api/memory").json()[0]["value"] == "Private project database detail"
            assert not calls and harness.calls == harness.paths == harness.bodies == []
    finally:
        harness.provider.close()


@pytest.mark.parametrize("stage", ["before", "start", "text"])
def test_cancelled_guard_does_not_persist_partial_turn(stage):
    app = create_test_app("sqlite+pysqlite:///:memory:", generator=forbidden_generator())
    with TestClient(app), app.state.database.session_factory() as session:
        before = durable_snapshot(app)
        event = Event()
        if stage == "before":
            event.set()
        service = ChatService(session, generator=app.state.generator)
        stream = service.stream_chat(conversation_id=None, message="Cancel that.", cancel_event=event)
        if stage != "before":
            assert next(stream).event == "start"
            if stage == "text":
                assert next(stream).event == "text"
            event.set()
        assert list(stream) == []
        assert durable_snapshot(app) == before


@pytest.mark.parametrize("prompt", ["What is this?", "Explain that error"])
@pytest.mark.parametrize("streaming", [False, True])
def test_local_vision_excludes_text_guards(prompt, streaming):
    engine = VisionEngine()
    generator, loads = vision_generator(engine)
    messages = [Message("user", prompt)]
    result = (list(generator.stream_vision_response(messages, image_bytes(), cancel_event=Event()))[-1]
              if streaming else generator.generate_vision_response(messages, image_bytes()))
    assert "epistemic_guardrail" not in result.validator
    assert len(engine.calls) == len(loads) == len(engine.images) == 1
    assert_closed(engine.images[0])


@pytest.mark.parametrize("prompt", ["What is this?", "Explain that error"])
@pytest.mark.parametrize("streaming", [False, True])
def test_remote_vision_excludes_text_guards_and_still_requires_consent(prompt, streaming):
    harness = VisionHarness()
    try:
        with pytest.raises(RemoteVisionDisclosureError):
            if streaming:
                list(harness.generator.stream_vision_response(
                    [Message("user", prompt)], image_bytes(), cancel_event=Event(),
                ))
            else:
                harness.generator.generate_vision_response([Message("user", prompt)], image_bytes())
        assert not harness.requests and not harness.engine.calls
        result = harness.run(streaming, prompt=prompt)
        assert "epistemic_guardrail" not in result.validator
        assert len(harness.requests) == len(harness.engine.calls) == 1
        assert harness.requests[0].url.path.startswith("/v1/vision")
        assert_closed(harness.engine.images[0])
    finally:
        harness.provider.close()
        harness.client.close()
