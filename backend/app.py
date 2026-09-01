"""FastAPI application exposing the persistent AmitAI chat contract."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.types import ASGIApp, Receive, Scope, Send

from .chat_service import (
    ChatGenerationError,
    ChatPrivacyError,
    ChatService,
    ChatStreamEvent,
    ConversationNotFoundError,
    GenerationCallable,
    ResponseGenerator,
    StreamingResponseGenerator,
)
from .database import DEFAULT_DATABASE_URL, Database, DatabaseKeyInput
from .memory import (
    LOCAL_MEMORY_OWNER_ID,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryService,
    MemoryValidationError,
)
from .repositories import ConversationRepository
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationRename,
    MemoryCreate,
    MemoryRead,
    MemorySearch,
    MemoryUpdate,
)
from .security import LocalApiAuthMiddleware, security_state

LOGGER = logging.getLogger(__name__)
SSE_HEARTBEAT_SECONDS = 15.0


def _encode_sse(event: ChatStreamEvent) -> str:
    data = json.dumps(
        jsonable_encoder(event.data),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event.event}\ndata: {data}\n\n"


def get_session(request: Request) -> Iterator[Session]:
    factory = request.app.state.database.session_factory
    with factory() as session:
        yield session


def create_app(
    database_url: str = DEFAULT_DATABASE_URL,
    *,
    database_key: DatabaseKeyInput | None = None,
    encrypted_storage: bool = True,
    encrypt_existing_database: bool = False,
    generator: (
        ResponseGenerator | StreamingResponseGenerator | GenerationCallable | None
    ) = None,
    memory_owner_id: str = LOCAL_MEMORY_OWNER_ID,
    local_api_token: str | None = None,
    enforce_local_auth: bool = True,
    enable_dev_docs: bool = False,
) -> FastAPI:
    database = Database.from_url(
        database_url,
        encrypted=encrypted_storage,
        encryption_key=database_key,
        migrate_plaintext=encrypt_existing_database,
    )
    stream_executor: ThreadPoolExecutor | None = None
    stream_gate: asyncio.Semaphore | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal stream_executor, stream_gate
        database.create_schema()
        stream_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="amitai-chat-stream",
        )
        stream_gate = asyncio.Semaphore(1)
        try:
            yield
        finally:
            executor = stream_executor
            stream_executor = None
            stream_gate = None
            if executor is not None:
                await asyncio.to_thread(
                    executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
            database.engine.dispose()

    application = FastAPI(
        title="AmitAI Backend",
        lifespan=lifespan,
        docs_url="/docs" if enable_dev_docs else None,
        redoc_url="/redoc" if enable_dev_docs else None,
        openapi_url="/openapi.json" if enable_dev_docs else None,
    )
    if enforce_local_auth:
        application.add_middleware(LocalApiAuthMiddleware, token=local_api_token)
    application.state.database = database
    application.state.generator = generator
    application.state.memory_owner_id = memory_owner_id
    for key, value in security_state(
        auth_enabled=enforce_local_auth,
        docs_enabled=enable_dev_docs,
    ).items():
        setattr(application.state, key, value)

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/api/conversations",
        response_model=ConversationRead,
        status_code=status.HTTP_201_CREATED,
    )
    def create_conversation(
        payload: ConversationCreate | None = None,
        session: Session = Depends(get_session),
    ) -> ConversationRead:
        title = payload.title if payload and payload.title is not None else "New conversation"
        repository = ConversationRepository(session)
        with session.begin():
            conversation = repository.create(title)
        return ConversationRead.model_validate(conversation)

    @application.get("/api/conversations", response_model=list[ConversationRead])
    def list_conversations(session: Session = Depends(get_session)) -> list[ConversationRead]:
        conversations = ConversationRepository(session).list()
        return [ConversationRead.model_validate(item) for item in conversations]

    @application.get(
        "/api/conversations/{conversation_id}",
        response_model=ConversationDetail,
    )
    def read_conversation(
        conversation_id: str,
        session: Session = Depends(get_session),
    ) -> ConversationDetail:
        conversation = ConversationRepository(session).get_with_messages(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return ConversationDetail.model_validate(conversation)

    @application.patch(
        "/api/conversations/{conversation_id}",
        response_model=ConversationRead,
    )
    def rename_conversation(
        conversation_id: str,
        payload: ConversationRename,
        session: Session = Depends(get_session),
    ) -> ConversationRead:
        repository = ConversationRepository(session)
        with session.begin():
            conversation = repository.get(conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            repository.rename(conversation, payload.title)
        return ConversationRead.model_validate(conversation)

    @application.delete(
        "/api/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_conversation(
        conversation_id: str,
        session: Session = Depends(get_session),
    ) -> Response:
        repository = ConversationRepository(session)
        with session.begin():
            conversation = repository.get(conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            repository.delete(conversation)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        request: Request,
        session: Session = Depends(get_session),
    ) -> ChatResponse:
        service = ChatService(
            session,
            generator=request.app.state.generator,
            memory_owner_id=request.app.state.memory_owner_id,
        )
        try:
            result = service.chat(
                conversation_id=payload.conversation_id,
                message=payload.message,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        except ChatPrivacyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except ChatGenerationError as exc:
            raise HTTPException(status_code=500, detail="Assistant generation failed") from exc
        except MemoryConflictError as exc:
            raise HTTPException(status_code=409, detail="Memory changed; retry") from exc

        return ChatResponse.model_validate(result, from_attributes=True)

    @application.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        selected_generator = request.app.state.generator

        async def event_source() -> AsyncIterator[str]:
            gate = stream_gate
            executor = stream_executor
            if gate is None or executor is None:
                raise RuntimeError("Streaming runtime is not available outside app lifespan")

            while True:
                try:
                    await asyncio.wait_for(
                        gate.acquire(),
                        timeout=SSE_HEARTBEAT_SECONDS,
                    )
                    break
                except TimeoutError:
                    yield ": keep-alive\n\n"

            loop = asyncio.get_running_loop()
            events: asyncio.Queue[ChatStreamEvent | object] = asyncio.Queue()
            stream_end = object()
            cancel_event = threading.Event()

            def publish(item: ChatStreamEvent | object, *, force: bool = False) -> None:
                if cancel_event.is_set() and not force:
                    return
                try:
                    loop.call_soon_threadsafe(events.put_nowait, item)
                except RuntimeError:
                    cancel_event.set()

            def produce() -> None:
                service_stream: Iterator[ChatStreamEvent] | None = None
                try:
                    with database.session_factory() as stream_session:
                        service = ChatService(
                            stream_session,
                            generator=selected_generator,
                            memory_owner_id=request.app.state.memory_owner_id,
                        )
                        service_stream = service.stream_chat(
                            conversation_id=payload.conversation_id,
                            message=payload.message,
                            cancel_event=cancel_event,
                        )
                        for event in service_stream:
                            if cancel_event.is_set():
                                break
                            publish(event)
                except ConversationNotFoundError:
                    publish(
                        ChatStreamEvent(
                            event="error",
                            data={"detail": "Conversation not found"},
                        ),
                        force=True,
                    )
                except ChatPrivacyError as exc:
                    publish(
                        ChatStreamEvent(event="error", data={"detail": str(exc)}),
                        force=True,
                    )
                except ChatGenerationError:
                    LOGGER.error("Streaming assistant generation failed")
                    publish(
                        ChatStreamEvent(
                            event="error",
                            data={"detail": "Assistant generation failed"},
                        ),
                        force=True,
                    )
                except MemoryConflictError:
                    LOGGER.error("Streaming memory commit conflicted")
                    publish(
                        ChatStreamEvent(
                            event="error",
                            data={"detail": "Memory changed; retry"},
                        ),
                        force=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Unexpected streaming chat failure failure=%s",
                        type(exc).__name__,
                    )
                    publish(
                        ChatStreamEvent(
                            event="error",
                            data={"detail": "Assistant generation failed"},
                        ),
                        force=True,
                    )
                finally:
                    if service_stream is not None:
                        service_stream.close()
                    publish(stream_end, force=True)

            try:
                producer = loop.run_in_executor(executor, produce)
            except Exception:
                gate.release()
                raise
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(
                            events.get(),
                            timeout=SSE_HEARTBEAT_SECONDS,
                        )
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if item is stream_end:
                        break
                    if not isinstance(item, ChatStreamEvent):
                        raise TypeError("Streaming producer returned an invalid event")
                    yield _encode_sse(item)
            finally:
                cancel_event.set()
                if producer.done():
                    gate.release()
                else:
                    producer.add_done_callback(lambda _: gate.release())

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/api/memory", response_model=list[MemoryRead])
    def list_memory(
        request: Request,
        memory_status: str = Query(default="active", alias="status"),
        category: str | None = None,
        session: Session = Depends(get_session),
    ) -> list[MemoryRead]:
        if any(key not in {"status", "category"} for key in request.query_params):
            raise HTTPException(status_code=422, detail="Unsupported memory list parameter")
        service = MemoryService(session, owner_id=memory_owner_id)
        try:
            records = service.list_memories(status=memory_status, category=category)
        except MemoryValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [MemoryRead.model_validate(record) for record in records]

    @application.post("/api/memory/search", response_model=list[MemoryRead])
    def search_memory(
        payload: MemorySearch,
        session: Session = Depends(get_session),
    ) -> list[MemoryRead]:
        records = MemoryService(session, owner_id=memory_owner_id).retrieve(payload.query)
        return [MemoryRead.model_validate(record) for record in records]

    @application.post(
        "/api/memory",
        response_model=MemoryRead,
        status_code=status.HTTP_201_CREATED,
    )
    def create_memory(
        payload: MemoryCreate,
        session: Session = Depends(get_session),
    ) -> MemoryRead:
        service = MemoryService(session, owner_id=memory_owner_id)
        try:
            with session.begin():
                mutation = service.stage_create(
                    category=payload.category,
                    key=payload.key,
                    value=payload.value,
                    sensitivity=payload.sensitivity,
                )
                record = service.apply(mutation)
        except MemoryConflictError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryRead.model_validate(record)

    @application.patch("/api/memory/{memory_id}", response_model=MemoryRead)
    def update_memory(
        memory_id: str,
        payload: MemoryUpdate,
        session: Session = Depends(get_session),
    ) -> MemoryRead:
        service = MemoryService(session, owner_id=memory_owner_id)
        try:
            with session.begin():
                mutation = service.stage_update(
                    memory_id, value=payload.value, sensitivity=payload.sensitivity
                )
                record = service.apply(mutation)
        except MemoryNotFoundError as exc:
            session.rollback()
            raise HTTPException(status_code=404, detail="Memory not found") from exc
        except MemoryConflictError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryRead.model_validate(record)

    @application.delete(
        "/api/memory/{memory_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_memory(
        memory_id: str,
        session: Session = Depends(get_session),
    ) -> Response:
        service = MemoryService(session, owner_id=memory_owner_id)
        try:
            with session.begin():
                mutation = service.stage_delete(memory_id)
                service.apply(mutation)
        except MemoryNotFoundError as exc:
            session.rollback()
            raise HTTPException(status_code=404, detail="Memory not found") from exc
        except MemoryConflictError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return application


class LazyConfiguredApplication:
    """Delay environment-backed construction until ASGI startup."""

    def __init__(self, factory: Callable[[], ASGIApp]) -> None:
        self.factory = factory
        self.application: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.application is None:
            self.application = self.factory()
        await self.application(scope, receive, send)


def create_configured_app() -> FastAPI:
    """Fail closed when bypassing the interactive secure launcher."""

    raise RuntimeError(
        "Direct backend ASGI startup is unsupported; use python -m runtime.serve"
    )


app = LazyConfiguredApplication(create_configured_app)
