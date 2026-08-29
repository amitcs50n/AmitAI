"""FastAPI application exposing the persistent AmitAI chat contract."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from .chat_service import (
    ChatGenerationError,
    ChatService,
    ConversationNotFoundError,
    GenerationCallable,
    ResponseGenerator,
)
from .database import DEFAULT_DATABASE_URL, Database
from .repositories import ConversationRepository
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    ConversationRename,
)


def get_session(request: Request) -> Iterator[Session]:
    factory = request.app.state.database.session_factory
    with factory() as session:
        yield session


def create_app(
    database_url: str = DEFAULT_DATABASE_URL,
    *,
    generator: ResponseGenerator | GenerationCallable | None = None,
) -> FastAPI:
    database = Database.from_url(database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.create_schema()
        try:
            yield
        finally:
            database.engine.dispose()

    application = FastAPI(title="AmitAI Backend", lifespan=lifespan)
    application.state.database = database
    application.state.generator = generator

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
        service = ChatService(session, generator=request.app.state.generator)
        try:
            result = service.chat(
                conversation_id=payload.conversation_id,
                message=payload.message,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        except ChatGenerationError as exc:
            raise HTTPException(status_code=500, detail="Assistant generation failed") from exc

        return ChatResponse.model_validate(result, from_attributes=True)

    return application


app = create_app()
