"""Narrow, authenticated-by-app asset endpoints; bounded in-memory multipart."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, Request, Response
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

from .asset_storage import MAX_ASSET_BYTES
from .assets import AssetError, AssetService, normalize_image
from .schemas import AssetRead

MAX_UPLOAD_BODY_BYTES = MAX_ASSET_BYTES + 16 * 1024
router = APIRouter(prefix="/api/assets")


def parse_upload(body: bytes, content_type: str) -> tuple[bytes, str, str, str, str | None]:
    """No temporary raw files and no logging of parser-supplied diagnostics."""
    media_type, parameters = parse_options_header(content_type)
    boundary = parameters.get(b"boundary", b"")
    if media_type != b"multipart/form-data":
        raise AssetError("Upload requires multipart/form-data", 415)
    if set(parameters) != {b"boundary"} or not re.fullmatch(
        rb"[A-Za-z0-9'()+_,./:=?-]{1,70}", boundary
    ):
        raise AssetError()
    fields: dict[str, str] = {}
    file_data: bytes | None = None
    filename = ""
    image_type = ""
    headers: dict[bytes, bytes] = {}
    header_name = bytearray()
    header_value = bytearray()
    data = BytesIO()
    ended = False
    part_count = 0

    def begin() -> None:
        nonlocal headers, data, part_count
        part_count += 1
        if part_count > 3:
            raise AssetError()
        headers = {}
        data = BytesIO()

    def header_field(chunk: bytes, start: int, end: int) -> None:
        header_name.extend(chunk[start:end])
        if len(header_name) > 64:
            raise AssetError()

    def header_content(chunk: bytes, start: int, end: int) -> None:
        header_value.extend(chunk[start:end])
        if len(header_value) > 2048:
            raise AssetError()

    def header_end() -> None:
        name = bytes(header_name).lower()
        if name not in {b"content-type", b"content-disposition"} or name in headers:
            raise AssetError()
        headers[name] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def part_data(chunk: bytes, start: int, end: int) -> None:
        if data.tell() + end - start > MAX_ASSET_BYTES:
            raise AssetError("Image exceeds 20 MiB", 413)
        data.write(chunk[start:end])

    def part_end() -> None:
        nonlocal file_data, filename, image_type
        disposition, options = parse_options_header(headers.get(b"content-disposition", b""))
        if disposition != b"form-data":
            raise AssetError()
        name = options.get(b"name", b"").decode("ascii")
        if name == "file":
            if file_data is not None or set(options) != {b"name", b"filename"}:
                raise AssetError()
            file_data = data.getvalue()
            filename = options[b"filename"].decode("utf-8", errors="replace")
            image_type = headers.get(b"content-type", b"").decode("ascii")
        else:
            if name not in {"persistence_mode", "conversation_id"} or name in fields:
                raise AssetError()
            if set(options) != {b"name"} or len(headers) != 1 or data.tell() > 128:
                raise AssetError()
            fields[name] = data.getvalue().decode("ascii")

    def end() -> None:
        nonlocal ended
        ended = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": begin,
            "on_header_field": header_field,
            "on_header_value": header_content,
            "on_header_end": header_end,
            "on_part_data": part_data,
            "on_part_end": part_end,
            "on_end": end,
        },
    )
    silent = logging.Logger("asset-multipart-private")  # noqa: LOG001 - isolated per-parser sink
    silent.addHandler(logging.NullHandler())
    silent.propagate = False
    parser.logger = silent
    try:
        if not body.endswith(b"--" + boundary + b"--\r\n") and not body.endswith(
            b"--" + boundary + b"--"
        ):
            raise AssetError()
        for offset in range(0, len(body), 16 * 1024):
            parser.write(body[offset : offset + 16 * 1024])
        parser.finalize()
        if not ended or file_data is None:
            raise AssetError()
        conversation_id = fields.get("conversation_id")
        if conversation_id is not None and str(UUID(conversation_id)) != conversation_id:
            raise AssetError()
        return (
            file_data,
            image_type,
            filename,
            fields.get("persistence_mode", "temporary"),
            conversation_id,
        )
    except AssetError:
        raise
    except Exception:  # noqa: BLE001 - never echo multipart payloads or parser errors
        raise AssetError() from None


@router.post("", response_model=AssetRead, status_code=201)
async def upload(request: Request) -> AssetRead:
    if request.query_params or request.headers.get("content-encoding"):
        raise AssetError()
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_UPLOAD_BODY_BYTES:
            raise AssetError("Upload body too large", 413)
        chunks.extend(chunk)

    def create() -> AssetRead:
        content, mime, filename, mode, conversation_id = parse_upload(
            bytes(chunks),
            request.headers.get("content-type", ""),
        )
        normalized = normalize_image(content, mime)
        with request.app.state.database.session_factory() as session:
            asset = AssetService(session, request.app.state.asset_storage).create(
                normalized,
                filename=filename,
                persistence_mode=mode,
                conversation_id=conversation_id,
            )
            return AssetRead.model_validate(asset)

    return await run_in_threadpool(create)


@router.get("/{asset_id}", response_model=AssetRead)
def read(asset_id: str, request: Request) -> AssetRead:
    if request.query_params:
        raise AssetError()
    with request.app.state.database.session_factory() as session:
        return AssetRead.model_validate(
            AssetService(session, request.app.state.asset_storage).get(asset_id)
        )


@router.get("/{asset_id}/content")
def content(asset_id: str, request: Request) -> Response:
    if request.query_params:
        raise AssetError()
    with request.app.state.database.session_factory() as session:
        data = AssetService(session, request.app.state.asset_storage).processing_bytes(asset_id)
    return Response(
        data,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": 'inline; filename="image.png"',
            "Cross-Origin-Resource-Policy": "same-origin",
        },
    )


@router.delete("/{asset_id}", status_code=204)
def delete_asset(asset_id: str, request: Request) -> Response:
    if request.query_params:
        raise AssetError()
    with request.app.state.database.session_factory() as session:
        AssetService(session, request.app.state.asset_storage).delete(asset_id)
    return Response(status_code=204)
