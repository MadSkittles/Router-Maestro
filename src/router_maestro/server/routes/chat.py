"""Chat completions route."""

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from router_maestro.providers import ChatRequest, Message, ProviderError
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from router_maestro.utils import get_logger

logger = get_logger("server.routes.chat")

router = APIRouter()


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    logger.info(
        "Received chat completion request: model=%s, stream=%s",
        request.model,
        request.stream,
    )
    model_router = get_router()

    # Convert to internal format
    chat_request = ChatRequest(
        model=request.model,
        messages=[Message(role=m.role, content=m.content) for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    if request.stream:
        return StreamingResponse(
            stream_response(model_router, chat_request),
            media_type="text/event-stream",
        )

    try:
        response, provider_name = await model_router.chat_completion(chat_request)

        usage = None
        if response.usage:
            usage = ChatCompletionUsage(
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.content),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=usage,
        )
    except ProviderError as e:
        logger.error("Chat completion request failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))


async def stream_response(model_router: Router, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    try:
        stream, provider_name = await model_router.chat_completion_stream(request)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Send initial chunk with role
        initial_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        async for chunk in stream:
            if chunk.content:
                chunk_response = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=chunk.content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"

            if chunk.finish_reason:
                final_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=chunk.finish_reason,
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except ProviderError as e:
        error_data = {"error": {"message": str(e), "type": "provider_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
