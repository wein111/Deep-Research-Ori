from __future__ import annotations

import json
from pathlib import Path
import uuid

from fastapi import FastAPI, Form, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates

from agent import resume_agent, run_agent

# 创建整个web应用
app = FastAPI(title="Deep Research API")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent))

# 规定传入json格式
class ResearchRequest(BaseModel):
    query: str
    thread_id: str = "1"

class ResumeRequest(BaseModel):
    action: str
    thread_id: str = "1"

#规定传出json格式
class ResearchResponse(BaseModel):
    final_report: str
    messages: list[str]
    paused: bool = False
    interrupts: list[str] = Field(default_factory=list)


def _state_dict_from_invoke_result(result: object) -> dict:
    """Normalize ainvoke output: dict state, or GraphOutput-style with .value dict."""
    if isinstance(result, dict):
        return result
    inner = getattr(result, "value", None)
    if isinstance(inner, dict):
        return inner
    return {}


def _interrupt_texts_from_result(result: object) -> list[str]:
    """Read payloads from interrupt(...) when the graph pauses (__interrupt__ on invoke output)."""
    raw: object | None = None
    if isinstance(result, dict):
        raw = result.get("__interrupt__")
    elif result is not None and hasattr(result, "interrupts"):
        raw = getattr(result, "interrupts")
    if not raw:
        return []
    seq = raw if isinstance(raw, (list, tuple)) else (raw,)
    out: list[str] = []
    for item in seq:
        val = getattr(item, "value", item)
        if isinstance(val, str):
            out.append(val)
        else:
            try:
                out.append(json.dumps(val, ensure_ascii=False))
            except (TypeError, ValueError):
                out.append(str(val))
    return out


def _extract_message_strings(messages: list) -> list[str]:
    contents = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            contents.append(str(content))
        else:
            contents.append(str(content))
    return contents


def _extract_message_entries(messages: list) -> list[dict[str, str]]:
    entries = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content_text = str(content)
        else:
            content_text = str(content)

        message_type = message.__class__.__name__.replace("Message", "")
        role_map = {
            "Human": "Human",
            "AI": "Assistant",
            "Ai": "Assistant",
            "Tool": "Tool",
            "System": "System",
        }
        role = role_map.get(message_type, message_type or "Message")
        entries.append({"role": role, "content": content_text})
    return entries


def _template_context(
    request: Request,
    query: str = "",
    thread_id: str = "1",
    output_text: str = "",
    error_message: str = "",
    message_entries: list[dict[str, str]] | None = None,
    interrupt_messages: list[str] | None = None,
) -> dict:
    message_entries = message_entries or []
    interrupt_messages = interrupt_messages or []
    return {
        "request": request,
        "query": query,
        "thread_id": thread_id,
        "output_text": output_text,
        "error_message": error_message,
        "message_entries": message_entries,
        "interrupt_messages": interrupt_messages,
    }


@app.get("/")
async def index(request: Request):
    thread_id = request.query_params.get("thread_id") or str(uuid.uuid4())
    return templates.TemplateResponse(
        request=request,
        name="templates.html",
        context=_template_context(request, thread_id=thread_id),
    )

# 定义健康检查接口
@app.get("/health")
async def health():
    return {"status": "ok"}

# 定义dr接口
@app.post("/research", response_model=ResearchResponse)
async def research(payload: ResearchRequest):
    try:
        result = await run_agent(payload.query, payload.thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    st = _state_dict_from_invoke_result(result)
    final_report = st.get("final_report", "")
    messages = _extract_message_strings(st.get("messages", []))

    if not final_report and messages:
        final_report = messages[-1]

    iv = _interrupt_texts_from_result(result)
    return ResearchResponse(
        final_report=final_report,
        messages=messages,
        paused=bool(iv),
        interrupts=iv,
    )

@app.post("/research/resume", response_model=ResearchResponse)
async def research_resume(payload: ResumeRequest):
    action = payload.action.strip().lower()
    if action not in {"approve", "revise"}:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'revise'")
    try:
        result = await resume_agent(action=action, thread_id=payload.thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    st = _state_dict_from_invoke_result(result)
    final_report = st.get("final_report", "")
    messages = _extract_message_strings(st.get("messages", []))
    if not final_report and messages:
        final_report = messages[-1]
    iv = _interrupt_texts_from_result(result)
    return ResearchResponse(
        final_report=final_report,
        messages=messages,
        paused=bool(iv),
        interrupts=iv,
    )


@app.post("/research-form")
async def research_form(
    request: Request,
    query: str = Form(...),
    thread_id: str = Form("1"),
):
    try:
        result = await run_agent(query, thread_id)
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="templates.html",
            context=_template_context(
                request,
                query=query,
                thread_id=thread_id,
                error_message=str(exc),
            ),
            status_code=500,
        )

    st = _state_dict_from_invoke_result(result)
    final_report = st.get("final_report", "")
    messages = _extract_message_strings(st.get("messages", []))
    message_entries = _extract_message_entries(st.get("messages", []))
    latest_assistant_message = next(
        (entry["content"] for entry in reversed(message_entries) if entry["role"] == "Assistant"),
        "",
    )
    if not final_report and messages:
        final_report = messages[-1]
    interrupt_messages = _interrupt_texts_from_result(result)

    return templates.TemplateResponse(
        request=request,
        name="templates.html",
        context=_template_context(
            request,
            query=query,
            thread_id=thread_id,
            output_text=latest_assistant_message or final_report,
            message_entries=message_entries,
            interrupt_messages=interrupt_messages,
        ),
    )


@app.post("/research-form-resume")
async def research_form_resume(
    request: Request,
    action: str = Form(...),
    thread_id: str = Form("1"),
):
    normalized_action = action.strip().lower()
    if normalized_action not in {"approve", "revise"}:
        return templates.TemplateResponse(
            request=request,
            name="templates.html",
            context=_template_context(
                request,
                thread_id=thread_id,
                error_message="action must be approve or revise",
            ),
            status_code=400,
        )
    try:
        result = await resume_agent(
            action=normalized_action,
            thread_id=thread_id,
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="templates.html",
            context=_template_context(
                request,
                thread_id=thread_id,
                error_message=str(exc),
            ),
            status_code=500,
        )

    st = _state_dict_from_invoke_result(result)
    final_report = st.get("final_report", "")
    messages = _extract_message_strings(st.get("messages", []))
    message_entries = _extract_message_entries(st.get("messages", []))
    latest_assistant_message = next(
        (entry["content"] for entry in reversed(message_entries) if entry["role"] == "Assistant"),
        "",
    )
    if not final_report and messages:
        final_report = messages[-1]
    interrupt_messages = _interrupt_texts_from_result(result)

    return templates.TemplateResponse(
        request=request,
        name="templates.html",
        context=_template_context(
            request,
            thread_id=thread_id,
            output_text=latest_assistant_message or final_report,
            message_entries=message_entries,
            interrupt_messages=interrupt_messages,
        ),
    )
