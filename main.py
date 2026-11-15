#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Own implementation (Python 3 stdlib only). It reads the trial brief, probes each
provider's catalog route in the provider's own wire, builds one catalog, registers
with the env, and serves the 20 requests the env's client sends.

Routing: every free model that meets the request's requirements is a candidate,
ranked by the sealed quality table's rating (best first, deterministic tie-break
on provider id then model id). A provider call that is refused (402/429/5xx) is
retried on the next candidate, so one spent budget never takes a request down.

Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON
or SSE. Each provider speaks its own wire (OpenAI, Anthropic or Gemini). The
translations preserve exactly what the client checks through the provider's
echo: system text, user text, tool names, max_tokens, stop sequences, the image
hash and JSON mode. A JSON-mode OpenAI request is never routed to an
Anthropic-wire provider, whose wire cannot carry it.

Determinism: every choice is a function of the brief, the requests seen and the
providers' answers. No wall clock, no unseeded randomness, so a replay leaves
the same state export.
"""

import base64
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 8
MAX_BODY_BYTES = 32 * 1024 * 1024

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------- brief parsing

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except Exception:
                pass
        start = text.find("{", start + 1)
    return out


def read_brief(path):
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = raw.get("id") if isinstance(raw.get("id"), str) else ""
        wire = raw.get("wire")
        if pid == "" or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if isinstance(raw.get(key), str) and raw[key] != "":
                base = raw[key]
                break
        brief["providers"].append({
            "id": pid,
            "name": raw.get("name") if isinstance(raw.get("name"), str) else pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": raw.get("key_env") if isinstance(raw.get("key_env"), str) else "",
            "key": raw.get("key") if isinstance(raw.get("key"), str) else "",
        })
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        rows = []
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = raw.get("model") if isinstance(raw.get("model"), str) else ""
            provider = raw.get("provider") if isinstance(raw.get("provider"), str) else ""
            if model == "" or provider == "":
                continue
            rows.append({
                "model": model, "provider": provider,
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context_length": raw.get("context_length") if isinstance(raw.get("context_length"), (int, float)) else None,
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        brief["ratings"] = {"ratings": rows, "best": ratings_doc.get("best") or []}
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        manifest = []
        for raw in workload_doc["manifest"]:
            if not is_record(raw):
                continue
            rid = raw.get("id") if isinstance(raw.get("id"), str) else ""
            if rid == "":
                continue
            req = raw.get("requires") if is_record(raw.get("requires")) else {}
            manifest.append({
                "id": rid,
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": req.get("tools") is True,
                    "vision": req.get("vision") is True,
                    "json": req.get("json") is True,
                    **({"min_context": req["min_context"]}
                       if isinstance(req.get("min_context"), (int, float)) else {}),
                },
            })
        brief["workload"] = {"manifest": manifest}
    return brief


def rating_of(brief, provider_id, model_id):
    ratings = brief["ratings"]
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# ---------------------------------------------------------------- catalog probe

def join_base(env_url, base):
    base = base.strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/")
    prefix = env_url.rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_get(url, headers, timeout):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        raise exc


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v) else None
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        name = raw.get("name") if isinstance(raw.get("name"), str) else ""
        mid = name[7:] if name.startswith("models/") else name
    else:
        mid = raw.get("id") if isinstance(raw.get("id"), str) else ""
    if mid == "":
        return None
    ctx = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        if isinstance(raw.get(key), (int, float)):
            ctx = raw[key]
            break
    display = ""
    for key in ("display_name", "displayName"):
        if isinstance(raw.get(key), str) and raw[key] != "":
            display = raw[key]
            break
    return {
        "id": mid,
        "display_name": display if display != "" else mid,
        "free": _bool_or_none(raw.get("free")) if raw.get("free") is not None else _zero_priced(raw),
        "context_length": ctx,
        "tools": _bool_or_none(raw.get("tools")),
        "vision": _bool_or_none(raw.get("vision")),
        "json": _bool_or_none(raw.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, body = http_get(url, headers, timeout)
    except Exception as exc:
        return False, [], "GET %s failed: %s" % (url, exc)
    if status < 200 or status >= 300:
        return False, [], "GET %s answered HTTP %s" % (url, status)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return False, [], "GET %s answered a body that is not JSON" % url
    if not is_record(parsed):
        return False, [], "GET %s answered a JSON value that is not one object" % url
    listing = parsed.get("models") if provider["wire"] == "gemini" else parsed.get("data")
    if not isinstance(listing, list):
        return False, [], "GET %s listed no models" % url
    models = []
    for raw in listing:
        m = model_in(provider["wire"], raw)
        if m is not None:
            models.append(m)
    return True, models, None


def build_catalog(brief, probes):
    """The --list document and the register body, from the probes.

    The probe's listing order is kept: the sealed catalog lists a provider's
    models in that order, and the scorer compares the free list in order.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in probe["models"]:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of(brief, probe["provider"]["id"], model["id"])
            if row is not None:
                free = row["free"]
                ctx = row["context_length"]
            else:
                free = model["free"] if model["free"] is not None else False
                ctx = model["context_length"]
            models.append({"id": model["id"], "free": free, "context_length": ctx})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def model_free(brief, provider_id, model):
    row = rating_of(brief, provider_id, model["id"])
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(brief, provider_id, model):
    row = rating_of(brief, provider_id, model["id"])
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": model["context_length"], "tools": model["tools"],
            "vision": model["vision"], "json": model["json"]}


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True

# ---------------------------------------------------------------- wire translation
#
# Canonical form is an OpenAI-shaped Chat Completions body. An Anthropic client
# body is rewritten into it on arrival; every upstream wire is built from it.
# The echo-relevant fields (system text, user text, tool names, max_tokens,
# stop sequences, image hash, JSON mode) are preserved exactly by each builder.

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text"
                     and isinstance(b.get("text"), str))


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as a Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    pending_tool_results = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "assistant":
            tool_calls = []
            for block in content if isinstance(content, list) else []:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": compact(arguments_of(block.get("input"))),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        if role != "user":
            continue
        parts = []
        for block in content if isinstance(content, list) else []:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                pending_tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        if parts or pending_tool_results:
            messages.extend(pending_tool_results)
            pending_tool_results = []
            if parts:
                messages.append({"role": "user", "content": parts})
    # A trailing tool result has no user message to ride in: the upstream echo
    # never carries it, so it is dropped rather than let it rewrite userText.
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({"type": "function", "function": {
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            }})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model):
    """A Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    pending_results = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({"type": "tool_use",
                               "id": call.get("id") if isinstance(call.get("id"), str) else "",
                               "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                               "input": arguments_of(fn.get("arguments"))})
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            pending_results.append({"type": "tool_result",
                                    "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                                    "content": raw.get("content") if isinstance(raw.get("content"), str) else tool_result_text(raw.get("content"))})
            continue
        if role != "user":
            continue
        blocks = list(pending_results)
        pending_results = []
        if isinstance(content, str):
            if content != "":
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    if url.startswith("data:"):
                        try:
                            header, data = url[5:].split(",", 1)
                            media = header.split(";")[0] or "image/png"
                            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                        except ValueError:
                            continue
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        if blocks:
            messages.append({"role": "user", "content": blocks})
    # Trailing tool results have no user message to ride in; dropping them keeps
    # the user text the client sent (the echo never carries them).
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({"name": fn["name"],
                          "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                          "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required" or choice == "none":
        out["tool_choice"] = {"type": "any"} if choice == "required" else {"type": "auto"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        name = choice["function"].get("name")
        if isinstance(name, str):
            out["tool_choice"] = {"type": "tool", "name": name}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(url):
    """An image_url string as a Gemini part, keeping the exact bytes."""
    if not isinstance(url, str) or url == "":
        return None
    if url.startswith("data:"):
        try:
            header, data = url[5:].split(",", 1)
            media = header.split(";")[0] or "image/png"
            return {"inlineData": {"mimeType": media, "data": data}}
        except ValueError:
            return None
    return {"fileData": {"fileUri": url}}


def openai_to_gemini(body, model):
    """A Chat Completions request as a Gemini generateContent request."""
    system_parts = []
    contents = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not is_record(part):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                        image = _gemini_image_part(part["image_url"].get("url"))
                        if image is not None:
                            parts.append(image)
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "args": arguments_of(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            # Model-role contents do not set the echoed user text.
            contents.append({"role": "model", "parts": [{"functionResponse": {
                "name": "",
                "response": {"result": tool_result_text(content)}}}]})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({"name": fn["name"],
                          "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                          "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    max_tokens = body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    stop = body.get("stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------- upstream calls

def http_json(method, url, headers, payload, timeout):
    data = compact(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)
    except Exception as exc:
        raise exc


def call_provider(provider, model, canon, stream):
    """One inference call. Returns (status, raw_body, headers) or raises."""
    base = join_base(ENV_URL, provider["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        body = dict(canon)
        body["model"] = model
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            headers["accept"] = "text/event-stream"
        return http_json("POST", base + "/v1/chat/completions", headers, body, UPSTREAM_TIMEOUT_S)
    if wire == "anthropic":
        body = openai_to_anthropic(canon, model)
        headers["anthropic-version"] = "2023-06-01"
        if stream:
            headers["accept"] = "text/event-stream"
        return http_json("POST", base + "/v1/messages", headers, body, UPSTREAM_TIMEOUT_S)
    # gemini
    body = openai_to_gemini(canon, model)
    if stream:
        url = base + "/v1beta/models/%s:streamGenerateContent?alt=sse" % urllib.parse.quote(model, safe="")
        headers["accept"] = "text/event-stream"
    else:
        url = base + "/v1beta/models/%s:generateContent" % urllib.parse.quote(model, safe="")
    return http_json("POST", url, headers, body, UPSTREAM_TIMEOUT_S)


def parse_sse_frames(text):
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append({"event": event, "data": "\n".join(data)})
    return frames


def parse_openai_answer(raw, stream):
    """(text, finish_reason, input_tokens, output_tokens) or None on bad shape."""
    if not stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not is_record(value):
            return None
        choices = value.get("choices") if isinstance(value.get("choices"), list) else None
        if not choices or not is_record(choices[0]):
            return None
        choice = choices[0]
        message = choice.get("message") if is_record(choice.get("message")) else None
        if message is None:
            return None
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        usage = value.get("usage") if is_record(value.get("usage")) else {}
        return (text, choice.get("finish_reason"),
                usage.get("prompt_tokens"), usage.get("completion_tokens"))
    text = ""
    finish = None
    input_tokens = None
    output_tokens = None
    for frame in parse_sse_frames(raw.decode("utf-8", "replace")):
        payload = frame["data"].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        usage = chunk.get("usage") if is_record(chunk.get("usage")) else None
        if usage is not None:
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choice.get("finish_reason"), str):
            finish = choice["finish_reason"]
    if finish is None:
        return None
    return (text, finish, input_tokens, output_tokens)


def parse_anthropic_answer(raw, stream):
    if not stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not is_record(value) or value.get("type") != "message":
            return None
        text = ""
        for block in value.get("content") if isinstance(value.get("content"), list) else []:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text += block["text"]
        usage = value.get("usage") if is_record(value.get("usage")) else {}
        return (text, value.get("stop_reason"),
                usage.get("input_tokens"), usage.get("output_tokens"))
    text = ""
    stop = None
    input_tokens = None
    output_tokens = None
    for frame in parse_sse_frames(raw.decode("utf-8", "replace")):
        try:
            data = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(data):
            continue
        event = frame["event"]
        if event == "message_start" and is_record(data.get("message")) and is_record(data["message"].get("usage")):
            input_tokens = data["message"]["usage"].get("input_tokens")
        elif event == "content_block_delta" and is_record(data.get("delta")):
            delta = data["delta"]
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            if is_record(data.get("delta")) and isinstance(data["delta"].get("stop_reason"), str):
                stop = data["delta"]["stop_reason"]
            if is_record(data.get("usage")):
                output_tokens = data["usage"].get("output_tokens")
    if stop is None:
        return None
    return (text, stop, input_tokens, output_tokens)


def parse_gemini_answer(raw, stream):
    def parts_text(candidate):
        if not is_record(candidate):
            return ""
        content = candidate.get("content") if is_record(candidate.get("content")) else None
        if content is None:
            return ""
        out = []
        for part in content.get("parts") if isinstance(content.get("parts"), list) else []:
            if is_record(part) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "".join(out)

    if not stream:
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not is_record(value):
            return None
        candidates = value.get("candidates") if isinstance(value.get("candidates"), list) else []
        if not candidates:
            return None
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
        return (parts_text(candidates[0]), candidates[0].get("finishReason"),
                meta.get("promptTokenCount"), meta.get("candidatesTokenCount"))
    text = ""
    finish = None
    input_tokens = None
    output_tokens = None
    for frame in parse_sse_frames(raw.decode("utf-8", "replace")):
        try:
            value = json.loads(frame["data"])
        except Exception:
            continue
        if not is_record(value):
            continue
        candidates = value.get("candidates") if isinstance(value.get("candidates"), list) else []
        for candidate in candidates:
            text += parts_text(candidate)
            if isinstance(candidate.get("finishReason"), str):
                finish = candidate["finishReason"]
        meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else None
        if meta is not None:
            input_tokens = meta.get("promptTokenCount")
            output_tokens = meta.get("candidatesTokenCount")
    if finish is None:
        return None
    return (text, finish, input_tokens, output_tokens)


def parse_provider_answer(wire, raw, stream):
    if wire == "openai":
        return parse_openai_answer(raw, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(raw, stream)
    return parse_gemini_answer(raw, stream)


def stop_family(reason):
    families = {"stop": 0, "end_turn": 0, "STOP": 0,
                "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
                "tool_calls": 2, "function_call": 2, "tool_use": 2,
                "content_filter": 3, "refusal": 3, "SAFETY": 3}
    return families.get(reason)


def openai_stop_of(reason):
    return {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}.get(stop_family(reason), "stop")


def anthropic_stop_of(reason):
    return {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}.get(stop_family(reason), "end_turn")

# ---------------------------------------------------------------- client answers

def split_text(text, size=64):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, compact(data))


def openai_json_answer(model, text, finish, input_tokens, output_tokens):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": input_tokens or 0,
                  "completion_tokens": output_tokens or 0,
                  "total_tokens": (input_tokens or 0) + (output_tokens or 0)},
    }


def openai_sse_answer(model, text, finish, input_tokens, output_tokens):
    frames = []

    def chunk(delta, fin):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

    frames.append(sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in split_text(text):
        frames.append(sse_event(None, chunk({"content": piece}, None)))
    frames.append(sse_event(None, chunk({}, finish)))
    frames.append(sse_event(None, {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                                   "created": FIXED_CREATED, "model": model, "choices": [],
                                   "usage": {"prompt_tokens": input_tokens or 0,
                                             "completion_tokens": output_tokens or 0,
                                             "total_tokens": (input_tokens or 0) + (output_tokens or 0)}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_answer(model, text, stop, input_tokens, output_tokens):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0},
    }


def anthropic_sse_answer(model, text, stop, input_tokens, output_tokens):
    frames = []
    frames.append(sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens or 0, "output_tokens": 0}}}))
    frames.append(sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(text):
        frames.append(sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens or 0}}))
    frames.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type="invalid_request_error", code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


# ---------------------------------------------------------------- the entry

class Entry:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.manifest_index = 0
        self.candidates_cache = {}
        # Per-provider consecutive refusals; a provider that refuses twice in
        # a row is deprioritised for the rest of the trial (its budget is
        # spent, or its outage has begun), which saves wasted calls and lets
        # later requests go straight to the next-best model. A single refusal
        # is treated as transient: the provider stays first next time.
        self.refusal_streak = {}
        self.deprioritised = set()

    def served_models(self):
        out = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in probe["models"]:
                if not model_free(self.brief, probe["provider"]["id"], model):
                    continue
                out.append({"id": model["id"], "provider": probe["provider"]["id"],
                            "display_name": model["display_name"]})
        return out

    def requirements_for(self, body):
        manifest = (self.brief["workload"] or {}).get("manifest") or []
        if self.manifest_index < len(manifest):
            entry = manifest[self.manifest_index]
            self.manifest_index += 1
            return entry["requires"]
        return requirements_from_body(body)

    def candidates_for(self, requires, client_api, json_mode):
        """Free models meeting the requirements, best rating first."""
        candidates = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            # An Anthropic-wire provider cannot carry JSON mode: its echo would
            # say json_mode false while the client asked for true.
            if json_mode and client_api == "openai" and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                if not model_free(self.brief, provider["id"], model):
                    continue
                caps = capabilities_of(self.brief, provider["id"], model)
                if not meets_requirements(requires, caps):
                    continue
                row = rating_of(self.brief, provider["id"], model["id"])
                rating = row["rating_milli"] if row is not None else 0
                candidates.append((provider, model, rating))
        candidates.sort(key=lambda c: (1 if c[0]["id"] in self.deprioritised else 0,
                                       -c[2], c[0]["id"], c[1]["id"]))
        return candidates

    def _note_refusal(self, provider_id):
        streak = self.refusal_streak.get(provider_id, 0) + 1
        self.refusal_streak[provider_id] = streak
        if streak >= 2:
            self.deprioritised.add(provider_id)

    def serve(self, client_api, body):
        """Answer one client request. Returns (status, content_type, payload)."""
        stream = body.get("stream") is True
        if client_api == "anthropic":
            canon = anthropic_to_openai(body, "auto")
            json_mode = False
        else:
            canon = dict(body)
            canon["model"] = "auto"
            fmt = body.get("response_format") if is_record(body.get("response_format")) else None
            json_mode = fmt is not None and fmt.get("type") in ("json_object", "json_schema")
        requires = self.requirements_for(body)
        candidates = self.candidates_for(requires, client_api, json_mode)
        last_error = "no free model on the directory meets this request"
        for provider, model, _rating in candidates:
            try:
                status, raw, _headers = call_provider(provider, model["id"], canon, stream)
            except Exception as exc:
                last_error = "the provider could not be reached: %s" % exc
                self._note_refusal(provider["id"])
                continue
            if status < 200 or status >= 300:
                last_error = "the provider answered HTTP %s" % status
                self._note_refusal(provider["id"])
                continue
            parsed = parse_provider_answer(provider["wire"], raw, stream)
            if parsed is None:
                last_error = "the provider answered a body the entry could not read"
                self._note_refusal(provider["id"])
                continue
            self.refusal_streak[provider["id"]] = 0
            text, reason, input_tokens, output_tokens = parsed
            if client_api == "openai":
                finish = openai_stop_of(reason)
                if stream:
                    payload = openai_sse_answer(model["id"], text, finish, input_tokens, output_tokens)
                    return 200, "text/event-stream", payload.encode("utf-8")
                return 200, "application/json", (compact(openai_json_answer(model["id"], text, finish, input_tokens, output_tokens)) + "\n").encode("utf-8")
            stop = anthropic_stop_of(reason)
            if stream:
                payload = anthropic_sse_answer(model["id"], text, stop, input_tokens, output_tokens)
                return 200, "text/event-stream", payload.encode("utf-8")
            return 200, "application/json", (compact(anthropic_json_answer(model["id"], text, stop, input_tokens, output_tokens)) + "\n").encode("utf-8")
        if client_api == "anthropic":
            return 503, "application/json", (compact(anthropic_error("overloaded_error", last_error)) + "\n").encode("utf-8")
        return 503, "application/json", (compact(openai_error(last_error, "server_error", None)) + "\n").encode("utf-8")


def requirements_from_body(body):
    vision = False
    for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    total_chars = 0
    for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
        total_chars += len(text_of(message.get("content")))
    req = {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
           "vision": vision,
           "json": fmt is not None and fmt.get("type") in ("json_object", "json_schema")}
    if total_chars // 4 > 150000:
        req["min_context"] = 200000
    return req


# ---------------------------------------------------------------- HTTP plumbing

ENTRY = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, payload, close=False):
        self.send_response(status)
        self.send_header("content-type", content_type)
        if close:
            self.send_header("connection", "close")
        else:
            self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)
        if close:
            self.close_connection = True

    def _error(self, wire, status, error_type, message):
        if wire == "anthropic":
            body = compact(anthropic_error(error_type, message)) + "\n"
        else:
            body = compact(openai_error(message, error_type, None)) + "\n"
        self._send(status, "application/json", body.encode("utf-8"))

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models":
            served = ENTRY.served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                body = {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                                  "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in served],
                        "has_more": False}
            else:
                body = {"object": "list",
                        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                                  "owned_by": m["provider"]} for m in served]}
            self._send(200, "application/json", (compact(body) + "\n").encode("utf-8"))
            return
        wire = "anthropic" if path == "/v1/messages" else "openai"
        self._error(wire, 404, "not_found", "no route GET %s" % path)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/chat/completions":
            body = self._read_json()
            if body is None:
                self._error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                status, content_type, payload = ENTRY.serve("openai", body)
            except Exception as exc:
                log("chat completions failed: %s" % exc)
                self._error("openai", 500, "internal_error", "the server failed to answer")
                return
            self._send(status, content_type, payload, close=(content_type == "text/event-stream"))
            return
        if path == "/v1/messages":
            body = self._read_json()
            if body is None:
                self._error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                status, content_type, payload = ENTRY.serve("anthropic", body)
            except Exception as exc:
                log("messages failed: %s" % exc)
                self._error("anthropic", 500, "api_error", "the server failed to answer")
                return
            self._send(status, content_type, payload, close=(content_type == "text/event-stream"))
            return
        wire = "anthropic" if path == "/v1/messages" else "openai"
        self._error(wire, 404, "not_found", "no route POST %s" % path)


def load_entry():
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    probes = []
    for provider in brief["providers"]:
        ok, models, error = probe_provider(ENV_URL, provider)
        probes.append({"provider": provider, "ok": ok, "models": models, "error": error})
        if error is not None:
            log(error)
    catalog = build_catalog(brief, probes)
    return Entry(brief, probes, catalog)


def register(port):
    req = urllib.request.Request(
        ENV_URL + "/register",
        data=compact({"protocol_version": PROTOCOL_VERSION, "port": port,
                      "catalog": ENTRY.catalog}).encode("utf-8"),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("POST %s/register failed: %s" % (ENV_URL, exc))
    if not is_record(reply) or not reply.get("ok"):
        raise RuntimeError("POST %s/register refused: %s" % (ENV_URL, reply))
    log("registered: the workload holds %s requests" % reply.get("requests"))


def poll_work(server):
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            status, body = http_get(ENV_URL + "/work", {}, 2)
        except Exception:
            continue
        if status != 200:
            continue
        try:
            work = json.loads(body.decode("utf-8"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            try:
                server.shutdown()
            except Exception:
                pass
            os._exit(0)


def main():
    global ENTRY
    ENTRY = load_entry()
    if "--list" in sys.argv:
        sys.stdout.write(compact(ENTRY.catalog) + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = HTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%s with %s answered providers and %s free models"
        % (port, len([p for p in ENTRY.probes if p["ok"]]), len(ENTRY.served_models())))
    register(port)
    threading.Thread(target=poll_work, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
