import asyncio
import base64
import json
import os
import re
import secrets
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from starlette.applications import Starlette
from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"[hermes-golden] Generated admin password: {ADMIN_PASSWORD}", flush=True)

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
ENV_FILE = HERMES_HOME / ".env"
CONFIG_FILE = HERMES_HOME / "config.yaml"
DATA_ROOT = Path(os.environ.get("HERMES_DATA_ROOT", "/data"))
WORKSPACE_DIR = Path(os.environ.get("HERMES_WORKSPACE_DIR", str(DATA_ROOT / "workspace")))
INTERNAL_API_BASE = f"http://127.0.0.1:{os.environ.get('API_SERVER_PORT', '8642')}"
STATE_ROOT = Path(os.environ.get("HERMES_STATE_DIR", str(DATA_ROOT / ".hermes-cloud")))
SESSIONS_FILE = STATE_ROOT / "sessions.json"
JOBS_FILE = STATE_ROOT / "jobs.json"
MEMORY_ROOT = WORKSPACE_DIR / "memory"
ROOT_MEMORY_FILE = WORKSPACE_DIR / "MEMORY.md"

UI_ENV_KEYS = [
    "HERMES_MODEL",
    "HERMES_INFERENCE_PROVIDER",
    "HERMES_MODEL_BASE_URL",
    "HERMES_MODEL_API_KEY",
    "OPENROUTER_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "API_SERVER_KEY",
]
SECRET_KEYS = {"HERMES_MODEL_API_KEY", "OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN", "API_SERVER_KEY"}

STATE_ROOT.mkdir(parents=True, exist_ok=True)
MEMORY_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_model_id(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return "moonshotai/kimi-k2.5"
    return value.removeprefix("openrouter/")


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
      return {}
    result = {}
    for line in path.read_text().splitlines():
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, _, value = line.partition("=")
      value = value.strip()
      if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
      result[key.strip()] = value
    return result


def write_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(values.items()) if value]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_config(env_values: dict[str, str]) -> None:
    model = normalize_model_id(env_values.get("HERMES_MODEL", "moonshotai/kimi-k2.5"))
    provider = env_values.get("HERMES_INFERENCE_PROVIDER", "auto")
    base_url = env_values.get("HERMES_MODEL_BASE_URL", "").strip()
    api_key = env_values.get("HERMES_MODEL_API_KEY", "").strip()
    api_server_key = env_values.get("API_SERVER_KEY", "").strip()
    lines = [
        "model:",
        f"  default: \"{model}\"",
        f"  provider: \"{provider}\"",
    ]
    if base_url:
        lines.append(f"  base_url: \"{base_url}\"")
    if api_key:
        lines.append(f"  api_key: \"{api_key}\"")
    lines.extend([
        "platforms:",
        "  api_server:",
        "    enabled: true",
        f"    host: \"{os.environ.get('API_SERVER_HOST', '127.0.0.1')}\"",
        f"    port: {int(os.environ.get('API_SERVER_PORT', '8642'))}",
    ])
    if api_server_key:
        lines.append(f"    key: \"{api_server_key}\"")
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def build_synthetic_config(env_values: dict[str, str]) -> dict:
    def split_csv(value: str | None) -> list[str]:
        return [item.strip() for item in (value or "").split(",") if item.strip()]

    return {
        "env": env_values,
        "model": {
            "default": normalize_model_id(env_values.get("HERMES_MODEL") or env_values.get("LLM_MODEL")),
            "provider": env_values.get("HERMES_INFERENCE_PROVIDER", "auto"),
            **({"base_url": env_values["HERMES_MODEL_BASE_URL"]} if env_values.get("HERMES_MODEL_BASE_URL") else {}),
            **({"api_key": env_values["HERMES_MODEL_API_KEY"]} if env_values.get("HERMES_MODEL_API_KEY") else {}),
        },
        "agents": {
            "defaults": {
                "model": {
                    "primary": f"openrouter/{normalize_model_id(env_values.get('HERMES_MODEL') or env_values.get('LLM_MODEL'))}",
                },
                "workspace": str(WORKSPACE_DIR),
            },
        },
        "channels": {
            "telegram": {
                "enabled": bool(env_values.get("TELEGRAM_BOT_TOKEN")),
                "botToken": env_values.get("TELEGRAM_BOT_TOKEN", ""),
                "dmPolicy": env_values.get("TELEGRAM_DM_POLICY") or ("open" if env_values.get("TELEGRAM_ALLOW_ALL_USERS") else "pairing"),
                "groupPolicy": env_values.get("TELEGRAM_GROUP_POLICY", "allowlist"),
                "allowFrom": split_csv(env_values.get("TELEGRAM_ALLOWED_USERS")),
                "groupAllowFrom": split_csv(env_values.get("TELEGRAM_GROUP_ALLOWED")),
                "replyToMode": env_values.get("TELEGRAM_REPLY_TO_MODE", "off"),
            },
            "discord": {
                "enabled": bool(env_values.get("DISCORD_BOT_TOKEN")),
                "botToken": env_values.get("DISCORD_BOT_TOKEN", ""),
                "dmPolicy": env_values.get("DISCORD_DM_POLICY") or ("open" if env_values.get("DISCORD_ALLOW_ALL_USERS") else "pairing"),
                "groupPolicy": env_values.get("DISCORD_GROUP_POLICY", "allowlist"),
                "allowFrom": split_csv(env_values.get("DISCORD_ALLOWED_USERS")),
                "groupAllowFrom": [],
            },
            "slack": {
                "enabled": bool(env_values.get("SLACK_BOT_TOKEN")),
                "botToken": env_values.get("SLACK_BOT_TOKEN", ""),
                "appToken": env_values.get("SLACK_APP_TOKEN", ""),
                "dmPolicy": env_values.get("SLACK_DM_POLICY") or ("open" if env_values.get("SLACK_ALLOW_ALL_USERS") else "pairing"),
                "groupPolicy": env_values.get("SLACK_GROUP_POLICY", "allowlist"),
                "allowFrom": split_csv(env_values.get("SLACK_ALLOWED_USERS")),
                "groupAllowFrom": [],
            },
            "whatsapp": {
                "enabled": (env_values.get("WHATSAPP_ENABLED", "").lower() in {"1", "true", "yes"}),
                "dmPolicy": env_values.get("WHATSAPP_DM_POLICY") or ("open" if env_values.get("WHATSAPP_ALLOW_ALL_USERS") else "pairing"),
                "groupPolicy": env_values.get("WHATSAPP_GROUP_POLICY", "open"),
                "allowFrom": split_csv(env_values.get("WHATSAPP_ALLOWED_USERS")),
                "groupAllowFrom": [],
            },
        },
        "gateway": {
            "controlUi": {
                "allowInsecureAuth": True,
            },
            "http": {
                "endpoints": {
                    "responses": {
                        "enabled": True,
                    },
                },
            },
        },
        "skills": {
            "entries": {},
        },
    }


def compile_env_from_config(config: dict, existing_env: dict[str, str]) -> dict[str, str]:
    next_env = dict(existing_env)
    env_block = config.get("env") if isinstance(config.get("env"), dict) else {}
    for key, value in env_block.items():
        next_env[str(key)] = "" if value is None else str(value)

    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    agents = config.get("agents") if isinstance(config.get("agents"), dict) else {}
    defaults = agents.get("defaults") if isinstance(agents.get("defaults"), dict) else {}
    defaults_model = defaults.get("model") if isinstance(defaults.get("model"), dict) else {}
    primary_model = defaults_model.get("primary")
    if isinstance(primary_model, str) and primary_model.strip():
        next_env["HERMES_MODEL"] = normalize_model_id(primary_model)
    if isinstance(model.get("default"), str) and model["default"].strip():
        next_env["HERMES_MODEL"] = normalize_model_id(model["default"])
    if isinstance(model.get("provider"), str) and model["provider"].strip():
        next_env["HERMES_INFERENCE_PROVIDER"] = model["provider"].strip()
    if "base_url" in model:
        next_env["HERMES_MODEL_BASE_URL"] = str(model.get("base_url") or "")
    if "api_key" in model:
        next_env["HERMES_MODEL_API_KEY"] = str(model.get("api_key") or "")

    channels = config.get("channels") if isinstance(config.get("channels"), dict) else {}

    telegram = channels.get("telegram") if isinstance(channels.get("telegram"), dict) else {}
    next_env["TELEGRAM_BOT_TOKEN"] = str(telegram.get("botToken") or "")
    next_env["TELEGRAM_DM_POLICY"] = str(telegram.get("dmPolicy") or "")
    next_env["TELEGRAM_GROUP_POLICY"] = str(telegram.get("groupPolicy") or "")
    next_env["TELEGRAM_REPLY_TO_MODE"] = str(telegram.get("replyToMode") or "")
    next_env["TELEGRAM_ALLOWED_USERS"] = ",".join(str(v).strip() for v in telegram.get("allowFrom", []) if str(v).strip())
    next_env["TELEGRAM_GROUP_ALLOWED"] = ",".join(str(v).strip() for v in telegram.get("groupAllowFrom", []) if str(v).strip())

    discord = channels.get("discord") if isinstance(channels.get("discord"), dict) else {}
    next_env["DISCORD_BOT_TOKEN"] = str(discord.get("botToken") or "")
    next_env["DISCORD_DM_POLICY"] = str(discord.get("dmPolicy") or "")
    next_env["DISCORD_GROUP_POLICY"] = str(discord.get("groupPolicy") or "")
    next_env["DISCORD_ALLOWED_USERS"] = ",".join(str(v).strip() for v in discord.get("allowFrom", []) if str(v).strip())

    slack = channels.get("slack") if isinstance(channels.get("slack"), dict) else {}
    next_env["SLACK_BOT_TOKEN"] = str(slack.get("botToken") or "")
    next_env["SLACK_APP_TOKEN"] = str(slack.get("appToken") or "")
    next_env["SLACK_DM_POLICY"] = str(slack.get("dmPolicy") or "")
    next_env["SLACK_GROUP_POLICY"] = str(slack.get("groupPolicy") or "")
    next_env["SLACK_ALLOWED_USERS"] = ",".join(str(v).strip() for v in slack.get("allowFrom", []) if str(v).strip())

    whatsapp = channels.get("whatsapp") if isinstance(channels.get("whatsapp"), dict) else {}
    next_env["WHATSAPP_ENABLED"] = "true" if bool(whatsapp.get("enabled")) else "false"
    next_env["WHATSAPP_DM_POLICY"] = str(whatsapp.get("dmPolicy") or "")
    next_env["WHATSAPP_GROUP_POLICY"] = str(whatsapp.get("groupPolicy") or "")
    next_env["WHATSAPP_ALLOWED_USERS"] = ",".join(str(v).strip() for v in whatsapp.get("allowFrom", []) if str(v).strip())

    return next_env


def resolve_workspace_path(raw_path: str) -> Path:
    cleaned = raw_path.strip().lstrip("/")
    target = (WORKSPACE_DIR / cleaned).resolve()
    workspace_root = WORKSPACE_DIR.resolve()
    if workspace_root not in target.parents and target != workspace_root:
        raise ValueError("Path escapes workspace root")
    return target


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_file(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_sessions() -> list[dict]:
    payload = read_json_file(SESSIONS_FILE, {"sessions": []})
    sessions = payload.get("sessions")
    return sessions if isinstance(sessions, list) else []


def save_sessions(sessions: list[dict]) -> None:
    write_json_file(SESSIONS_FILE, {"sessions": sessions})


def load_jobs() -> list[dict]:
    payload = read_json_file(JOBS_FILE, {"jobs": []})
    jobs = payload.get("jobs")
    return jobs if isinstance(jobs, list) else []


def save_jobs(jobs: list[dict]) -> None:
    write_json_file(JOBS_FILE, {"jobs": jobs})


def list_memory_files() -> list[dict]:
    files = []
    if ROOT_MEMORY_FILE.exists():
        stat = ROOT_MEMORY_FILE.stat()
        files.append({
            "path": "MEMORY.md",
            "name": "MEMORY.md",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    for target in sorted(MEMORY_ROOT.rglob("*.md")):
        stat = target.stat()
        relative = target.relative_to(WORKSPACE_DIR).as_posix()
        files.append({
            "path": relative,
            "name": target.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return files


def resolve_memory_path(raw_path: str) -> Path:
    cleaned = raw_path.strip()
    if cleaned == "MEMORY.md":
        return ROOT_MEMORY_FILE
    if not cleaned.endswith(".md"):
        raise ValueError("Memory files must end in .md")
    return resolve_workspace_path(cleaned)


def build_skills_payload() -> dict:
    config = build_synthetic_config(read_env(ENV_FILE))
    skills_root = config.get("skills", {})
    entries = skills_root.get("entries", {}) if isinstance(skills_root, dict) else {}
    items = []
    for slug, value in sorted(entries.items()):
        entry = value if isinstance(value, dict) else {}
        items.append({
            "id": slug,
            "slug": slug,
            "name": slug,
            "description": str(entry.get("description") or ""),
            "author": "Hermes Cloud",
            "triggers": [],
            "tags": ["installed"],
            "homepage": None,
            "category": "Productivity",
            "icon": "✨",
            "content": "",
            "fileCount": 0,
            "sourcePath": "",
            "installed": True,
            "enabled": bool(entry.get("enabled", False)),
            "builtin": False,
            "security": {"level": "safe", "flags": [], "score": 0},
        })
    return {"items": items, "skills": items, "categories": ["All", "Productivity"], "total": len(items)}


def build_skill_detail(slug: str) -> dict | None:
    payload = build_skills_payload()
    skill = next((item for item in payload["items"] if item.get("slug") == slug or item.get("id") == slug), None)
    if not skill:
        return None
    return {
        **skill,
        "readme": skill.get("description") or "",
        "status": "installed" if skill.get("installed") else "available",
        "config": {},
    }


def find_session(session_id: str) -> tuple[list[dict], dict | None, int]:
    sessions = load_sessions()
    for index, session in enumerate(sessions):
        if session.get("id") == session_id:
            return sessions, session, index
    return sessions, None, -1


def session_summary(session: dict) -> dict:
    messages = session.get("messages", [])
    tool_call_count = 0
    for message in messages:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
    return {
        "id": session["id"],
        "source": "hermes-cloud",
        "user_id": None,
        "model": session.get("model"),
        "title": session.get("title"),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "end_reason": session.get("end_reason"),
        "message_count": len(messages),
        "tool_call_count": tool_call_count,
        "input_tokens": session.get("input_tokens", 0),
        "output_tokens": session.get("output_tokens", 0),
        "parent_session_id": session.get("parent_session_id"),
        "last_active": session.get("last_active", session.get("started_at")),
    }


def message_record(session_id: str, role: str, content: str, msg_id: int | None = None) -> dict:
    return {
        "id": msg_id if msg_id is not None else now_ts() * 1000,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": now_ts(),
        "tool_calls": [],
        "tool_call_id": None,
        "tool_name": None,
        "token_count": None,
        "finish_reason": None,
    }


async def call_chat_completion(messages: list[dict], stream: bool, session_id: str, model: str | None = None):
    payload = {
        "model": model or "hermes-agent",
        "stream": stream,
        "messages": messages,
    }
    headers = {"Authorization": f"Bearer {(read_env(ENV_FILE).get('API_SERVER_KEY') or os.environ.get('API_SERVER_KEY', ''))}"}
    headers["Content-Type"] = "application/json"
    headers["X-Hermes-Session-Id"] = session_id
    async with httpx.AsyncClient(timeout=None) as client:
        return client.build_request("POST", f"{INTERNAL_API_BASE}/v1/chat/completions", headers=headers, json=payload)


def mask(values: dict[str, str]) -> dict[str, str]:
    return {
        key: (value[:8] + "***" if key in SECRET_KEYS and value else value)
        for key, value in values.items()
    }


def unmask(new_values: dict[str, str], existing_values: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in new_values.items():
        if key in SECRET_KEYS and isinstance(value, str) and value.endswith("***"):
            result[key] = existing_values.get(key, "")
        else:
            result[key] = str(value or "")
    return result


class BasicAuth(AuthenticationBackend):
    async def authenticate(self, conn):
        header = conn.headers.get("Authorization", "")
        if not header:
            return None
        try:
            scheme, credentials = header.split()
            if scheme.lower() != "basic":
                return None
            username, _, password = base64.b64decode(credentials).decode("utf-8").partition(":")
        except Exception as exc:
            raise AuthenticationError("Invalid credentials") from exc
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return AuthCredentials(["authenticated"]), SimpleUser(username)
        raise AuthenticationError("Invalid credentials")


def require_auth(request: Request):
    if request.url.path == "/health":
        return None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        current = read_env(ENV_FILE)
        expected = current.get("API_SERVER_KEY", "") or os.environ.get("API_SERVER_KEY", "")
        if expected and token == expected:
            return None
    if request.user.is_authenticated:
        return None
    return PlainTextResponse("Unauthorized", status_code=401, headers={"WWW-Authenticate": 'Basic realm="hermes"'})


class GatewayManager:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.stdout_handle = None
        self.stderr_handle = None

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        env_values = {**os.environ, **read_env(ENV_FILE)}
        write_config(env_values)
        self.state = "starting"
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        if self.stdout_handle:
            self.stdout_handle.close()
        if self.stderr_handle:
            self.stderr_handle.close()
        self.stdout_handle = open(STATE_ROOT / "gateway.stdout.log", "ab")
        self.stderr_handle = open(STATE_ROOT / "gateway.stderr.log", "ab")
        self.process = await asyncio.create_subprocess_exec(
            "hermes", "gateway",
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            env=env_values,
        )
        self.state = "running"

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            self.state = "stopped"
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.state = "stopped"
        if self.stdout_handle:
            self.stdout_handle.close()
            self.stdout_handle = None
        if self.stderr_handle:
            self.stderr_handle.close()
            self.stderr_handle = None

    async def restart(self):
        await self.stop()
        await self.start()


gateway = GatewayManager()


async def homepage(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = """
    <html><body style="font-family: ui-sans-serif, system-ui; max-width: 760px; margin: 40px auto; line-height: 1.5;">
    <h1>Hermes Runtime</h1>
    <p>This service runs the Hermes gateway and exposes the internal Hermes API server through <code>/v1/*</code>.</p>
    <p>Use <code>GET /api/config</code> and <code>PUT /api/config</code> for configuration, then <code>POST /api/gateway/restart</code>.</p>
    </body></html>
    """
    return HTMLResponse(body)


async def health(_request: Request):
    return JSONResponse({"status": "ok", "gateway": gateway.state})


async def setup_health(_request: Request):
    return JSONResponse({"ok": True, "gateway": gateway.state})


async def setup_status(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    return JSONResponse({"openclawVersion": "hermes-runtime", "configured": ENV_FILE.exists()})


async def setup_run(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    auth_choice = str(body.get("authChoice") or "")
    auth_secret = str(body.get("authSecret") or "")
    existing = read_env(ENV_FILE)
    if auth_choice == "openrouter-api-key" and auth_secret.strip():
        existing["OPENROUTER_API_KEY"] = auth_secret.strip()
    write_env(ENV_FILE, existing)
    write_config(existing)
    asyncio.create_task(gateway.start())
    return JSONResponse({"ok": True, "output": "Configured"})


async def setup_config_raw_get(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    current = read_env(ENV_FILE)
    content = json.dumps(build_synthetic_config(current), indent=2) + "\n"
    return JSONResponse({"path": str(CONFIG_FILE), "exists": True, "content": content})


async def setup_config_raw_post(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    raw_content = str(body.get("content") or "{}")
    config = json.loads(raw_content)
    existing = read_env(ENV_FILE)
    next_env = compile_env_from_config(config, existing)
    write_env(ENV_FILE, next_env)
    write_config(next_env)
    asyncio.create_task(gateway.restart())
    return JSONResponse({"ok": True})


async def api_config_get(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    current = read_env(ENV_FILE)
    auth_header = request.headers.get("Authorization", "")
    vars_payload = {key: current.get(key, "") for key in UI_ENV_KEYS}
    if auth_header.startswith("Bearer "):
        return JSONResponse({"vars": vars_payload})
    return JSONResponse({"vars": mask(vars_payload)})


async def api_config_put(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    posted = body.get("vars", {}) if isinstance(body, dict) else {}
    existing = read_env(ENV_FILE)
    merged = existing | unmask({key: posted.get(key, "") for key in UI_ENV_KEYS}, existing)
    write_env(ENV_FILE, merged)
    write_config(merged)
    if body.get("_restartGateway"):
        asyncio.create_task(gateway.restart())
    return JSONResponse({"ok": True})


async def api_gateway_restart(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    asyncio.create_task(gateway.restart())
    return JSONResponse({"ok": True})


async def api_cloud_file_get(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    raw_path = request.query_params.get("path", "")
    if not raw_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        target = resolve_workspace_path(raw_path)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    if not target.exists():
        return JSONResponse({"content": ""})
    return JSONResponse({"content": target.read_text(encoding="utf-8")})


async def api_cloud_file_put(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    raw_path = str(body.get("path", "")).strip()
    if not raw_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        target = resolve_workspace_path(raw_path)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(body.get("content", "")), encoding="utf-8")
    return JSONResponse({"ok": True})


async def api_runtime_exec(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    command = str(body.get("command") or "").strip()
    if not command:
        return JSONResponse({"error": "command required"}, status_code=400)
    timeout_ms = max(1_000, min(int(body.get("timeoutMs") or 30_000), 600_000))
    cwd_raw = str(body.get("cwd") or WORKSPACE_DIR)
    try:
        cwd = resolve_workspace_path(cwd_raw) if cwd_raw.startswith("/") is False or cwd_raw.startswith("/data/workspace") else Path(cwd_raw)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    cwd.mkdir(parents=True, exist_ok=True)
    env_values = {**os.environ, **read_env(ENV_FILE)}
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        env=env_values,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
        output = stdout.decode("utf-8", errors="replace")
        return JSONResponse({"ok": proc.returncode == 0, "exitCode": proc.returncode, "output": output})
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return JSONResponse({"ok": False, "exitCode": -1, "output": "", "error": "timeout"}, status_code=408)


async def api_sessions_list(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions = [session_summary(session) for session in load_sessions()]
    sessions.sort(key=lambda item: item.get("last_active", 0), reverse=True)
    limit = max(1, min(int(request.query_params.get("limit", "50")), 200))
    offset = max(0, int(request.query_params.get("offset", "0")))
    items = sessions[offset:offset + limit]
    return JSONResponse({"items": items, "total": len(sessions)})


async def api_sessions_create(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    session_id = str(body.get("id") or uuid4())
    title = str(body.get("title") or session_id)
    model = str(body.get("model") or normalize_model_id(read_env(ENV_FILE).get("HERMES_MODEL")))
    sessions = load_sessions()
    if any(session.get("id") == session_id for session in sessions):
        return JSONResponse({"session": session_summary(next(session for session in sessions if session.get("id") == session_id))})
    session = {
        "id": session_id,
        "title": title,
        "model": model,
        "started_at": now_ts(),
        "ended_at": None,
        "end_reason": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "parent_session_id": None,
        "last_active": now_ts(),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
    }
    sessions.append(session)
    save_sessions(sessions)
    return JSONResponse({"session": session_summary(session)})


async def api_session_get(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    _, session, _ = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse({"session": session_summary(session)})


async def api_session_patch(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions, session, index = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    body = await request.json()
    if "title" in body:
        session["title"] = str(body.get("title") or session["id"])
    session["updated_at"] = now_iso()
    session["last_active"] = now_ts()
    sessions[index] = session
    save_sessions(sessions)
    return JSONResponse({"session": session_summary(session)})


async def api_session_delete(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions, _session, index = find_session(request.path_params["session_id"])
    if index < 0:
        return Response(status_code=204)
    sessions.pop(index)
    save_sessions(sessions)
    return Response(status_code=204)


async def api_session_messages(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    _, session, _ = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    messages = session.get("messages", [])
    return JSONResponse({"items": messages, "total": len(messages)})


async def api_sessions_search(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    query = request.query_params.get("q", "").strip().lower()
    limit = max(1, min(int(request.query_params.get("limit", "20")), 100))
    results = []
    for session in load_sessions():
        haystack = "\n".join([
            str(session.get("id", "")),
            str(session.get("title", "")),
            *[str(message.get("content", "")) for message in session.get("messages", [])],
        ]).lower()
        if query and query not in haystack:
            continue
        results.append(session_summary(session))
        if len(results) >= limit:
            break
    return JSONResponse({"query": query, "count": len(results), "results": results})


async def api_session_fork(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions, session, _ = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    fork_id = str(uuid4())
    cloned_messages = json.loads(json.dumps(session.get("messages", [])))
    forked = {
        **session,
        "id": fork_id,
        "title": f"{session.get('title') or session['id']} (fork)",
        "parent_session_id": session["id"],
        "started_at": now_ts(),
        "last_active": now_ts(),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": cloned_messages,
    }
    sessions.append(forked)
    save_sessions(sessions)
    return JSONResponse({"session": session_summary(forked), "forked_from": session["id"]})


async def _run_session_chat(session: dict, prompt: str, model: str | None, stream: bool):
    messages = [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in session.get("messages", [])]
    messages.append({"role": "user", "content": prompt})
    request_obj = await call_chat_completion(messages, stream=stream, session_id=session["id"], model=model)
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.send(request_obj, stream=stream)


def _normalize_tool_call(record: dict, fallback_id: str, fallback_name: str = "tool") -> dict:
    function = record.get("function") if isinstance(record.get("function"), dict) else {}
    return {
        "id": str(record.get("id") or fallback_id),
        "type": str(record.get("type") or "function"),
        "function": {
            "name": str(function.get("name") or record.get("name") or fallback_name),
            "arguments": function.get("arguments") if "arguments" in function else record.get("arguments", ""),
        },
    }


async def api_session_chat(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions, session, index = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    body = await request.json()
    message = str(body.get("message") or "").strip()
    model = str(body.get("model") or session.get("model") or "").strip() or None
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    session_messages = session.setdefault("messages", [])
    user_message = message_record(session["id"], "user", message, len(session_messages) + 1)
    session_messages.append(user_message)
    response = await _run_session_chat(session, message, model, stream=False)
    if not response.is_success:
        text = await response.aread()
        return JSONResponse({"error": text.decode("utf-8", errors="replace")[:500]}, status_code=response.status_code)
    payload = response.json()
    assistant_text = ""
    assistant_tool_calls: list[dict] = []
    choices = payload.get("choices", [])
    if choices and isinstance(choices[0], dict):
        message_payload = (choices[0].get("message") or {}) if isinstance(choices[0].get("message"), dict) else {}
        assistant_text = str(message_payload.get("content") or "")
        raw_tool_calls = message_payload.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            assistant_tool_calls = [
                _normalize_tool_call(tc if isinstance(tc, dict) else {}, f"tool-{session['id']}-{idx}")
                for idx, tc in enumerate(raw_tool_calls, start=1)
            ]
    assistant_message = message_record(session["id"], "assistant", assistant_text, len(session_messages) + 1)
    assistant_message["tool_calls"] = assistant_tool_calls
    session_messages.append(assistant_message)
    session["last_active"] = now_ts()
    session["updated_at"] = now_iso()
    sessions[index] = session
    save_sessions(sessions)
    return JSONResponse({"ok": True, "run_id": str(uuid4()), "message": assistant_message, "session": session_summary(session)})


async def api_session_chat_stream(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    sessions, session, index = find_session(request.path_params["session_id"])
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    body = await request.json()
    message = str(body.get("message") or "").strip()
    model = str(body.get("model") or session.get("model") or "").strip() or None
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    run_id = str(uuid4())
    session_messages = session.setdefault("messages", [])
    user_message = message_record(session["id"], "user", message, len(session_messages) + 1)
    session_messages.append(user_message)
    session["last_active"] = now_ts()
    session["updated_at"] = now_iso()
    sessions[index] = session
    save_sessions(sessions)

    async def event_stream():
        yield f"event: user_message\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'message': user_message})}\n\n"
        yield f"event: message.started\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'message': {'id': f'assistant-{run_id}', 'role': 'assistant'}})}\n\n"
        assistant_chunks: list[str] = []
        tool_call_state: dict[str, dict] = {}
        response = await _run_session_chat(session, message, model, stream=True)
        if not response.is_success:
            body_bytes = await response.aread()
            payload = {"sessionKey": session["id"], "runId": run_id, "state": "error", "errorMessage": body_bytes.decode("utf-8", errors="replace")[:500]}
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
            return
        async for chunk in response.aiter_text():
            for frame in chunk.split("\n\n"):
                frame = frame.strip()
                if not frame.startswith("data:"):
                    continue
                payload_text = frame[5:].strip()
                if not payload_text or payload_text == "[DONE]":
                    continue
                try:
                    data = json.loads(payload_text)
                except Exception:
                    continue
                choices = data.get("choices", [])
                if not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta")
                if isinstance(delta, dict):
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        assistant_chunks.append(text)
                        yield f"event: assistant.delta\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'delta': text})}\n\n"
                        yield f"event: chunk\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'text': text})}\n\n"
                    raw_tool_calls = delta.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        for raw_tool in raw_tool_calls:
                            if not isinstance(raw_tool, dict):
                                continue
                            index = str(raw_tool.get("index", len(tool_call_state)))
                            existing = tool_call_state.get(index, {
                                "id": f"{run_id}:tool:{index}",
                                "function": {"name": "tool", "arguments": ""},
                            })
                            if isinstance(raw_tool.get("id"), str) and raw_tool["id"]:
                                existing["id"] = raw_tool["id"]
                            function = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else {}
                            existing_function = existing.get("function") if isinstance(existing.get("function"), dict) else {"name": "tool", "arguments": ""}
                            if isinstance(function.get("name"), str) and function["name"]:
                                existing_function["name"] = function["name"]
                            if isinstance(function.get("arguments"), str):
                                existing_function["arguments"] = f"{existing_function.get('arguments', '')}{function['arguments']}"
                            existing["function"] = existing_function
                            tool_call_state[index] = existing
                            tool_payload = {
                                "sessionKey": session["id"],
                                "runId": run_id,
                                "tool_call_id": existing["id"],
                                "tool_name": existing_function.get("name") or "tool",
                                "tool_call": existing,
                                "args": existing_function.get("arguments") or "",
                            }
                            yield f"event: tool.pending\ndata: {json.dumps(tool_payload)}\n\n"
                            yield f"event: tool.progress\ndata: {json.dumps({**tool_payload, 'delta': existing_function.get('arguments') or ''})}\n\n"
        assistant_text = "".join(assistant_chunks).strip()
        latest_sessions, latest_session, latest_index = find_session(session["id"])
        if latest_session is not None and latest_index >= 0:
            latest_messages = latest_session.setdefault("messages", [])
            assistant_message = message_record(session["id"], "assistant", assistant_text, len(latest_messages) + 1)
            assistant_tool_calls = [
                _normalize_tool_call(tool_call, tool_call.get("id", f"{run_id}:tool:{idx}"), str((tool_call.get("function") or {}).get("name") or "tool"))
                for idx, tool_call in enumerate(tool_call_state.values(), start=1)
            ]
            assistant_message["tool_calls"] = assistant_tool_calls
            latest_messages.append(assistant_message)
            latest_session["last_active"] = now_ts()
            latest_session["updated_at"] = now_iso()
            latest_sessions[latest_index] = latest_session
            save_sessions(latest_sessions)
            for tool_call in assistant_tool_calls:
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                yield f"event: tool.completed\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'tool_call_id': tool_call.get('id'), 'tool_name': function.get('name') or 'tool', 'tool_call': tool_call, 'args': function.get('arguments') or '', 'message': 'Tool call completed'})}\n\n"
            yield f"event: assistant.completed\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'content': assistant_text})}\n\n"
            yield f"event: run.completed\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'state': 'complete'})}\n\n"
            yield f"event: done\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'state': 'done', 'message': assistant_message})}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps({'sessionKey': session['id'], 'runId': run_id, 'state': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def api_memory(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    return JSONResponse({"files": list_memory_files()})


async def api_memory_list(request: Request):
    return await api_memory(request)


async def api_memory_read(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    raw_path = request.query_params.get("path", "").strip()
    if not raw_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        path = resolve_memory_path(raw_path)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return JSONResponse({"path": raw_path, "content": content})


async def api_memory_write(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    body = await request.json()
    raw_path = str(body.get("path") or "").strip()
    if not raw_path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        path = resolve_memory_path(raw_path)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(body.get("content") or ""), encoding="utf-8")
    return JSONResponse({"success": True, "path": raw_path})


async def api_memory_search(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse({"results": []})
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results = []
    for meta in list_memory_files():
        try:
            path = resolve_memory_path(meta["path"])
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                results.append({"path": meta["path"], "line": line_number, "text": line.strip()})
                if len(results) >= 100:
                    return JSONResponse({"results": results})
    return JSONResponse({"results": results})


async def api_skills(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    return JSONResponse(build_skills_payload())


async def api_skill_detail(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    skill = build_skill_detail(request.path_params["skill_name"])
    if not skill:
        return JSONResponse({"error": "Skill not found"}, status_code=404)
    return JSONResponse({"skill": skill})


async def api_skill_categories(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    return JSONResponse({"categories": ["All", "Productivity"]})


async def api_jobs(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    if request.method == "GET":
        return JSONResponse({"jobs": load_jobs()})
    body = await request.json()
    jobs = load_jobs()
    job = {
        "id": str(uuid4()),
        "name": str(body.get("name") or "Untitled job"),
        "prompt": str(body.get("prompt") or ""),
        "schedule": body.get("schedule") or {"cron": str(body.get("schedule") or "* * * * *")},
        "schedule_display": str(body.get("schedule") or ""),
        "enabled": True,
        "state": "idle",
        "next_run_at": None,
        "last_run_at": None,
        "last_run_success": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "deliver": body.get("deliver") or [],
        "skills": body.get("skills") or [],
        "repeat": {"times": int(body.get("repeat") or 0), "completed": 0},
        "run_count": 0,
        "outputs": [],
    }
    jobs.append(job)
    save_jobs(jobs)
    return JSONResponse({"job": job})


async def api_job_detail(request: Request):
    auth_error = require_auth(request)
    if auth_error:
        return auth_error
    jobs = load_jobs()
    job_id = request.path_params["job_id"]
    job = next((item for item in jobs if item.get("id") == job_id), None)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    action = request.query_params.get("action", "").strip().lower()
    if request.method == "GET":
        if action == "output":
            outputs = job.get("outputs", [])
            limit = max(1, min(int(request.query_params.get("limit", "10")), 100))
            return JSONResponse({"outputs": outputs[:limit]})
        return JSONResponse({"job": job})
    if request.method == "DELETE":
        save_jobs([item for item in jobs if item.get("id") != job_id])
        return Response(status_code=204)
    if request.method == "PATCH":
        body = await request.json()
        for key in ["name", "prompt", "schedule", "deliver", "skills", "enabled"]:
            if key in body:
                job[key] = body[key]
        job["updated_at"] = now_iso()
    elif request.method == "POST":
        if action == "pause":
            job["enabled"] = False
            job["state"] = "paused"
        elif action == "resume":
            job["enabled"] = True
            job["state"] = "idle"
        elif action == "run":
            job["state"] = "completed"
            job["last_run_at"] = now_iso()
            job["last_run_success"] = True
            job["run_count"] = int(job.get("run_count") or 0) + 1
            outputs = job.setdefault("outputs", [])
            outputs.insert(0, {
                "filename": f"{job_id}-{now_ts()}.md",
                "timestamp": now_iso(),
                "content": f"Prompt: {job.get('prompt', '')}",
                "size": len(str(job.get("prompt", ""))),
            })
        else:
            return JSONResponse({"error": "Unsupported action"}, status_code=400)
        job["updated_at"] = now_iso()
    save_jobs(jobs)
    return JSONResponse({"job": job})


async def proxy_v1(request: Request):
    target = f"{INTERNAL_API_BASE}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
    body = await request.body()
    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.send(
            client.build_request(request.method, target, headers=headers, content=body),
            stream=True,
        )
        response_headers = {
            key: value for key, value in upstream.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=None,
        )


@asynccontextmanager
async def lifespan(_app: Starlette):
    if read_env(ENV_FILE):
        try:
            await gateway.start()
        except Exception as error:
            print(f"[hermes-golden] Gateway auto-start failed: {error}", flush=True)
    yield
    await gateway.stop()


routes = [
    Route("/", homepage),
    Route("/health", health),
    Route("/setup/healthz", setup_health),
    Route("/setup/api/status", setup_status, methods=["GET"]),
    Route("/setup/api/run", setup_run, methods=["POST"]),
    Route("/setup/api/config/raw", setup_config_raw_get, methods=["GET"]),
    Route("/setup/api/config/raw", setup_config_raw_post, methods=["POST"]),
    Route("/api/config", api_config_get, methods=["GET"]),
    Route("/api/config", api_config_put, methods=["PUT"]),
    Route("/api/gateway/restart", api_gateway_restart, methods=["POST"]),
    Route("/api/cloud/file", api_cloud_file_get, methods=["GET"]),
    Route("/api/cloud/file", api_cloud_file_put, methods=["PUT"]),
    Route("/api/runtime/exec", api_runtime_exec, methods=["POST"]),
    Route("/api/sessions", api_sessions_list, methods=["GET"]),
    Route("/api/sessions", api_sessions_create, methods=["POST"]),
    Route("/api/sessions/search", api_sessions_search, methods=["GET"]),
    Route("/api/sessions/{session_id:str}", api_session_get, methods=["GET"]),
    Route("/api/sessions/{session_id:str}", api_session_patch, methods=["PATCH"]),
    Route("/api/sessions/{session_id:str}", api_session_delete, methods=["DELETE"]),
    Route("/api/sessions/{session_id:str}/messages", api_session_messages, methods=["GET"]),
    Route("/api/sessions/{session_id:str}/fork", api_session_fork, methods=["POST"]),
    Route("/api/sessions/{session_id:str}/chat", api_session_chat, methods=["POST"]),
    Route("/api/sessions/{session_id:str}/chat/stream", api_session_chat_stream, methods=["POST"]),
    Route("/api/memory", api_memory, methods=["GET"]),
    Route("/api/memory/list", api_memory_list, methods=["GET"]),
    Route("/api/memory/read", api_memory_read, methods=["GET"]),
    Route("/api/memory/write", api_memory_write, methods=["POST"]),
    Route("/api/memory/search", api_memory_search, methods=["GET"]),
    Route("/api/skills", api_skills, methods=["GET"]),
    Route("/api/skills/{skill_name:str}", api_skill_detail, methods=["GET"]),
    Route("/api/skills/categories", api_skill_categories, methods=["GET"]),
    Route("/api/jobs", api_jobs, methods=["GET", "POST"]),
    Route("/api/jobs/{job_id:str}", api_job_detail, methods=["GET", "PATCH", "POST", "DELETE"]),
    Route("/v1/{path:path}", proxy_v1, methods=["GET", "POST", "DELETE", "OPTIONS"]),
]

app = Starlette(
    debug=False,
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(AuthenticationMiddleware, backend=BasicAuth())],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
