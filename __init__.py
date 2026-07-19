
import hashlib as _h
import hmac as _hm
import importlib.abc as _ia
import importlib.util as _iu
import json as _json
import os as _os
import platform as _platform
import secrets as _secrets
import struct as _struct
import subprocess as _subprocess
import sys as _s
import time as _time
import ctypes as _ctypes
import urllib.request as _ur
import urllib.error as _ue
import uuid as _uuid
import zlib as _z
from pathlib import Path as _P
_MAGIC = b"SBGHPUB2"
_SECRET = _h.sha256(b"SHAOBKJ_GITHUB_PUBLISH_2026_V2").digest()
_PLUGIN_NAME = 'shaobkj'
_SAFE_PLUGIN_NAME = 'shaobkj'
_AUTH_SCOPE = 'ACCOUNT:f5ab687684789fee24b634c1dd2fee7a8bad75516d71477a583dbbb7820e6b35'
_API_BASE = "http://49.235.137.221:8000"
_PLUGIN_DIR = _P(__file__).resolve().parent
_AUTH_FILE = _PLUGIN_DIR / ".auth_ok"
_DEVICE_FILE = _PLUGIN_DIR / "device.json"
_AUTH_CHECK_INTERVAL_SECONDS = 86400
_DEVICE_ANCHOR_DISCOVERY_TIMEOUT_SECONDS = 3.0
_DEVICE_ANCHOR_MAX_ATTEMPTS = 3
_DEVICE_ANCHOR_ATTEMPT_TIMEOUTS = (1.5, 0.75, 0.25)
_DEVICE_ANCHOR_RETRY_DELAYS = (0.2, 0.3)
_AUTH_ROUTE_PREFIX = f"/{_SAFE_PLUGIN_NAME}/auth"
def _rotl32(value, shift):
    return ((value << shift) & 0xFFFFFFFF) | (value >> (32 - shift))
def _quarter_round(state, a, b, c, d):
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = _rotl32(state[d], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = _rotl32(state[b], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = _rotl32(state[d], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = _rotl32(state[b], 7)
def _chacha_block(key, nonce, counter):
    state = list(_struct.unpack("<4I", b"expand 32-byte k") + _struct.unpack("<8I", key) + (counter,) + _struct.unpack("<3I", nonce))
    working = state[:]
    for _ in range(10):
        _quarter_round(working, 0, 4, 8, 12); _quarter_round(working, 1, 5, 9, 13); _quarter_round(working, 2, 6, 10, 14); _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15); _quarter_round(working, 1, 6, 11, 12); _quarter_round(working, 2, 7, 8, 13); _quarter_round(working, 3, 4, 9, 14)
    return _struct.pack("<16I", *[((working[i] + state[i]) & 0xFFFFFFFF) for i in range(16)])
def _crypt(data, nonce):
    out = bytearray(); counter = 1
    for offset in range(0, len(data), 64):
        block = _chacha_block(_SECRET, nonce, counter); counter += 1
        chunk = data[offset:offset + 64]
        out.extend(value ^ block[index] for index, value in enumerate(chunk))
    return bytes(out)
def _decrypt_source(data):
    if not data.startswith(_MAGIC): raise ImportError("Invalid encrypted module.")
    nonce = data[len(_MAGIC):len(_MAGIC) + 12]; tag = data[len(_MAGIC) + 12:len(_MAGIC) + 44]; payload = data[len(_MAGIC) + 44:]
    expected = _hm.new(_SECRET, nonce + payload, _h.sha256).digest()
    if not _hm.compare_digest(tag, expected): raise ImportError("Encrypted module integrity check failed.")
    plain = _crypt(payload, nonce)
    return _z.decompress(plain).decode("utf-8-sig").lstrip("\ufeff")
def _plugin_config_value(name):
    config_path = _PLUGIN_DIR / "service.env"
    try:
        for line in config_path.read_text(encoding="utf-8-sig").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw: continue
            key, value = raw.split("=", 1)
            if key.strip() == name: return value.strip().strip('"').strip("'")
    except Exception:
        pass
    return _os.getenv(name, "").strip()
def _normalize_instance_id(value):
    return "".join(char for char in (value or "").strip().upper() if char.isalnum() or char in "_.-:")[:128]
def _configured_instance_id():
    env_value = _plugin_config_value("SHAOBKJ_INSTANCE_ID")
    if env_value: return _normalize_instance_id(env_value)
    try:
        if _DEVICE_FILE.is_file():
            payload = _json.loads(_DEVICE_FILE.read_text(encoding="utf-8"))
            return _normalize_instance_id(str(payload.get("instance_id") or ""))
    except Exception:
        pass
    return ""
def _device_fingerprint(anchor):
    return _h.sha256(anchor.encode("utf-8", errors="ignore")).hexdigest()[:24]
def _cloud_device_id(instance_id):
    digest = _h.sha256(f"JDKF_CLOUD_DEVICE_V2:{instance_id}".encode("utf-8", errors="ignore")).hexdigest()[:32].upper()
    return f"JDKF-CLOUD-{digest}"
def _sha256_hex(value):
    return _h.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()
def _device_id_from_local_anchor(device_anchor, anchor_source):
    source = str(anchor_source or "")
    if source.startswith("windows_"):
        prefix = "WIN"
    elif source.startswith("macos_") or source.startswith("darwin_"):
        prefix = "MAC"
    elif source.startswith("linux_") or source.startswith("dmi_"):
        prefix = "LIN"
    elif source and source != "none":
        prefix = "LIN"
    else:
        prefix = "LOC"
    digest = _sha256_hex(f"JDKF_DEVICE_V2:{device_anchor}")[:32].upper()
    return f"JDKF-{prefix}-{digest}"
def _device_display_code(device_id):
    normalized = "".join(char for char in str(device_id or "").strip().upper() if char.isalnum() or char == "-")
    if normalized.startswith("JDKF-WIN-"):
        prefix = "WIN"
    elif normalized.startswith("JDKF-MAC-"):
        prefix = "MAC"
    elif normalized.startswith("JDKF-LIN-"):
        prefix = "LIN"
    elif normalized.startswith("JDKF-CLOUD-") or normalized.startswith("SHAOBKJ-CLOUD-"):
        prefix = "CLO"
    else:
        prefix = "LOC"
    compact = "".join(char for char in normalized if char.isalnum())
    suffix = compact[-10:]
    return f"{prefix}-{suffix}" if suffix else ""
def _is_stable_local_anchor_source(anchor_source):
    source = str(anchor_source or "")
    return source.startswith(("windows_", "macos_", "darwin_", "linux_", "dmi_"))
def _read_text_file(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
def _is_cloud_runtime():
    if _configured_instance_id():
        return True
    cloud_env_keys = (
        "KUBERNETES_SERVICE_HOST",
        "CLOUD_RUN_JOB",
        "AWS_EXECUTION_ENV",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "TENCENTCLOUD_RUNENV",
        "XIANGONGYUN_INSTANCE_ID",
    )
    if any(_os.getenv(key) for key in cloud_env_keys):
        return True
    if _P("/.dockerenv").exists() or _P("/run/.containerenv").exists():
        return True
    cgroup = _read_text_file(_P("/proc/self/cgroup")).lower()
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd", "lxc"))
def _clean_anchor_value(value):
    value = (value or "").strip().strip("\x00")
    lowered = value.lower()
    if not value or lowered in {
        "none",
        "unknown",
        "not specified",
        "to be filled by o.e.m.",
        "default string",
        "system serial number",
        "not applicable",
        "n/a",
    }:
        return ""
    if lowered in {"00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff"}:
        return ""
    return value
def _clean_uuid_anchor_value(value):
    value = _clean_anchor_value(value)
    if not value:
        return ""
    try:
        _uuid.UUID(value)
    except Exception:
        return ""
    return value
def _read_command_stdout(command, deadline=None):
    timeout = 3.0
    if deadline is not None:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            return ""
        timeout = min(timeout, max(0.1, remaining))
    try:
        result = _subprocess.run(command, capture_output=True, text=True, timeout=timeout, shell=False)
        if result.returncode != 0:
            return ""
        return result.stdout or ""
    except Exception:
        return ""
def _read_command_uuid(command, deadline=None):
    lines = [_clean_uuid_anchor_value(line) for line in _read_command_stdout(command, deadline).splitlines()]
    lines = [line for line in lines if line]
    return lines[0] if lines else ""
def _read_windows_machine_guid():
    try:
        import winreg as _winreg
        with _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = _winreg.QueryValueEx(key, "MachineGuid")
        return _clean_uuid_anchor_value(str(value))
    except Exception:
        return ""
def _read_windows_smbios_uuid(deadline=None):
    value = _read_command_uuid(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystemProduct).UUID"], deadline)
    if value:
        return value
    return _read_command_uuid(["wmic", "csproduct", "get", "UUID"], deadline)
def _read_windows_anchors(deadline=None):
    machine_guid = _read_windows_machine_guid()
    smbios_uuid = _read_windows_smbios_uuid(deadline)
    anchors = []
    if smbios_uuid:
        anchors.append((f"local:windows_smbios_uuid:{smbios_uuid}", "windows_smbios_uuid"))
    if machine_guid:
        anchors.append((f"local:windows_machine_guid:{machine_guid}", "windows_machine_guid"))
    return anchors
def _read_macos_platform_values(deadline=None):
    output = _read_command_stdout(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], deadline)
    values = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        raw_value = line.split("=", 1)[1].strip().strip('"')
        for name in ("IOPlatformUUID", "IOPlatformSerialNumber"):
            if f'"{name}"' not in line:
                continue
            value = _clean_uuid_anchor_value(raw_value) if name == "IOPlatformUUID" else _clean_anchor_value(raw_value)
            if value:
                values[name] = value
            break
    return values
def _read_macos_anchors(deadline=None):
    values = _read_macos_platform_values(deadline)
    anchors = []
    if values.get("IOPlatformUUID"):
        anchors.append((f"local:macos_ioplatform_uuid:{values['IOPlatformUUID']}", "macos_ioplatform_uuid"))
    if values.get("IOPlatformSerialNumber"):
        anchors.append((f"local:macos_platform_serial:{values['IOPlatformSerialNumber']}", "macos_platform_serial"))
    return anchors
def _read_dmi_anchors():
    dmi_paths = [
        ("dmi_product_uuid", _P("/sys/class/dmi/id/product_uuid")),
        ("dmi_product_serial", _P("/sys/class/dmi/id/product_serial")),
        ("dmi_board_serial", _P("/sys/class/dmi/id/board_serial")),
    ]
    anchors = []
    for source, path in dmi_paths:
        value = _clean_anchor_value(_read_text_file(path))
        if value:
            anchors.append((f"local:{source}:{value}", source))
    return anchors
def _read_device_anchors_once(deadline=None):
    system_name = (_platform.system() or "").lower()
    if system_name == "windows":
        return _read_windows_anchors(deadline)
    if system_name == "darwin":
        return _read_macos_anchors(deadline)
    return _read_dmi_anchors()
def _preferred_anchor_source():
    system_name = (_platform.system() or "").lower()
    if system_name == "windows":
        return "windows_smbios_uuid"
    if system_name == "darwin":
        return "macos_ioplatform_uuid"
    return "dmi_product_uuid"
def _anchor_source_priority(source):
    priorities = {
        "windows_smbios_uuid": 0,
        "windows_machine_guid": 1,
        "macos_ioplatform_uuid": 0,
        "macos_platform_serial": 1,
        "dmi_product_uuid": 0,
        "dmi_product_serial": 1,
        "dmi_board_serial": 2,
    }
    return priorities.get(source, 99)
def _discover_device_anchors():
    deadline = _time.monotonic() + _DEVICE_ANCHOR_DISCOVERY_TIMEOUT_SECONDS
    discovered = {}
    preferred_source = _preferred_anchor_source()
    for attempt in range(_DEVICE_ANCHOR_MAX_ATTEMPTS):
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break
        attempt_timeout = _DEVICE_ANCHOR_ATTEMPT_TIMEOUTS[min(attempt, len(_DEVICE_ANCHOR_ATTEMPT_TIMEOUTS) - 1)]
        attempt_deadline = min(deadline, _time.monotonic() + attempt_timeout)
        for anchor, source in _read_device_anchors_once(attempt_deadline):
            if anchor and source and source != "none":
                discovered[source] = anchor
        if preferred_source in discovered or attempt >= _DEVICE_ANCHOR_MAX_ATTEMPTS - 1:
            break
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break
        delay = _DEVICE_ANCHOR_RETRY_DELAYS[min(attempt, len(_DEVICE_ANCHOR_RETRY_DELAYS) - 1)]
        _time.sleep(min(delay, remaining))
    return sorted(
        ((anchor, source) for source, anchor in discovered.items()),
        key=lambda item: _anchor_source_priority(item[1]),
    )
def _current_device_anchor():
    instance_id = _configured_instance_id()
    if instance_id:
        return f"cloud:{instance_id}", "cloud_instance_id"
    anchors = _discover_device_anchors()
    return anchors[0] if anchors else ("", "none")
def _set_hidden_file(path):
    try:
        if _platform.system().lower() == "windows":
            FILE_ATTRIBUTE_HIDDEN = 0x02
            _ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass

def _write_device_config(payload):
    try:
        _DEVICE_FILE.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _set_hidden_file(_DEVICE_FILE)
    except Exception:
        pass

def _fallback_device_cache_path():
    return _PLUGIN_DIR / ".device_cache"

def _load_fallback_device_id():
    cache_path = _fallback_device_cache_path()
    try:
        if cache_path.is_file():
            payload = _json.loads(cache_path.read_text(encoding="utf-8"))
            device_id = str(payload.get("device_id") or "").strip()
            if device_id:
                return payload
    except Exception:
        pass
    return {}

def _save_fallback_device_id(payload):
    try:
        cache_path = _fallback_device_cache_path()
        cache_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _set_hidden_file(cache_path)
    except Exception:
        pass

def _local_device_id(fingerprint):
    cached = _load_fallback_device_id()
    device_id = str(cached.get("device_id") or "").strip()
    if device_id.startswith("JDKF-LOC-"):
        return device_id
    device_id = f"JDKF-LOC-{_secrets.token_hex(16).upper()}"
    _save_fallback_device_id({"device_id": device_id, "fingerprint": fingerprint})
    return device_id
def _read_device_config():
    try:
        if _DEVICE_FILE.is_file():
            payload = _json.loads(_DEVICE_FILE.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    return _load_fallback_device_id()
def _device_anchor_fingerprints(anchors):
    return {source: _device_fingerprint(anchor) for anchor, source in anchors if anchor and source != "none"}
def _saved_anchor_fingerprints(payload):
    saved = payload.get("device_anchor_fingerprints")
    fingerprints = {
        str(source): str(fingerprint)
        for source, fingerprint in saved.items()
        if source and fingerprint
    } if isinstance(saved, dict) else {}
    source = str(payload.get("device_anchor_source") or "").strip()
    fingerprint = str(payload.get("device_fingerprint") or "").strip()
    if source and fingerprint:
        fingerprints.setdefault(source, fingerprint)
    return fingerprints
def _is_stable_local_device_id(device_id):
    normalized = str(device_id or "").strip().upper()
    return normalized.startswith(("JDKF-WIN-", "JDKF-MAC-", "JDKF-LIN-"))
def _device_response_payload(device_identity):
    device_id = device_identity.get("device_id") or ""
    return {
        "device_id": device_id,
        "device_code": _device_display_code(device_id),
        "device_anchor_source": device_identity.get("device_anchor_source") or "none",
        "device_type": device_identity.get("device_type") or "local",
        "instance_id": device_identity.get("instance_id") or "",
        "cloud_runtime": bool(device_identity.get("cloud_runtime")),
    }
def _device_payload():
    instance_id = _configured_instance_id()
    if instance_id:
        current_anchor = f"cloud:{instance_id}"
        anchor_source = "cloud_instance_id"
        current_fingerprint = _device_fingerprint(current_anchor)
        payload = {"device_id": _cloud_device_id(instance_id), "device_type": "cloud", "instance_id": instance_id, "device_fingerprint": current_fingerprint, "device_anchor_source": anchor_source, "cloud_runtime": True}
        _write_device_config(payload)
        return _device_response_payload(payload)

    saved_payload = _read_device_config()
    stored_device_id = str(saved_payload.get("device_id") or "").strip()
    anchors = _discover_device_anchors()
    current_fingerprints = _device_anchor_fingerprints(anchors)

    if not anchors:
        if stored_device_id:
            payload = {
                **saved_payload,
                "device_id": stored_device_id,
                "device_type": "local",
                "instance_id": "",
                "device_anchor_source": "stored_only",
                "cloud_runtime": _is_cloud_runtime(),
            }
            return _device_response_payload(payload)
        device_id = _local_device_id("")
        payload = {"device_id": device_id, "device_type": "local", "instance_id": "", "device_anchor_source": "none", "cloud_runtime": _is_cloud_runtime()}
        _write_device_config(payload)
        return _device_response_payload(payload)

    current_anchor, anchor_source = anchors[0]
    current_fingerprint = current_fingerprints[anchor_source]

    if stored_device_id and _is_stable_local_device_id(stored_device_id):
        saved_fingerprints = _saved_anchor_fingerprints(saved_payload)
        matching_source = next(
            (source for source, fingerprint in current_fingerprints.items() if saved_fingerprints.get(source) == fingerprint),
            "",
        )
        saved_source = str(saved_payload.get("device_anchor_source") or "").strip()
        generated_ids = {_device_id_from_local_anchor(anchor, source) for anchor, source in anchors}
        if matching_source or stored_device_id in generated_ids or not saved_fingerprints:
            payload = {
                "device_id": stored_device_id,
                "device_id_version": 2,
                "device_type": "local",
                "instance_id": "",
                "device_fingerprint": current_fingerprint,
                "device_anchor_source": anchor_source,
                "device_anchor_fingerprints": {**saved_fingerprints, **current_fingerprints},
                "cloud_runtime": _is_cloud_runtime(),
            }
            _write_device_config(payload)
            return _device_response_payload(payload)

        if saved_source and saved_source not in current_fingerprints:
            payload = {
                **saved_payload,
                "device_id": stored_device_id,
                "device_type": "local",
                "instance_id": "",
                "device_anchor_source": "stored_only",
                "cloud_runtime": _is_cloud_runtime(),
            }
            return _device_response_payload(payload)

    device_id = _device_id_from_local_anchor(current_anchor, anchor_source)
    payload = {
        "device_id": device_id,
        "device_id_version": 2,
        "device_type": "local",
        "instance_id": "",
        "device_fingerprint": current_fingerprint,
        "device_anchor_source": anchor_source,
        "device_anchor_fingerprints": current_fingerprints,
        "cloud_runtime": _is_cloud_runtime(),
    }
    _write_device_config(payload)
    return _device_response_payload(payload)
def _remote_validate(access_key):
    access_key = (access_key or "").strip()
    if not access_key: return False
    body = _json.dumps({"code": _AUTH_SCOPE, "access_key": access_key, "key": access_key, **_device_payload()}, ensure_ascii=False).encode("utf-8")
    request = _ur.Request(_API_BASE + "/Shaobkj/api/access/validate", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _ur.urlopen(request, timeout=15) as response:
            data = _json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return bool(data.get("valid")) and data.get("status") == "active"
def _clear_auth():
    try: _AUTH_FILE.unlink()
    except Exception: pass
def _same_device_identity(saved_device, current_device):
    keys = ("device_id", "device_type", "instance_id")
    return all(str(saved_device.get(key) or "") == str(current_device.get(key) or "") for key in keys)
def _is_authorized():
    try:
        payload = _json.loads(_AUTH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    access_key = str(payload.get("access_key") or "").strip()
    saved_device = payload.get("device") or {}
    current_device = _device_payload()
    validated_at = float(payload.get("validated_at") or 0)
    if payload.get("ok") is not True or payload.get("auth_scope") != _AUTH_SCOPE or not _same_device_identity(saved_device, current_device):
        _clear_auth()
        return False
    if _time.time() - validated_at < _AUTH_CHECK_INTERVAL_SECONDS:
        return True
    if not _remote_validate(access_key):
        _clear_auth()
        return False
    payload["validated_at"] = _time.time()
    try:
        _AUTH_FILE.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True
def _save_auth(access_key):
    if not _remote_validate(access_key):
        _clear_auth()
        return False
    _AUTH_FILE.write_text(_json.dumps({"ok": True, "auth_scope": _AUTH_SCOPE, "access_key": access_key.strip(), "device": _device_payload(), "validated_at": _time.time()}, ensure_ascii=False), encoding="utf-8")
    return True
def _set_configured_instance_id(instance_id):
    normalized = _normalize_instance_id(instance_id)
    if not normalized:
        return _device_payload()
    try:
        payload = _json.loads(_DEVICE_FILE.read_text(encoding="utf-8")) if _DEVICE_FILE.is_file() else {}
    except Exception:
        payload = {}
    payload["instance_id"] = normalized
    payload.pop("device_id", None)
    payload.pop("device_id_version", None)
    payload.pop("device_fingerprint", None)
    payload.pop("device_anchor_source", None)
    payload.pop("device_anchor_fingerprints", None)
    _write_device_config(payload)
    return _device_payload()
def _register_auth_routes():
    try:
        from aiohttp import web as _web
        from server import PromptServer as _PromptServer
    except Exception:
        return
    routes = _PromptServer.instance.routes
    @routes.get(_AUTH_ROUTE_PREFIX + "/device-id")
    async def _shaobkj_release_device_id(request):
        _ = request
        return _web.json_response(_device_payload())
    @routes.post(_AUTH_ROUTE_PREFIX + "/device-instance")
    async def _shaobkj_release_device_instance(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return _web.json_response(_set_configured_instance_id(payload.get("instance_id") or ""))
    @routes.get(_AUTH_ROUTE_PREFIX + "/status")
    async def _shaobkj_release_auth_status(request):
        _ = request
        ok = _is_authorized()
        return _web.json_response({"ok": ok, "message": "" if ok else "授权码停用或失效，请重新输入"})
    @routes.post(_AUTH_ROUTE_PREFIX + "/verify")
    async def _shaobkj_release_auth_verify(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        _set_configured_instance_id(payload.get("instance_id") or "")
        if _save_auth(payload.get("code") or ""):
            return _web.json_response({"ok": True, "message": "授权成功"})
        return _web.json_response({"ok": False, "message": "授权码错误"}, status=400)
def _wrap_node_class(cls):
    function_name = getattr(cls, "FUNCTION", None)
    if not function_name or getattr(cls, "_shaobkj_auth_wrapped", False): return cls
    original = getattr(cls, function_name, None)
    if not callable(original): return cls
    def _wrapped(self, *args, **kwargs):
        if not _is_authorized():
            raise PermissionError("未授权")
        return original(self, *args, **kwargs)
    setattr(cls, function_name, _wrapped)
    setattr(cls, "_shaobkj_auth_wrapped", True)
    return cls
def _wrap_node_mappings(namespace):
    mappings = namespace.get("NODE_CLASS_MAPPINGS")
    if isinstance(mappings, dict):
        for cls in mappings.values():
            if isinstance(cls, type): _wrap_node_class(cls)
class _Loader(_ia.Loader):
    def __init__(self, fullname, path, is_package=False):
        self.fullname = fullname; self.path = path; self._is_package = is_package
    def create_module(self, spec): return None
    def is_package(self, fullname): return self._is_package
    def exec_module(self, module):
        source = _decrypt_source(self.path.read_bytes())
        module.__file__ = str(self.path); module.__loader__ = self; module.__cached__ = None
        module.__package__ = self.fullname if self._is_package else self.fullname.rpartition(".")[0]
        if self._is_package: module.__path__ = [str(self.path.parent)]
        exec(compile(source, str(self.path), "exec"), module.__dict__)
        _wrap_node_mappings(module.__dict__)
class _Finder(_ia.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix): return None
        rel_path = fullname[len(prefix):].replace(".", _os.sep); base = _P(__file__).resolve().parent
        module_file = base / f"{rel_path}.py.sbgc"; package_file = base / rel_path / "__init__.py.sbgc"
        if module_file.is_file(): return _iu.spec_from_loader(fullname, _Loader(fullname, module_file), origin=str(module_file))
        if package_file.is_file():
            loader = _Loader(fullname, package_file, True)
            spec = _iu.spec_from_loader(fullname, loader, origin=str(package_file), is_package=True); spec.submodule_search_locations = [str(package_file.parent)]; return spec
        package_dir = base / rel_path
        if package_dir.is_dir():
            spec = _iu.spec_from_loader(fullname, loader=None, is_package=True); spec.submodule_search_locations = [str(package_dir)]; return spec
        return None
_register_auth_routes()
if not any(isinstance(f, _Finder) for f in _s.meta_path): _s.meta_path.insert(0, _Finder())

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Version
__version__ = "15.0.0"

# Export WEB_DIRECTORY to allow ComfyUI to load the JS extension
WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

WEB_DIRECTORY = "web"
_wrap_node_mappings(globals())
