
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
    digest = _h.sha256(f"SHAOBKJ_CLOUD_DEVICE:{instance_id}".encode("utf-8", errors="ignore")).hexdigest()[:24].upper()
    return f"SHAOBKJ-CLOUD-{digest}"
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
    if not value or lowered in ('none', 'unknown', 'not specified', 'to be filled by o.e.m.'):
        return ""
    return value
def _read_dmi_anchor():
    dmi_paths = [
        ("dmi_machine_id", _P("/etc/machine-id")),
        ("dmi_product_uuid", _P("/sys/class/dmi/id/product_uuid")),
        ("dmi_product_serial", _P("/sys/class/dmi/id/product_serial")),
        ("dmi_board_serial", _P("/sys/class/dmi/id/board_serial")),
    ]
    values = []
    for source, path in dmi_paths:
        value = _clean_anchor_value(_read_text_file(path))
        if value:
            values.append(f"{source}={value}")
    if values:
        return "local:linux:" + "|".join(values), "linux_machine"
    return "", "none"
def _read_command_value(command):
    try:
        result = _subprocess.run(command, capture_output=True, text=True, timeout=3, shell=False)
        if result.returncode != 0:
            return ""
        lines = [_clean_anchor_value(line) for line in (result.stdout or "").splitlines()]
        lines = [line for line in lines if line and not line.lower().startswith(("uuid", "serialnumber", "identifyingnumber"))]
        return lines[0] if lines else ""
    except Exception:
        return ""
def _read_windows_anchor():
    values = []
    try:
        import winreg as _winreg
        with _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = _winreg.QueryValueEx(key, "MachineGuid")
            value = _clean_anchor_value(str(value))
            if value:
                values.append(f"machine_guid={value}")
    except Exception:
        pass
    commands = [
        ("csproduct_uuid", ["wmic", "csproduct", "get", "UUID"]),
        ("bios_serial", ["wmic", "bios", "get", "SerialNumber"]),
        ("baseboard_serial", ["wmic", "baseboard", "get", "SerialNumber"]),
    ]
    for source, command in commands:
        value = _clean_anchor_value(_read_command_value(command))
        if value:
            values.append(f"{source}={value}")
    computer_name = _clean_anchor_value(_os.getenv("COMPUTERNAME") or _platform.node())
    if computer_name:
        values.append(f"computer_name={computer_name}")
    if values:
        return "local:windows:" + "|".join(values), "windows_machine"
    return "", "none"
def _read_platform_anchor():
    system_name = (_platform.system() or "").lower()
    if system_name == "windows":
        return _read_windows_anchor()
    if system_name == "darwin":
        values = []
        value = _clean_anchor_value(_read_command_value(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]))
        if "IOPlatformUUID" in value:
            value = value.split("IOPlatformUUID", 1)[-1].strip().strip(' ="')
        if value:
            values.append(f"platform_uuid={value}")
        hostname = _clean_anchor_value(_platform.node())
        if hostname:
            values.append(f"hostname={hostname}")
        if values:
            return "local:darwin:" + "|".join(values), "darwin_machine"
    return "", "none"
def _read_container_anchor():
    cgroup = _read_text_file(_P("/proc/self/cgroup"))
    for raw_line in cgroup.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tail = line.rsplit("/", 1)[-1].strip()
        tail = tail.removeprefix("docker-").removesuffix(".scope")
        if len(tail) >= 12 and all(char.isalnum() or char in "_.-" for char in tail):
            return f"local:container_id:{tail}", "container_id"
    hostname = _clean_anchor_value(_os.getenv("HOSTNAME") or _read_text_file(_P("/etc/hostname")))
    if hostname:
        return f"local:hostname:{hostname}", "hostname"
    return "", "none"
def _current_device_anchor():
    instance_id = _configured_instance_id()
    if instance_id:
        return f"cloud:{instance_id}", "cloud_instance_id"
    platform_anchor, platform_source = _read_platform_anchor()
    if platform_anchor:
        return platform_anchor, platform_source
    dmi_anchor, dmi_source = _read_dmi_anchor()
    if dmi_anchor:
        return dmi_anchor, dmi_source
    container_anchor, container_source = _read_container_anchor()
    if container_anchor:
        return container_anchor, container_source
    fallback_values = []
    hostname = _clean_anchor_value(_platform.node() or _os.getenv("HOSTNAME"))
    if hostname:
        fallback_values.append(f"hostname={hostname}")
    user_name = _clean_anchor_value(_os.getenv("USERNAME") or _os.getenv("USER"))
    if user_name:
        fallback_values.append(f"user={user_name}")
    if fallback_values:
        return "local:fallback:" + "|".join(fallback_values), "fallback_machine"
    return "", "none"
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
    if device_id:
        return device_id
    device_id = f"RELEASE-{_SAFE_PLUGIN_NAME}-{_secrets.token_hex(16).upper()}"
    _save_fallback_device_id({"device_id": device_id, "fingerprint": fingerprint})
    return device_id
def _device_response_payload(device_identity):
    device_id = device_identity.get("device_id") or ""
    return {
        "device_id": device_id,
        "device_code": ((device_identity.get("instance_id") or device_id[-12:]) or "").upper(),
        "device_anchor_source": device_identity.get("device_anchor_source") or "none",
        "device_type": device_identity.get("device_type") or "local",
        "instance_id": device_identity.get("instance_id") or "",
        "cloud_runtime": bool(device_identity.get("cloud_runtime")),
    }
def _device_payload():
    current_anchor, anchor_source = _current_device_anchor()
    current_fingerprint = _device_fingerprint(current_anchor) if current_anchor else ""
    instance_id = _configured_instance_id()
    if instance_id:
        payload = {"device_id": _cloud_device_id(instance_id), "device_type": "cloud", "instance_id": instance_id, "device_fingerprint": current_fingerprint, "device_anchor_source": anchor_source, "cloud_runtime": True}
        _write_device_config(payload)
        return _device_response_payload(payload)
    try:
        payload = _json.loads(_DEVICE_FILE.read_text(encoding="utf-8")) if _DEVICE_FILE.is_file() else {}
        device_id = str(payload.get("device_id") or "").strip()
        saved_fingerprint = str(payload.get("device_fingerprint") or "").strip()
        if device_id and saved_fingerprint and saved_fingerprint == current_fingerprint:
            payload.update({"device_type": "local", "instance_id": "", "device_anchor_source": anchor_source, "cloud_runtime": _is_cloud_runtime()})
            return _device_response_payload(payload)
    except Exception:
        pass
    device_id = _local_device_id(current_fingerprint)
    payload = {"device_id": device_id, "device_type": "local", "instance_id": "", "device_fingerprint": current_fingerprint, "device_anchor_source": anchor_source, "cloud_runtime": _is_cloud_runtime()}
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
def _is_authorized():
    try:
        payload = _json.loads(_AUTH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    access_key = str(payload.get("access_key") or "").strip()
    saved_device = payload.get("device") or {}
    current_device = _device_payload()
    validated_at = float(payload.get("validated_at") or 0)
    if payload.get("ok") is not True or payload.get("auth_scope") != _AUTH_SCOPE or saved_device != current_device:
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
    payload.pop("device_fingerprint", None)
    payload.pop("device_anchor_source", None)
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
