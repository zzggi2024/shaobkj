
import hashlib as _h
import hmac as _hm
import importlib.abc as _ia
import importlib.util as _iu
import json as _json
import os as _os
import struct as _struct
import sys as _s
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
def _write_device_config(payload):
    try:
        _DEVICE_FILE.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
def _device_payload():
    instance_id = _configured_instance_id()
    if instance_id:
        anchor = f"cloud:{instance_id}"
        payload = {"device_id": _cloud_device_id(instance_id), "device_type": "cloud", "instance_id": instance_id, "device_fingerprint": _device_fingerprint(anchor)}
        _write_device_config(payload)
        return {"device_id": payload["device_id"], "device_type": "cloud", "instance_id": instance_id}
    try:
        payload = _json.loads(_DEVICE_FILE.read_text(encoding="utf-8")) if _DEVICE_FILE.is_file() else {}
        device_id = str(payload.get("device_id") or "").strip()
        if device_id and str(payload.get("device_type") or "local") == "local":
            return {"device_id": device_id, "device_type": "local", "instance_id": ""}
    except Exception:
        payload = {}
    try:
        import uuid as _uuid
        device_id = f"RELEASE-{_SAFE_PLUGIN_NAME}-{_uuid.getnode():012x}"
    except Exception:
        device_id = f"RELEASE-{_SAFE_PLUGIN_NAME}-LOCAL"
    _write_device_config({"device_id": device_id, "device_type": "local", "instance_id": ""})
    return {"device_id": device_id, "device_type": "local", "instance_id": ""}
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
    if payload.get("ok") is not True or payload.get("auth_scope") != _AUTH_SCOPE or saved_device != current_device or not _remote_validate(access_key):
        _clear_auth()
        return False
    return True
def _save_auth(access_key):
    if not _remote_validate(access_key):
        _clear_auth()
        return False
    _AUTH_FILE.write_text(_json.dumps({"ok": True, "auth_scope": _AUTH_SCOPE, "access_key": access_key.strip(), "device": _device_payload()}, ensure_ascii=False), encoding="utf-8")
    return True
def _register_auth_routes():
    try:
        from aiohttp import web as _web
        from server import PromptServer as _PromptServer
    except Exception:
        return
    routes = _PromptServer.instance.routes
    @routes.get(_AUTH_ROUTE_PREFIX + "/status")
    async def _shaobkj_release_auth_status(request):
        ok = _is_authorized()
        return _web.json_response({"ok": ok, "message": "" if ok else "授权码停用或失效，请重新输入"})
    @routes.post(_AUTH_ROUTE_PREFIX + "/verify")
    async def _shaobkj_release_auth_verify(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
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
