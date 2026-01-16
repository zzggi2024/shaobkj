import os
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f"[ComfyUI-shaobkj] Error loading config.json: {e}")


def get_config_value(key, env_key, default):
    if env_key and os.environ.get(env_key):
        return os.environ.get(env_key)
    if key in CONFIG:
        return CONFIG[key]
    return default

