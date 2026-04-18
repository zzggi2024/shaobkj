import json
from pathlib import Path
import re
from urllib.parse import quote

from aiohttp import web
from server import PromptServer


RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "font_styles"
STYLE_DATA_FILE = RESOURCE_DIR / "styles.json"
STYLE_IMAGE_DIR = RESOURCE_DIR / "images"
STYLE_THUMB_DIR = RESOURCE_DIR / "thumbs"
DEFAULT_TEXT = "妙笔生花"
DEFAULT_TEXT_PATTERN = re.compile(re.escape(DEFAULT_TEXT))


def _read_style_items():
    if not STYLE_DATA_FILE.exists():
        return []

    with STYLE_DATA_FILE.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    items = []
    for raw_item in data:
        name = str(raw_item.get("name") or raw_item.get("name_cn") or "").strip()
        if not name:
            continue

        thumbnail = str(raw_item.get("thumbnail") or f"{name}.png").strip()
        image = str(raw_item.get("image") or f"{name}.png").strip()
        prompt = str(raw_item.get("prompt") or "").strip()
        items.append(
            {
                "name": name,
                "name_cn": str(raw_item.get("name_cn") or name).strip(),
                "prompt": prompt,
                "thumbnail": thumbnail,
                "image": image,
            }
        )
    return items


def _get_style_map():
    return {item["name"]: item for item in _read_style_items()}


@PromptServer.instance.routes.get("/shaobkj/font_styles/list")
async def shaobkj_font_styles_list(_request):
    styles = []
    for item in _read_style_items():
        styles.append(
            {
                **item,
                "image_url": f"/shaobkj/font_styles/image?name={quote(item['name'])}",
            }
        )
    return web.json_response({"styles": styles})


@PromptServer.instance.routes.get("/shaobkj/font_styles/image")
async def shaobkj_font_styles_image(request):
    name = str(request.rel_url.query.get("name", "") or "").strip()
    if not name:
        return web.Response(status=400, text="Missing style name")

    style_item = _get_style_map().get(name)
    if not style_item:
        return web.Response(status=404, text="Style not found")

    image_path = (STYLE_THUMB_DIR / style_item["thumbnail"]).resolve()
    if not image_path.exists():
        return web.Response(status=404, text="Image not found")

    try:
        image_path.relative_to(STYLE_THUMB_DIR.resolve())
    except ValueError:
        return web.Response(status=403, text="Invalid image path")

    return web.FileResponse(image_path)


class Shaobkj_FontStyleSelector:
    CATEGORY = "🤖shaobkj-APIbox/提示词"
    FUNCTION = "build_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "替换文字": ("STRING", {"default": DEFAULT_TEXT, "multiline": False, "tooltip": "会替换风格提示词中的“妙笔生花”"}),
                "选择风格": ("STRING", {"default": "", "multiline": True, "tooltip": "由前端选择器写入，多个风格用英文逗号分隔"}),
            }
        }

    def build_prompt(self, 替换文字, 选择风格):
        style_map = _get_style_map()
        selected_names = []
        selected_prompts = []
        seen = set()
        replace_text = str(替换文字 or "").strip() or DEFAULT_TEXT

        for raw_name in str(选择风格 or "").split(","):
            name = raw_name.strip()
            if not name or name in seen:
                continue

            item = style_map.get(name)
            if not item:
                continue

            seen.add(name)
            if item["prompt"]:
                selected_prompts.append(DEFAULT_TEXT_PATTERN.sub(replace_text, item["prompt"]))

        selected_prompt_text = "\n".join(selected_prompts)
        return (selected_prompt_text,)
