import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PREVIEW_EVENT = "shaobkj.free_color.preview";
const PREVIEW_EVENTS = [PREVIEW_EVENT, "free_color_preview"];
const PREVIEW_HEIGHT = 220;
const PREVIEW_PADDING = 10;
const PRESET_WIDGET_NAME = "一键恢复";
const TARGET_COLOR_WIDGET_NAME = "目标颜色";
const PRESET_OPTION_DEFAULT = "默认";
const PRESET_OPTION_CUSTOM = "当前已修改";
const PRESET_OPTION_ADD = "记录并新增";
const PRESET_OPTION_MANAGE_DELETE = "管理删除";
const PRESET_RESERVED = new Set([PRESET_OPTION_DEFAULT, PRESET_OPTION_CUSTOM, PRESET_OPTION_ADD]);
const SLIDER_WIDGETS = new Set([
    "色相",
    "饱和度",
    "亮度",
    "对比度",
    "色温",
    "色调偏移",
    "红通道",
    "绿通道",
    "蓝通道",
    "强度",
    "遮罩羽化",
]);
const PRESET_PARAM_DEFAULTS = {
    目标颜色: "全部",
    色相: 0,
    饱和度: 0,
    亮度: 0,
    对比度: 0,
    色温: 0,
    色调偏移: 0,
    红通道: 0,
    绿通道: 0,
    蓝通道: 0,
    强度: 100,
    遮罩羽化: 0,
};
const PARAM_NAMES = {
    hue: ["色相", "hue"],
    saturation: ["饱和度", "saturation"],
    brightness: ["亮度", "brightness"],
    contrast: ["对比度", "contrast"],
    temperature: ["色温", "temperature"],
    tint: ["色调偏移", "tint"],
    red: ["红通道", "red"],
    green: ["绿通道", "green"],
    blue: ["蓝通道", "blue"],
    strength: ["强度", "strength"],
};

function findNodeById(nodeId) {
    const graphNodes = app?.graph?._nodes;
    if (!graphNodes || !Array.isArray(graphNodes)) {
        return null;
    }
    const target = String(nodeId);
    for (const node of graphNodes) {
        if (String(node.id) === target) {
            return node;
        }
    }
    return null;
}

function findFreeColorNodes() {
    const graphNodes = app?.graph?._nodes;
    if (!graphNodes || !Array.isArray(graphNodes)) {
        return [];
    }
    return graphNodes.filter((node) => node?.type === "Shaobkj_FreeColor");
}

function ensureNodeHeight(node) {
    if (!node) {
        return;
    }
    if (!node.__shaobkjFreeColorBaseHeight) {
        node.__shaobkjFreeColorBaseHeight = node.size ? node.size[1] : 200;
    }
    const minHeight = node.__shaobkjFreeColorBaseHeight + PREVIEW_HEIGHT + PREVIEW_PADDING * 2;
    if (node.size && node.size[1] < minHeight) {
        node.size[1] = minHeight;
    }
}

function clamp01(v) {
    if (v < 0) {
        return 0;
    }
    if (v > 1) {
        return 1;
    }
    return v;
}

function getWidgetNumber(node, aliases, fallback = 0) {
    const widgets = node?.widgets || [];
    for (const alias of aliases) {
        const w = widgets.find((item) => item?.name === alias);
        if (!w) {
            continue;
        }
        const value = Number(w.value);
        if (Number.isFinite(value)) {
            return value;
        }
    }
    return fallback;
}

function getWidgetValue(node, aliases, fallback = null) {
    const widgets = node?.widgets || [];
    for (const alias of aliases) {
        const w = widgets.find((item) => item?.name === alias);
        if (!w) {
            continue;
        }
        if (w.value !== undefined && w.value !== null) {
            return w.value;
        }
    }
    return fallback;
}

function rgbToHsv(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    const s = max <= 0 ? 0 : d / Math.max(max, 1e-8);
    const v = max;
    if (d > 1e-8) {
        const rc = (max - r) / d;
        const gc = (max - g) / d;
        const bc = (max - b) / d;
        if (r === max) {
            h = bc - gc;
        }
        if (g === max) {
            h = 2 + rc - bc;
        }
        if (b === max) {
            h = 4 + gc - rc;
        }
        h /= 6;
        h = h - Math.floor(h);
    }
    return [h, s, v];
}

function hsvToRgb(h, s, v) {
    const h6 = h * 6;
    const i = Math.floor(h6) % 6;
    const f = h6 - Math.floor(h6);
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    if (i === 0) return [v, t, p];
    if (i === 1) return [q, v, p];
    if (i === 2) return [p, v, t];
    if (i === 3) return [p, q, v];
    if (i === 4) return [t, p, v];
    return [v, p, q];
}

function smoothstep(edge0, edge1, value) {
    const size = Math.max(edge1 - edge0, 1e-8);
    const t = clamp01((value - edge0) / size);
    return t * t * (3 - 2 * t);
}

function hueWeight(hue, center, innerWidth, outerWidth) {
    const distance = Math.abs((((hue - center) + 0.5) % 1 + 1) % 1 - 0.5);
    return 1 - smoothstep(innerWidth, outerWidth, distance);
}

function buildTargetMask(targetColor, h, s, v) {
    if (!targetColor || targetColor === "全部") {
        return 1;
    }
    const chromaGate = smoothstep(0.08, 0.2, s) * smoothstep(0.08, 0.18, v);
    const hueCenters = {
        红色: 0,
        黄色: 1 / 6,
        绿色: 2 / 6,
        青色: 3 / 6,
        蓝色: 4 / 6,
        洋红: 5 / 6,
    };
    if (Object.prototype.hasOwnProperty.call(hueCenters, targetColor)) {
        return hueWeight(h, hueCenters[targetColor], 0.055, 0.11) * chromaGate;
    }
    if (targetColor === "白色") {
        return (1 - smoothstep(0.08, 0.22, s)) * smoothstep(0.72, 0.92, v);
    }
    if (targetColor === "中性灰") {
        const lowSat = 1 - smoothstep(0.12, 0.28, s);
        const midVal = smoothstep(0.12, 0.35, v) * (1 - smoothstep(0.68, 0.88, v));
        return lowSat * midVal;
    }
    if (targetColor === "黑色") {
        return (1 - smoothstep(0.08, 0.25, s)) * (1 - smoothstep(0.08, 0.3, v));
    }
    return 1;
}

function renderLocalPreview(node) {
    if (!node || !node.__shaobkjFreeColorBaseImageData || !node.__shaobkjFreeColorPreviewCanvas) {
        return;
    }
    const base = node.__shaobkjFreeColorBaseImageData;
    const data = new Uint8ClampedArray(base.data);
    const targetColor = String(getWidgetValue(node, [TARGET_COLOR_WIDGET_NAME, "target_color"], "全部") || "全部");
    const hue = getWidgetNumber(node, PARAM_NAMES.hue, 0);
    const saturation = getWidgetNumber(node, PARAM_NAMES.saturation, 0);
    const brightness = getWidgetNumber(node, PARAM_NAMES.brightness, 0);
    const contrast = getWidgetNumber(node, PARAM_NAMES.contrast, 0);
    const temperature = getWidgetNumber(node, PARAM_NAMES.temperature, 0);
    const tint = getWidgetNumber(node, PARAM_NAMES.tint, 0);
    const red = getWidgetNumber(node, PARAM_NAMES.red, 0);
    const green = getWidgetNumber(node, PARAM_NAMES.green, 0);
    const blue = getWidgetNumber(node, PARAM_NAMES.blue, 0);
    const strength = getWidgetNumber(node, PARAM_NAMES.strength, 100);
    const invertMask = Boolean(getWidgetValue(node, ["反转遮罩", "invert_mask"], false));
    const hueShift = hue / 360;
    const satScale = 1 + saturation / 100;
    const brightnessShift = brightness / 100;
    const contrastScale = 1 + contrast / 100;
    const tempShift = (temperature / 100) * 0.25;
    const tintShift = (tint / 100) * 0.2;
    const redShift = red / 100;
    const greenShift = green / 100;
    const blueShift = blue / 100;
    const amount = strength / 100;
    const mask = node.__shaobkjFreeColorMaskImageData;
    for (let i = 0; i < data.length; i += 4) {
        const sr = data[i] / 255;
        const sg = data[i + 1] / 255;
        const sb = data[i + 2] / 255;
        const hsv = rgbToHsv(sr, sg, sb);
        const targetMask = buildTargetMask(targetColor, hsv[0], hsv[1], hsv[2]);
        let h = hsv[0] + hueShift;
        h -= Math.floor(h);
        const s = clamp01(hsv[1] * satScale);
        const v = hsv[2];
        const rgb = hsvToRgb(h, s, v);
        let r = rgb[0] + brightnessShift;
        let g = rgb[1] + brightnessShift;
        let b = rgb[2] + brightnessShift;
        r = (r - 0.5) * contrastScale + 0.5;
        g = (g - 0.5) * contrastScale + 0.5;
        b = (b - 0.5) * contrastScale + 0.5;
        r = r + tempShift + redShift;
        g = g - tintShift + greenShift;
        b = b - tempShift + blueShift;
        r = clamp01(r);
        g = clamp01(g);
        b = clamp01(b);
        r = clamp01(sr + (r - sr) * amount * targetMask);
        g = clamp01(sg + (g - sg) * amount * targetMask);
        b = clamp01(sb + (b - sb) * amount * targetMask);
        const rawMaskAlpha = mask ? (mask.data[i] / 255) : 1;
        const maskAlpha = invertMask ? (1 - rawMaskAlpha) : rawMaskAlpha;
        const fr = clamp01(sr + (r - sr) * maskAlpha);
        const fg = clamp01(sg + (g - sg) * maskAlpha);
        const fb = clamp01(sb + (b - sb) * maskAlpha);
        data[i] = Math.round(fr * 255);
        data[i + 1] = Math.round(fg * 255);
        data[i + 2] = Math.round(fb * 255);
    }
    const out = new ImageData(data, base.width, base.height);
    const ctx = node.__shaobkjFreeColorPreviewCanvas.getContext("2d");
    ctx.putImageData(out, 0, 0);
    node.setDirtyCanvas(true, true);
}

function scheduleLocalPreview(node) {
    if (!node) {
        return;
    }
    if (!node.__shaobkjHasRealtimeBase) {
        return;
    }
    if (node.__shaobkjFreeColorPreviewTimer) {
        clearTimeout(node.__shaobkjFreeColorPreviewTimer);
    }
    node.__shaobkjFreeColorPreviewTimer = setTimeout(() => {
        node.__shaobkjFreeColorPreviewTimer = null;
        renderLocalPreview(node);
    }, 16);
}

function installRealtimeHooks(node) {
    if (!node || node.__shaobkjRealtimeHooksInstalled) {
        return;
    }
    node.__shaobkjRealtimeHooksInstalled = true;
    const widgets = node.widgets || [];
    for (const widget of widgets) {
        if (!widget || widget.type === "button" || widget.__shaobkjRealtimeWrapped) {
            continue;
        }
        const originalCallback = widget.callback;
        widget.callback = function () {
            const ret = originalCallback ? originalCallback.apply(this, arguments) : undefined;
            syncPresetState(node);
            scheduleLocalPreview(node);
            return ret;
        };
        widget.__shaobkjRealtimeWrapped = true;
    }
}

function applyPreviewToNode(node, detail) {
    const outputSrc = detail?.image;
    const baseSrc = detail?.base_image || outputSrc;
    const maskSrc = detail?.mask_image || null;
    if (!node || !outputSrc) {
        return;
    }
    if (!node.__shaobkjFreeColorPreviewCanvas) {
        node.__shaobkjFreeColorPreviewCanvas = document.createElement("canvas");
    }
    const outputImg = new Image();
    outputImg.onload = () => {
        node.__shaobkjFreeColorPreviewCanvas.width = outputImg.width;
        node.__shaobkjFreeColorPreviewCanvas.height = outputImg.height;
        const previewCtx = node.__shaobkjFreeColorPreviewCanvas.getContext("2d");
        previewCtx.drawImage(outputImg, 0, 0);
        ensureNodeHeight(node);
        node.setDirtyCanvas(true, true);
    };
    outputImg.src = outputSrc;

    const baseImg = new Image();
    baseImg.onload = () => {
        const baseCanvas = document.createElement("canvas");
        baseCanvas.width = baseImg.width;
        baseCanvas.height = baseImg.height;
        const baseCtx = baseCanvas.getContext("2d");
        baseCtx.drawImage(baseImg, 0, 0);
        node.__shaobkjFreeColorBaseImageData = baseCtx.getImageData(0, 0, baseImg.width, baseImg.height);
        node.__shaobkjHasRealtimeBase = true;
        if (maskSrc) {
            const maskImg = new Image();
            maskImg.onload = () => {
                const maskCanvas = document.createElement("canvas");
                maskCanvas.width = baseImg.width;
                maskCanvas.height = baseImg.height;
                const maskCtx = maskCanvas.getContext("2d");
                maskCtx.drawImage(maskImg, 0, 0, baseImg.width, baseImg.height);
                node.__shaobkjFreeColorMaskImageData = maskCtx.getImageData(0, 0, baseImg.width, baseImg.height);
            };
            maskImg.src = maskSrc;
        } else {
            node.__shaobkjFreeColorMaskImageData = null;
        }
        installRealtimeHooks(node);
        ensureNodeHeight(node);
    };
    baseImg.src = baseSrc;
}

function getWidgetByName(node, name) {
    const widgets = node?.widgets || [];
    return widgets.find((w) => w?.name === name) || null;
}

function getPresetStore(node) {
    if (!node.properties) {
        node.properties = {};
    }
    if (!node.properties.__shaobkjFreeColorPresets || typeof node.properties.__shaobkjFreeColorPresets !== "object") {
        node.properties.__shaobkjFreeColorPresets = { order: [], presets: {} };
    }
    const store = node.properties.__shaobkjFreeColorPresets;
    if (!Array.isArray(store.order)) {
        store.order = [];
    }
    if (!store.presets || typeof store.presets !== "object") {
        store.presets = {};
    }
    if (typeof store.current !== "string" || !store.current) {
        store.current = PRESET_OPTION_DEFAULT;
    }
    store.order = store.order.filter((name) => typeof name === "string" && name.trim() && !PRESET_RESERVED.has(name));
    if (store.current !== PRESET_OPTION_DEFAULT && store.current !== PRESET_OPTION_CUSTOM && !store.order.includes(store.current)) {
        store.current = PRESET_OPTION_DEFAULT;
    }
    return store;
}

let activePresetPanel = null;
let lastPointerAnchor = { x: 120, y: 120 };

function closePresetPanel() {
    if (!activePresetPanel) {
        return;
    }
    if (activePresetPanel.cleanup) {
        activePresetPanel.cleanup();
    }
    if (activePresetPanel.el && activePresetPanel.el.parentNode) {
        activePresetPanel.el.parentNode.removeChild(activePresetPanel.el);
    }
    activePresetPanel = null;
}

function getPointerAnchor() {
    const x = Number(lastPointerAnchor?.x);
    const y = Number(lastPointerAnchor?.y);
    if (Number.isFinite(x) && Number.isFinite(y)) {
        return { x, y };
    }
    return { x: 120, y: 120 };
}

function getCurrentPresetName(node) {
    const store = getPresetStore(node);
    return store.current || PRESET_OPTION_DEFAULT;
}

function getPresetOptions(node) {
    const store = getPresetStore(node);
    const options = [PRESET_OPTION_DEFAULT];
    if (store.current === PRESET_OPTION_CUSTOM) {
        options.push(PRESET_OPTION_CUSTOM);
    }
    options.push(PRESET_OPTION_ADD, PRESET_OPTION_MANAGE_DELETE, ...store.order);
    return options;
}

function updatePresetWidgetDisplay(node) {
    const widget = getWidgetByName(node, PRESET_WIDGET_NAME);
    if (!widget) {
        return;
    }
    const current = getCurrentPresetName(node);
    widget.value = current;
    if (!widget.options) {
        widget.options = {};
    }
    widget.options.values = getPresetOptions(node);
    node.setDirtyCanvas(true, true);
}

function setCurrentPresetName(node, name) {
    const store = getPresetStore(node);
    store.current = name || PRESET_OPTION_DEFAULT;
    updatePresetWidgetDisplay(node);
}

function openPresetPanel(node, anchor = null) {
    closePresetPanel();
    const store = getPresetStore(node);
    const panel = document.createElement("div");
    panel.style.position = "fixed";
    panel.style.zIndex = "100000";
    panel.style.minWidth = "230px";
    panel.style.maxWidth = "300px";
    panel.style.background = "#1f2329";
    panel.style.border = "1px solid #4a4f59";
    panel.style.borderRadius = "8px";
    panel.style.padding = "8px";
    panel.style.boxShadow = "0 12px 30px rgba(0,0,0,0.4)";
    panel.style.color = "#e8ecf1";
    panel.style.fontSize = "13px";

    const title = document.createElement("div");
    title.textContent = "一键恢复";
    title.style.fontWeight = "600";
    title.style.marginBottom = "8px";
    panel.appendChild(title);

    const addRow = (name, onSelect, allowDelete = false) => {
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "8px";
        row.style.marginBottom = "6px";
        const itemBtn = document.createElement("button");
        itemBtn.type = "button";
        itemBtn.textContent = name;
        itemBtn.style.flex = "1";
        itemBtn.style.background = "transparent";
        itemBtn.style.color = name === getCurrentPresetName(node) ? "#7fd3ff" : "#e8ecf1";
        itemBtn.style.border = "1px solid #4a4f59";
        itemBtn.style.borderRadius = "6px";
        itemBtn.style.padding = "6px 8px";
        itemBtn.style.textAlign = "left";
        itemBtn.style.cursor = "pointer";
        itemBtn.onclick = () => {
            onSelect();
            closePresetPanel();
        };
        row.appendChild(itemBtn);
        if (allowDelete) {
            const delBtn = document.createElement("button");
            delBtn.type = "button";
            delBtn.textContent = "删除";
            delBtn.style.background = "#c62828";
            delBtn.style.color = "#fff";
            delBtn.style.border = "none";
            delBtn.style.borderRadius = "6px";
            delBtn.style.padding = "6px 10px";
            delBtn.style.cursor = "pointer";
            delBtn.onclick = (ev) => {
                ev.stopPropagation();
                deletePreset(node, name);
                if (getCurrentPresetName(node) === name) {
                    setCurrentPresetName(node, PRESET_OPTION_DEFAULT);
                }
                closePresetPanel();
            };
            row.appendChild(delBtn);
        }
        panel.appendChild(row);
    };

    addRow(PRESET_OPTION_DEFAULT, () => {
        applyParams(node, PRESET_PARAM_DEFAULTS);
        setCurrentPresetName(node, PRESET_OPTION_DEFAULT);
    });
    addRow(PRESET_OPTION_ADD, () => {
        const inputName = window.prompt("请输入名称");
        if (!inputName || !String(inputName).trim()) {
            return;
        }
        const savedName = savePreset(node, inputName);
        if (!savedName) {
            return;
        }
        setCurrentPresetName(node, savedName);
    });
    for (const name of store.order) {
        addRow(name, () => {
            const latestStore = getPresetStore(node);
            const params = latestStore.presets[name];
            if (params && typeof params === "object") {
                applyParams(node, params);
                setCurrentPresetName(node, name);
            }
        }, true);
    }

    if (anchor && Number.isFinite(anchor.x) && Number.isFinite(anchor.y)) {
        panel.style.left = `${Math.max(8, Math.round(anchor.x))}px`;
        panel.style.top = `${Math.max(8, Math.round(anchor.y))}px`;
    } else {
        const rect = app?.canvas?.canvas?.getBoundingClientRect?.();
        const ds = app?.canvas?.ds;
        if (rect && ds && node?.pos && node?.size) {
            const nodeScreenX = rect.left + node.pos[0] * ds.scale + ds.offset[0];
            const nodeScreenY = rect.top + node.pos[1] * ds.scale + ds.offset[1];
            const nodeScreenW = node.size[0] * ds.scale;
            panel.style.left = `${Math.round(nodeScreenX + nodeScreenW + 12)}px`;
            panel.style.top = `${Math.round(nodeScreenY + 42)}px`;
        } else if (rect) {
            panel.style.left = `${Math.round(rect.left + 60)}px`;
            panel.style.top = `${Math.round(rect.top + 80)}px`;
        } else {
            panel.style.left = "40px";
            panel.style.top = "40px";
        }
    }

    document.body.appendChild(panel);
    const panelRect = panel.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - panelRect.width - 8);
    const maxTop = Math.max(8, window.innerHeight - panelRect.height - 8);
    const currentLeft = Number.parseFloat(panel.style.left || "0");
    const currentTop = Number.parseFloat(panel.style.top || "0");
    panel.style.left = `${Math.min(maxLeft, Math.max(8, Number.isFinite(currentLeft) ? currentLeft : 8))}px`;
    panel.style.top = `${Math.min(maxTop, Math.max(8, Number.isFinite(currentTop) ? currentTop : 8))}px`;
    const onDocPointerDown = (ev) => {
        if (!panel.contains(ev.target)) {
            closePresetPanel();
        }
    };
    setTimeout(() => {
        document.addEventListener("pointerdown", onDocPointerDown, true);
    }, 0);

    activePresetPanel = {
        el: panel,
        cleanup: () => {
            document.removeEventListener("pointerdown", onDocPointerDown, true);
        },
    };
}

function collectCurrentParams(node) {
    const params = {};
    for (const key of Object.keys(PRESET_PARAM_DEFAULTS)) {
        const widget = getWidgetByName(node, key);
        if (!widget) {
            continue;
        }
        params[key] = widget.value;
    }
    return params;
}

function paramsEqual(left, right) {
    for (const [key, defaultValue] of Object.entries(PRESET_PARAM_DEFAULTS)) {
        const leftValue = Object.prototype.hasOwnProperty.call(left || {}, key) ? left[key] : defaultValue;
        const rightValue = Object.prototype.hasOwnProperty.call(right || {}, key) ? right[key] : defaultValue;
        if (leftValue !== rightValue) {
            return false;
        }
    }
    return true;
}

function resolvePresetNameFromParams(node) {
    const currentParams = collectCurrentParams(node);
    if (paramsEqual(currentParams, PRESET_PARAM_DEFAULTS)) {
        return PRESET_OPTION_DEFAULT;
    }
    const store = getPresetStore(node);
    for (const name of store.order) {
        const params = store.presets[name];
        if (params && typeof params === "object" && paramsEqual(currentParams, params)) {
            return name;
        }
    }
    return PRESET_OPTION_CUSTOM;
}

function syncPresetState(node) {
    if (!node) {
        return;
    }
    const store = getPresetStore(node);
    store.current = resolvePresetNameFromParams(node);
    updatePresetWidgetDisplay(node);
}

function applyParams(node, params) {
    let changed = false;
    for (const [key, defaultValue] of Object.entries(PRESET_PARAM_DEFAULTS)) {
        const widget = getWidgetByName(node, key);
        if (!widget) {
            continue;
        }
        const nextValue = Object.prototype.hasOwnProperty.call(params, key) ? params[key] : defaultValue;
        if (widget.value !== nextValue) {
            widget.value = nextValue;
            changed = true;
        }
    }
    if (changed) {
        scheduleLocalPreview(node);
        node.setDirtyCanvas(true, true);
    }
}

function savePreset(node, name) {
    const trimmed = String(name || "").trim();
    if (!trimmed || PRESET_RESERVED.has(trimmed)) {
        return null;
    }
    const store = getPresetStore(node);
    store.presets[trimmed] = collectCurrentParams(node);
    if (!store.order.includes(trimmed)) {
        store.order.push(trimmed);
    }
    updatePresetWidgetDisplay(node);
    return trimmed;
}

function deletePreset(node, name) {
    const trimmed = String(name || "").trim();
    if (!trimmed) {
        return false;
    }
    const store = getPresetStore(node);
    if (!store.presets[trimmed]) {
        return false;
    }
    delete store.presets[trimmed];
    store.order = store.order.filter((item) => item !== trimmed);
    updatePresetWidgetDisplay(node);
    return true;
}

function handlePresetSelection(node, widget, value) {
    const selected = String(value || "");
    if (selected === PRESET_OPTION_DEFAULT) {
        applyParams(node, PRESET_PARAM_DEFAULTS);
        setCurrentPresetName(node, PRESET_OPTION_DEFAULT);
        return;
    }
    if (selected === PRESET_OPTION_ADD) {
        const inputName = window.prompt("请输入名称");
        if (!inputName || !String(inputName).trim()) {
            updatePresetWidgetDisplay(node);
            return;
        }
        const savedName = savePreset(node, inputName);
        if (!savedName) {
            updatePresetWidgetDisplay(node);
            return;
        }
        setCurrentPresetName(node, savedName);
        return;
    }
    if (selected === PRESET_OPTION_MANAGE_DELETE) {
        openPresetPanel(node, getPointerAnchor());
        updatePresetWidgetDisplay(node);
        return;
    }
    const store = getPresetStore(node);
    const params = store.presets[selected];
    if (params && typeof params === "object") {
        applyParams(node, params);
        setCurrentPresetName(node, selected);
        return;
    }
    updatePresetWidgetDisplay(node);
}

function ensurePresetWidget(node) {
    if (!node) {
        return;
    }
    const existing = getWidgetByName(node, PRESET_WIDGET_NAME);
    const onSelect = function (value) {
        handlePresetSelection(node, existing || presetWidgetRef.current, value);
    };
    const presetWidgetRef = { current: null };
    if (!existing) {
        const presetWidget = node.addWidget(
            "combo",
            PRESET_WIDGET_NAME,
            getCurrentPresetName(node),
            (value) => handlePresetSelection(node, presetWidget, value),
            { values: getPresetOptions(node) },
        );
        presetWidgetRef.current = presetWidget;
        presetWidget.name = PRESET_WIDGET_NAME;
        presetWidget.label = PRESET_WIDGET_NAME;
        presetWidget.value = getCurrentPresetName(node);
        if (!presetWidget.options) {
            presetWidget.options = {};
        }
        presetWidget.options.values = getPresetOptions(node);
        node.setDirtyCanvas(true, true);
        return;
    }
    if (existing.type !== "combo") {
        existing.type = "combo";
    }
    existing.name = PRESET_WIDGET_NAME;
    existing.label = PRESET_WIDGET_NAME;
    existing.callback = onSelect;
    if (!existing.options) {
        existing.options = {};
    }
    existing.options.values = getPresetOptions(node);
    updatePresetWidgetDisplay(node);
}

function enforceSliderWidgets(node) {
    if (!node || !node.widgets) {
        return;
    }
    const numberSuffixPattern = /^(.+?)\s*:\s*-?\d+(?:\.\d+)?$/;
    const resolveBaseName = (widget) => {
        const candidates = [widget?.name, widget?.label];
        for (const raw of candidates) {
            if (typeof raw !== "string" || !raw.trim()) {
                continue;
            }
            const text = raw.trim();
            if (SLIDER_WIDGETS.has(text)) {
                return text;
            }
            const m = text.match(numberSuffixPattern);
            if (m && m[1] && SLIDER_WIDGETS.has(m[1].trim())) {
                return m[1].trim();
            }
        }
        return null;
    };
    let changed = false;
    for (const widget of node.widgets) {
        if (!widget) {
            continue;
        }
        const baseName = resolveBaseName(widget);
        if (!baseName) {
            continue;
        }
        if (widget.name !== baseName) {
            widget.name = baseName;
            changed = true;
        }
        if (widget.label !== baseName) {
            widget.label = baseName;
            changed = true;
        }
        if (widget.type !== "slider") {
            widget.type = "slider";
            changed = true;
        }
        if (!widget.options) {
            widget.options = {};
            changed = true;
        }
        if (widget.options.display !== "slider") {
            widget.options.display = "slider";
            changed = true;
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Shaobkj.FreeColorPreview",
    async setup() {
        if (!window.__shaobkjFreeColorPointerTrackInstalled) {
            window.__shaobkjFreeColorPointerTrackInstalled = true;
            window.addEventListener("pointermove", (event) => {
                if (!event) {
                    return;
                }
                const x = Number(event.clientX);
                const y = Number(event.clientY);
                if (!Number.isFinite(x) || !Number.isFinite(y)) {
                    return;
                }
                lastPointerAnchor = { x, y };
            }, true);
        }
        const handlePreviewEvent = (event) => {
            const detail = event?.detail || {};
            const src = detail.image;
            if (!src) {
                return;
            }
            const node = findNodeById(detail.node_id);
            if (node) {
                applyPreviewToNode(node, detail);
                return;
            }
            const candidates = findFreeColorNodes();
            for (const freeColorNode of candidates) {
                applyPreviewToNode(freeColorNode, detail);
            }
        };
        for (const eventName of PREVIEW_EVENTS) {
            api.addEventListener(eventName, handlePreviewEvent);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Shaobkj_FreeColor") {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.__shaobkjFreeColorBaseHeight = this.size ? this.size[1] : 200;
            ensureNodeHeight(this);
            ensurePresetWidget(this);
            enforceSliderWidgets(this);
            syncPresetState(this);
            installRealtimeHooks(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function () {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            ensureNodeHeight(this);
            ensurePresetWidget(this);
            enforceSliderWidgets(this);
            syncPresetState(this);
            installRealtimeHooks(this);
            this.setDirtyCanvas(true, true);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            ensurePresetWidget(this);
            enforceSliderWidgets(this);
            syncPresetState(this);
            return r;
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function () {
            const r = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
            syncPresetState(this);
            scheduleLocalPreview(this);
            return r;
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            const r = onDrawBackground ? onDrawBackground.apply(this, arguments) : undefined;
            if (this.flags?.collapsed) {
                return r;
            }
            const canvas = this.__shaobkjFreeColorPreviewCanvas;
            if (!canvas || !canvas.width || !canvas.height) {
                return r;
            }
            ensureNodeHeight(this);
            const contentWidth = this.size[0] - PREVIEW_PADDING * 2;
            const maxHeight = PREVIEW_HEIGHT;
            const ratio = canvas.width / canvas.height;
            let drawWidth = contentWidth;
            let drawHeight = drawWidth / ratio;
            if (drawHeight > maxHeight) {
                drawHeight = maxHeight;
                drawWidth = drawHeight * ratio;
            }
            const x = PREVIEW_PADDING + (contentWidth - drawWidth) * 0.5;
            const y = this.size[1] - drawHeight - PREVIEW_PADDING;
            ctx.save();
            ctx.fillStyle = "#0f1117";
            ctx.fillRect(PREVIEW_PADDING, this.size[1] - maxHeight - PREVIEW_PADDING, contentWidth, maxHeight);
            ctx.drawImage(canvas, x, y, drawWidth, drawHeight);
            ctx.restore();
            return r;
        };
    },
});
