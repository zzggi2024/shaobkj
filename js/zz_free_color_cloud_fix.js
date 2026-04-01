import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "Shaobkj.FreeColorCloudFix";
const NODE_TYPE = "Shaobkj_FreeColor";
const PREVIEW_HEIGHT = 220;
const PREVIEW_PADDING = 10;
const DEBUG_LINE_HEIGHT = 14;
const DEBUG_BLOCK_HEIGHT = DEBUG_LINE_HEIGHT * 3 + 8;
const SYNC_POLL_MS = 120;
const DEBUG_VERSION = "zz-fix-2026-04-01-1";
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
const SNAPSHOT_WIDGETS = [
    TARGET_COLOR_WIDGET_NAME,
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
    "反转遮罩",
    PRESET_WIDGET_NAME,
];
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

function findFreeColorNodes() {
    const graphNodes = app?.graph?._nodes;
    if (!Array.isArray(graphNodes)) {
        return [];
    }
    return graphNodes.filter((node) => node?.type === NODE_TYPE);
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

function normalizeWidgetText(raw) {
    if (typeof raw !== "string") {
        return "";
    }
    const text = raw.trim();
    if (!text) {
        return "";
    }
    const knownNames = [...SNAPSHOT_WIDGETS, ...SLIDER_WIDGETS];
    for (const name of knownNames) {
        if (text === name) {
            return name;
        }
        if (text.startsWith(`${name}:`) || text.startsWith(`${name}：`) || text.startsWith(`${name} `)) {
            return name;
        }
    }
    return text;
}

function findWidgetByAliases(node, aliases) {
    const widgets = node?.widgets || [];
    const targets = aliases.map((alias) => String(alias || "").trim()).filter(Boolean);
    for (const widget of widgets) {
        if (!widget) {
            continue;
        }
        const candidates = [widget.name, widget.label].map(normalizeWidgetText).filter(Boolean);
        for (const target of targets) {
            if (candidates.includes(target)) {
                return widget;
            }
        }
    }
    return null;
}

function getWidgetByName(node, name) {
    return findWidgetByAliases(node, [name]);
}

function getWidgetNumber(node, aliases, fallback = 0) {
    const widget = findWidgetByAliases(node, aliases);
    if (!widget) {
        return fallback;
    }
    const value = Number(widget.value);
    if (Number.isFinite(value)) {
        return value;
    }
    return fallback;
}

function getWidgetValue(node, aliases, fallback = null) {
    const widget = findWidgetByAliases(node, aliases);
    if (widget && widget.value !== undefined && widget.value !== null) {
        return widget.value;
    }
    return fallback;
}

function parseWidgetBoolean(value) {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return value !== 0;
    }
    const text = String(value || "").trim().toLowerCase();
    if (!text) {
        return false;
    }
    if (text === "false" || text === "0" || text === "off" || text === "no") {
        return false;
    }
    if (text === "true" || text === "1" || text === "on" || text === "yes") {
        return true;
    }
    return Boolean(value);
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

function ensureNodeHeight(node) {
    if (!node) {
        return;
    }
    if (!node.__shaobkjFreeColorCloudFixBaseHeight) {
        node.__shaobkjFreeColorCloudFixBaseHeight = node.size ? node.size[1] : 200;
    }
    const minHeight = node.__shaobkjFreeColorCloudFixBaseHeight + PREVIEW_HEIGHT + PREVIEW_PADDING * 2 + DEBUG_BLOCK_HEIGHT;
    if (node.size && node.size[1] < minHeight) {
        node.size[1] = minHeight;
    }
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
    widget.value = getCurrentPresetName(node);
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

function collectCurrentParams(node) {
    const params = {};
    for (const key of Object.keys(PRESET_PARAM_DEFAULTS)) {
        const widget = getWidgetByName(node, key);
        if (widget) {
            params[key] = widget.value;
        }
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

function handlePresetSelection(node, value) {
    const selected = String(value || "");
    if (selected === PRESET_OPTION_DEFAULT) {
        applyParams(node, PRESET_PARAM_DEFAULTS);
        setCurrentPresetName(node, PRESET_OPTION_DEFAULT);
        return true;
    }
    if (!selected || PRESET_RESERVED.has(selected)) {
        updatePresetWidgetDisplay(node);
        return true;
    }
    const store = getPresetStore(node);
    const params = store.presets[selected];
    if (params && typeof params === "object") {
        applyParams(node, params);
        setCurrentPresetName(node, selected);
        return true;
    }
    updatePresetWidgetDisplay(node);
    return false;
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
    const invertMask = parseWidgetBoolean(getWidgetValue(node, ["反转遮罩", "invert_mask"], false));
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
    if (!node || !node.__shaobkjHasRealtimeBase) {
        return;
    }
    if (node.__shaobkjFreeColorCloudFixTimer) {
        clearTimeout(node.__shaobkjFreeColorCloudFixTimer);
    }
    node.__shaobkjFreeColorCloudFixTimer = setTimeout(() => {
        node.__shaobkjFreeColorCloudFixTimer = null;
        renderLocalPreview(node);
    }, 16);
}

function buildWidgetSnapshot(node) {
    const snapshot = {};
    for (const name of SNAPSHOT_WIDGETS) {
        const widget = getWidgetByName(node, name);
        snapshot[name] = widget ? widget.value : undefined;
    }
    return snapshot;
}

function syncWidgetSnapshot(node) {
    node.__shaobkjFreeColorCloudFixSnapshot = buildWidgetSnapshot(node);
}

function snapshotEqual(left, right) {
    for (const name of SNAPSHOT_WIDGETS) {
        if ((left ? left[name] : undefined) !== (right ? right[name] : undefined)) {
            return false;
        }
    }
    return true;
}

function handleSnapshotDrivenSync(node) {
    if (!node) {
        return;
    }
    const previous = node.__shaobkjFreeColorCloudFixSnapshot;
    const current = buildWidgetSnapshot(node);
    if (!previous) {
        node.__shaobkjFreeColorCloudFixSnapshot = current;
        return;
    }
    if (previous[PRESET_WIDGET_NAME] !== current[PRESET_WIDGET_NAME]) {
        if (handlePresetSelection(node, current[PRESET_WIDGET_NAME])) {
            syncWidgetSnapshot(node);
            return;
        }
    }
    if (!snapshotEqual(previous, current)) {
        syncPresetState(node);
        scheduleLocalPreview(node);
        syncWidgetSnapshot(node);
    }
}

function getDebugLines(node) {
    const invertMask = parseWidgetBoolean(getWidgetValue(node, ["反转遮罩", "invert_mask"], false));
    return [
        `前端: ${DEBUG_VERSION}`,
        `反转遮罩: ${invertMask ? "true" : "false"}`,
        `预设状态: ${getCurrentPresetName(node) || PRESET_OPTION_DEFAULT}`,
    ];
}

function drawDebugOverlay(node, ctx) {
    if (!ctx || node.flags?.collapsed) {
        return;
    }
    ensureNodeHeight(node);
    const contentWidth = node.size[0] - PREVIEW_PADDING * 2;
    const debugTop = node.size[1] - PREVIEW_PADDING - PREVIEW_HEIGHT - DEBUG_BLOCK_HEIGHT;
    const lines = getDebugLines(node);
    ctx.save();
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(PREVIEW_PADDING, debugTop, contentWidth, DEBUG_BLOCK_HEIGHT);
    ctx.fillStyle = "#9fb3c8";
    ctx.font = "12px sans-serif";
    for (let i = 0; i < lines.length; i += 1) {
        ctx.fillText(lines[i], PREVIEW_PADDING + 6, debugTop + 16 + i * DEBUG_LINE_HEIGHT);
    }
    ctx.restore();
}

function startGlobalSyncLoop() {
    if (window.__shaobkjFreeColorCloudFixLoopStarted) {
        return;
    }
    window.__shaobkjFreeColorCloudFixLoopStarted = true;
    window.setInterval(() => {
        const nodes = findFreeColorNodes();
        for (const node of nodes) {
            handleSnapshotDrivenSync(node);
        }
    }, SYNC_POLL_MS);
}

app.registerExtension({
    name: EXTENSION_NAME,
    async setup() {
        startGlobalSyncLoop();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            ensureNodeHeight(this);
            syncPresetState(this);
            syncWidgetSnapshot(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function () {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            ensureNodeHeight(this);
            syncPresetState(this);
            syncWidgetSnapshot(this);
            this.setDirtyCanvas(true, true);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            ensureNodeHeight(this);
            syncPresetState(this);
            syncWidgetSnapshot(this);
            return r;
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function () {
            const r = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
            handleSnapshotDrivenSync(this);
            return r;
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            const r = onDrawBackground ? onDrawBackground.apply(this, arguments) : undefined;
            handleSnapshotDrivenSync(this);
            drawDebugOverlay(this, ctx);
            return r;
        };
    },
});
