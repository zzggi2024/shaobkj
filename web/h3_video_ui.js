import { app } from "/scripts/app.js";
import {
    getH3IncompatibleInputs,
    getH3InputPolicy,
    migrateLegacyH3WidgetValues,
} from "./h3_video_policy.mjs";

const NODE_TYPE = "Shaobkj_H3_Video";
const NODE_COLOR = "#0091EA";
const NODE_BACKGROUND = "#001A2E";
const IMAGE_PREFIX = "image";
const AUDIO_INPUT = "audio";
const H3_WIDGET_NAMES = [
    "API密钥",
    "模型",
    "提示词",
    "生成时长",
    "画幅比例",
    "分辨率",
    "长边设置",
    "等待时间",
    "返回末帧",
];
const MODEL_DESCRIPTIONS = {
    "minimax-h3-ow-i2v": "普通图生视频：必须连接 image1 首帧图，提示词可选。",
    "minimax-h3-ow-r2v": "普通参考生视频：必须连接 1 张 image1 参考图，提示词必填。",
    "minimax-h3-ow-r2v-fast": "Fast 多图参考生视频：提示词必填，支持 image1 到 image9 共 1～9 张参考图。",
    "minimax-h3-ow-i2v-fast": "Fast 图生视频：必须且只能连接 image1 首帧图，提示词可选。",
    "minimax-h3-ow-fl2va-audio-drive-fast": "Fast 首帧音频驱动：必须且只能连接 image1 和 1 段 audio，提示词可选。",
    "minimax-h3-ow-ref2va-audio-drive-fast": "Fast 参考图音频驱动：必须且只能连接 image1 和 1 段 audio，提示词可选。",
};

function findWidget(node, name) {
    return node?.widgets?.find((item) => item?.name === name || item?.label === name) || null;
}

function findInput(node, name) {
    return node?.inputs?.find((item) => item?.name === name) || null;
}

function isLinked(input) {
    return input && input.link !== null && input.link !== undefined && input.link !== -1;
}

function selectedModel(node) {
    return String(findWidget(node, "模型")?.value || "minimax-h3-ow-i2v").trim();
}

function ensureDescriptionPanel(node) {
    if (node.__shaobkjH3Description || typeof node.addDOMWidget !== "function") {
        return node.__shaobkjH3Description || null;
    }
    const root = document.createElement("div");
    root.style.boxSizing = "border-box";
    root.style.width = "100%";
    root.style.height = "160px";
    root.style.maxHeight = "160px";
    root.style.overflowY = "auto";
    root.style.overflowX = "hidden";
    root.style.padding = "10px 12px";
    root.style.borderTop = "1px solid rgba(255, 255, 255, 0.12)";
    root.style.color = "#d8e8f5";
    root.style.font = "12px/1.55 sans-serif";
    root.style.whiteSpace = "normal";
    root.style.wordBreak = "break-word";

    const title = document.createElement("div");
    title.textContent = "模型选择说明";
    title.style.color = "#78c7ff";
    title.style.fontWeight = "600";
    title.style.marginBottom = "6px";
    title.style.position = "sticky";
    title.style.top = "0";
    title.style.background = NODE_BACKGROUND;

    const current = document.createElement("div");
    const details = document.createElement("div");
    details.style.marginTop = "8px";
    details.style.color = "#aebdca";
    root.append(title, current, details);

    const domWidget = node.addDOMWidget(
        "模型选择说明",
        "shaobkj_h3_model_description",
        root,
        { serialize: false, hideOnZoom: false },
    );
    domWidget.serialize = false;
    domWidget.computeSize = () => [0, 180];
    node.__shaobkjH3Description = { root, current, details, domWidget };
    return node.__shaobkjH3Description;
}

function updateDescription(node, incompatibleInputs = []) {
    const panel = ensureDescriptionPanel(node);
    if (!panel) return false;
    const model = selectedModel(node);
    const warning = incompatibleInputs.length
        ? `\n\n注意：${incompatibleInputs.join("、")} 已连线，但当前模型不使用。为保护工作流连线未自动删除，请先断开。`
        : "";
    const currentText = `当前：${model}\n${MODEL_DESCRIPTIONS[model] || "请选择 H3 OW 模型。"}${warning}`;
    const detailText = Object.entries(MODEL_DESCRIPTIONS)
        .map(([name, description]) => `${name}\n${description}`)
        .join("\n\n");
    const changed = panel.current.textContent !== currentText || panel.details.textContent !== detailText;
    panel.current.textContent = currentText;
    panel.details.textContent = detailText;
    panel.current.style.whiteSpace = "pre-line";
    panel.current.style.color = incompatibleInputs.length ? "#ff9a8b" : "#d8e8f5";
    panel.details.style.whiteSpace = "pre-line";
    return changed;
}

function syncInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return false;
    const model = selectedModel(node);
    let changed = false;
    const policy = getH3InputPolicy(model);
    const targetCount = policy.imageCount;

    if (!findInput(node, "image1") && typeof node.addInput === "function") {
        node.addInput("image1", "IMAGE");
        changed = true;
    }
    for (let index = 2; index <= targetCount; index += 1) {
        if (!findInput(node, `${IMAGE_PREFIX}${index}`) && typeof node.addInput === "function") {
            node.addInput(`${IMAGE_PREFIX}${index}`, "IMAGE");
            changed = true;
        }
    }
    for (let index = 9; index > targetCount; index -= 1) {
        const input = findInput(node, `${IMAGE_PREFIX}${index}`);
        if (input && !isLinked(input) && typeof node.removeInput === "function") {
            node.removeInput(node.inputs.indexOf(input));
            changed = true;
        }
    }

    const audioInput = findInput(node, AUDIO_INPUT);
    if (policy.usesAudio) {
        if (!audioInput && typeof node.addInput === "function") {
            node.addInput(AUDIO_INPUT, "AUDIO");
            changed = true;
        }
    } else if (audioInput && !isLinked(audioInput) && typeof node.removeInput === "function") {
        node.removeInput(node.inputs.indexOf(audioInput));
        changed = true;
    }

    const incompatibleInputs = getH3IncompatibleInputs(
        model,
        node.inputs.filter(isLinked).map((input) => input.name),
    );
    for (const input of node.inputs) {
        const nextLabel = incompatibleInputs.includes(input.name)
            ? `${input.name}（当前模型不使用，请断开）`
            : null;
        if (input.label !== nextLabel) {
            input.label = nextLabel;
            changed = true;
        }
    }
    if (updateDescription(node, incompatibleInputs)) changed = true;
    if (changed) {
        node.onResize?.(node.size);
        node.setDirtyCanvas?.(true, true);
    }
    return changed;
}

app.registerExtension({
    name: "Shaobkj.H3VideoUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        nodeType.prototype.color = NODE_COLOR;
        nodeType.prototype.bgcolor = NODE_BACKGROUND;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated ? originalCreated.apply(this, arguments) : undefined;
            this.color = "#0091EA";
            this.bgcolor = "#001A2E";
            setTimeout(() => syncInputs(this), 30);
            return result;
        };
        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            const migratedValues = migrateLegacyH3WidgetValues(config?.widgets_values);
            if (migratedValues !== config?.widgets_values) {
                H3_WIDGET_NAMES.forEach((name, index) => {
                    const widget = findWidget(this, name);
                    if (widget && index < migratedValues.length) {
                        widget.value = migratedValues[index];
                    }
                });
                config = { ...config, widgets_values: migratedValues };
            }
            const result = originalConfigured
                ? originalConfigured.call(this, config)
                : undefined;
            this.color = "#0091EA";
            this.bgcolor = "#001A2E";
            setTimeout(() => syncInputs(this), 30);
            return result;
        };
        const originalWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value, oldValue, widget) {
            const result = originalWidgetChanged
                ? originalWidgetChanged.apply(this, arguments)
                : undefined;
            if (name === "模型") setTimeout(() => syncInputs(this), 0);
            return result;
        };
        const originalConnectionsChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalConnectionsChanged
                ? originalConnectionsChanged.apply(this, arguments)
                : undefined;
            setTimeout(() => syncInputs(this), 30);
            return result;
        };
    },
});
