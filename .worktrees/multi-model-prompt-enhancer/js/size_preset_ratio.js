import { app } from "/scripts/app.js";

const SIZE_PRESET_NODE = "Shaobkj_SizePreset";
const REAL_RATIO_WIDGET = "比例";
const DISPLAY_RATIO_WIDGET = "比例";
const DISPLAY_RATIO_INTERNAL_NAME = "shaobkj_ratio_select";
const CUSTOM_RATIO = "自定义";
const DEFAULT_RATIOS = ["9:16", "16:9", CUSTOM_RATIO];

function isValidRatio(value) {
    return /^\s*\d+\s*:\s*\d+\s*$/.test(String(value || ""));
}

function normalizeRatio(value) {
    const match = String(value || "").match(/^\s*(\d+)\s*:\s*(\d+)\s*$/);
    if (!match) return "9:16";
    return `${Math.max(1, Number(match[1]))}:${Math.max(1, Number(match[2]))}`;
}

function findRealRatioWidget(node) {
    if (!Array.isArray(node?.widgets)) return null;
    return node.widgets.find((widget) => widget && widget.name === REAL_RATIO_WIDGET && widget.__shaobkjRatioDisplay !== true);
}

function findDisplayRatioWidget(node) {
    if (!Array.isArray(node?.widgets)) return null;
    return node.widgets.find((widget) => widget && widget.name === DISPLAY_RATIO_INTERNAL_NAME);
}

function getRatioValues(value) {
    const values = DEFAULT_RATIOS.filter((item) => item !== CUSTOM_RATIO);
    if (isValidRatio(value)) {
        const ratio = normalizeRatio(value);
        if (!values.includes(ratio)) values.push(ratio);
    }
    values.push(CUSTOM_RATIO);
    return values;
}

function hideRealWidget(widget) {
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

function syncRatio(node, realWidget, displayWidget, value) {
    const ratio = normalizeRatio(value);
    realWidget.value = ratio;
    displayWidget.value = ratio;
    displayWidget.options.values = getRatioValues(ratio);
    node.setDirtyCanvas?.(true, true);
}

function promptCustomRatio(node, realWidget, displayWidget) {
    const current = isValidRatio(realWidget.value) ? normalizeRatio(realWidget.value) : "3:4";
    const value = window.prompt("请输入自定义比例，例如 3:4", current);
    if (value === null || !isValidRatio(value)) {
        syncRatio(node, realWidget, displayWidget, realWidget.value || "9:16");
        return;
    }
    syncRatio(node, realWidget, displayWidget, value);
}

function setupSizePresetRatio(node) {
    const realWidget = findRealRatioWidget(node);
    if (!realWidget) return;

    hideRealWidget(realWidget);

    let displayWidget = findDisplayRatioWidget(node);
    if (!displayWidget) {
        displayWidget = node.addWidget("combo", DISPLAY_RATIO_WIDGET, normalizeRatio(realWidget.value), (value) => {
            if (value === CUSTOM_RATIO) {
                promptCustomRatio(node, realWidget, displayWidget);
                return;
            }
            syncRatio(node, realWidget, displayWidget, value);
        }, { values: getRatioValues(realWidget.value) });
        displayWidget.name = DISPLAY_RATIO_INTERNAL_NAME;
        displayWidget.label = DISPLAY_RATIO_WIDGET;
        displayWidget.__shaobkjRatioDisplay = true;
    }

    displayWidget.options.values = getRatioValues(realWidget.value);
    displayWidget.value = isValidRatio(realWidget.value) ? normalizeRatio(realWidget.value) : "9:16";
    realWidget.value = displayWidget.value;

    const realIndex = node.widgets.indexOf(realWidget);
    const displayIndex = node.widgets.indexOf(displayWidget);
    if (realIndex >= 0 && displayIndex >= 0 && displayIndex !== realIndex + 1) {
        node.widgets.splice(displayIndex, 1);
        const nextRealIndex = node.widgets.indexOf(realWidget);
        node.widgets.splice(nextRealIndex + 1, 0, displayWidget);
    }

    node.onResize?.(node.size);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "shaobkj.SizePresetRatio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== SIZE_PRESET_NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => setupSizePresetRatio(this), 50);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => setupSizePresetRatio(this), 50);
            return result;
        };
    },
});
