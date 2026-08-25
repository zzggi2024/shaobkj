import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import {
  createLatestRequestQueue,
  computedNodeHeight,
  mediaSeriesForFeature,
  modelLockFromWorkflow,
  modelSelectionAfterRefresh,
  persistModelLock,
  reconcileFeaturePorts,
  reconcileSeries,
  visibleWidgetNames,
} from "./multi_prompt_enhancer_policy.mjs";

const NODE_CLASS = "Shaobkj_Multi_Prompt_Enhancer";
const DISPLAY_NAME = "✨ 多模型提示词增强器";
const MODEL_ENDPOINT = "/shaobkj/multi_prompt_enhancer/models";
const MODEL_WIDGET = "模型选择";
const FEATURE_WIDGET = "功能类型";
const ADVANCED_WIDGET = "高级设置";
const WARNING_SUFFIX = "（当前不参与）";

function findWidget(node, name) {
  return Array.isArray(node?.widgets)
    ? node.widgets.find((widget) => widget?.name === name)
    : null;
}

function widgetValues(node) {
  const values = {};
  for (const widget of node?.widgets || []) {
    if (widget?.name) values[widget.name] = widget.value;
  }
  values.__advancedExpanded = Boolean(findWidget(node, ADVANCED_WIDGET)?.value);
  return values;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!widget.__shaobkjEnhancerDisplayState) {
    widget.__shaobkjEnhancerDisplayState = {
      type: widget.type,
      computeSize: widget.computeSize,
    };
  }
  const original = widget.__shaobkjEnhancerDisplayState;
  if (visible) {
    widget.type = original.type;
    if (original.computeSize) widget.computeSize = original.computeSize;
    else delete widget.computeSize;
    widget.hidden = false;
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
  widget.hidden = true;
}

function updateCombo(widget, values, fallback) {
  if (!widget) return;
  const options = [...new Set(values)];
  widget.options = widget.options || {};
  widget.options.values = options;
  if (!options.includes(String(widget.value))) widget.value = fallback ?? options[0];
}

function durationOptions(feature) {
  if (feature === "H3") return Array.from({ length: 12 }, (_, index) => `${index + 4}秒`);
  if (feature === "Seedance") return ["自动", ...Array.from({ length: 27 }, (_, index) => `${index + 4}秒`)];
  return ["自动", ...Array.from({ length: 300 }, (_, index) => `${index + 1}秒`)];
}

function desiredFixedPorts(feature, values) {
  if (feature === "Seedance") return ["首帧图", "尾帧图"];
  if (feature !== "H3") return [];
  const mode = String(values["生成模式"] || "自动");
  if (mode === "I2VA") return ["首帧图"];
  if (mode === "FL2VA" || mode === "自动") return ["首帧图", "尾帧图"];
  if (mode === "L2VA") return ["尾帧图"];
  return [];
}

function inputLinked(input) {
  return input?.link !== null && input?.link !== undefined && input?.link !== -1;
}

function findInputIndex(node, name) {
  if (typeof node?.findInputSlot === "function") return node.findInputSlot(name);
  return (node?.inputs || []).findIndex((input) => input?.name === name);
}

function applyPortOperation(node, operation, conservative) {
  const index = findInputIndex(node, operation.name);
  if (operation.action === "add" && index < 0) {
    node.addInput?.(operation.name, operation.type);
    return true;
  }
  if (operation.action === "remove" && !conservative && index >= 0) {
    const input = node.inputs[index];
    if (!inputLinked(input)) {
      node.removeInput?.(index);
      return true;
    }
  }
  if (operation.action === "retain-warning" && index >= 0) {
    const input = node.inputs[index];
    if (input.label !== operation.label) {
      input.label = operation.label;
      return true;
    }
  }
  return false;
}

function reconcilePorts(node, feature, values, conservative = false) {
  if (!Array.isArray(node.inputs)) node.inputs = [];
  let changed = false;
  const dynamicName = /^(参考图片|参考视频|参考音频)\d+$/;
  for (const input of node.inputs) {
    if ((dynamicName.test(String(input?.name || "")) || ["首帧图", "尾帧图"].includes(input?.name)) && input.label !== input.name) {
      input.label = input.name;
      changed = true;
    }
  }

  for (const name of desiredFixedPorts(feature, values)) {
    changed = applyPortOperation(node, { action: "add", name, type: "IMAGE" }, conservative) || changed;
  }
  for (const series of mediaSeriesForFeature(feature)) {
    const snapshot = node.inputs.map((input) => ({ ...input, linked: inputLinked(input) }));
    for (const operation of reconcileSeries(snapshot, series, feature)) {
      changed = applyPortOperation(node, operation, conservative) || changed;
    }
  }
  const snapshot = node.inputs.map((input) => ({ ...input, linked: inputLinked(input) }));
  for (const operation of reconcileFeaturePorts(snapshot, feature, values)) {
    changed = applyPortOperation(node, operation, conservative) || changed;
  }
  return changed;
}

function updateWarningStatus(node) {
  const hasWarning = (node.inputs || []).some((input) => String(input?.label || "").endsWith(WARNING_SUFFIX));
  const baseTitle = node.__shaobkjEnhancerBaseTitle || node.title || DISPLAY_NAME;
  node.__shaobkjEnhancerBaseTitle = baseTitle.replace(/ · 已连接素材不参与$/, "");
  node.title = hasWarning
    ? `${node.__shaobkjEnhancerBaseTitle} · 已连接素材不参与`
    : node.__shaobkjEnhancerBaseTitle;
}

function ensureAdvancedToggle(node) {
  let widget = findWidget(node, ADVANCED_WIDGET);
  if (widget) return widget;
  widget = node.addWidget?.("toggle", ADVANCED_WIDGET, false, () => applyNodePolicy(node), { on: "展开", off: "收起" });
  if (widget) widget.serialize = false;
  return widget;
}

function applyNodePolicy(node, { conservative = false } = {}) {
  if (!node || node.__shaobkjEnhancerApplying) return;
  node.__shaobkjEnhancerApplying = true;
  try {
    ensureAdvancedToggle(node);
    const values = widgetValues(node);
    const feature = String(values[FEATURE_WIDGET] || "H3");
    const visible = visibleWidgetNames(feature, values);
    for (const widget of node.widgets || []) {
      setWidgetVisible(widget, widget.name === ADVANCED_WIDGET || visible.has(widget.name));
    }
    updateCombo(findWidget(node, "目标时长"), durationOptions(feature), feature === "H3" ? "4秒" : "自动");
    const portsChanged = reconcilePorts(node, feature, values, conservative);
    updateWarningStatus(node);

    const visibleCount = (node.widgets || []).filter((widget) => widget?.hidden !== true).length;
    const height = computedNodeHeight(visibleCount, Boolean(values.__advancedExpanded));
    const width = Math.max(420, Number(node.size?.[0]) || 420);
    if (node.size?.[0] !== width || node.size?.[1] !== height) node.setSize?.([width, height]);
    if (portsChanged) node.onResize?.(node.size);
    node.setDirtyCanvas?.(true, true);
  } finally {
    node.__shaobkjEnhancerApplying = false;
  }
}

function updateModelOptions(node, models) {
  const widget = findWidget(node, MODEL_WIDGET);
  if (!widget) return;
  const options = [...new Set(["智能选择", ...(models || []).filter(Boolean)])];
  const current = String(widget.value || "智能选择");
  updateCombo(widget, options, modelSelectionAfterRefresh(current, options));
  if (!options.includes(current)) setModelLocked(node, false);
}

function setModelLocked(node, locked) {
  node.__shaobkjModelLocked = Boolean(locked);
  node.properties = persistModelLock(node.properties, locked);
}

function restoreModelLock(node) {
  const current = String(findWidget(node, MODEL_WIDGET)?.value || "智能选择");
  setModelLocked(node, modelLockFromWorkflow(node.properties, current));
}

function modelRefreshQueue(node) {
  if (node.__shaobkjEnhancerModelQueue) return node.__shaobkjEnhancerModelQueue;
  node.__shaobkjEnhancerModelQueue = createLatestRequestQueue(async (apiKey, isLatest) => {
    try {
      const response = await api.fetchApi(MODEL_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey }),
      });
      const payload = await response.json();
      if (!response.ok || !Array.isArray(payload?.models)) throw new Error("model discovery failed");
      if (!isLatest()) return null;
      updateModelOptions(node, payload.models);
      node.__shaobkjEnhancerModelWarning = String(payload.warning || "");
      return payload.models;
    } catch (_error) {
      if (!isLatest()) return null;
      updateModelOptions(node, ["智能选择"]);
      node.__shaobkjEnhancerModelWarning = "模型列表暂时不可用";
      console.warn("[Shaobkj] 多模型提示词增强器暂时无法刷新模型列表");
      return null;
    } finally {
      if (isLatest()) node.setDirtyCanvas?.(true, true);
    }
  });
  return node.__shaobkjEnhancerModelQueue;
}

async function refreshModels(node) {
  if (!node) return null;
  const apiKey = String(findWidget(node, "API密钥")?.value || "").trim();
  return modelRefreshQueue(node).run(apiKey);
}

function installWidgetCallbacks(node) {
  for (const widget of node.widgets || []) {
    if (!widget || widget.__shaobkjEnhancerCallbackInstalled) continue;
    widget.__shaobkjEnhancerCallbackInstalled = true;
    const original = widget.callback;
    widget.callback = function (value) {
      const result = original ? original.apply(this, arguments) : undefined;
      if (widget.name === MODEL_WIDGET && !node.__shaobkjEnhancerSettingModel) {
        setModelLocked(node, String(value || "") !== "智能选择");
      }
      applyNodePolicy(node);
      if (widget.name === "API密钥") void refreshModels(node);
      return result;
    };
  }
}

function initializeNode(node, options = {}) {
  if (!node) return;
  restoreModelLock(node);
  applyNodePolicy(node, options);
  installWidgetCallbacks(node);
}

function schedulePolicy(node, options = {}) {
  setTimeout(() => initializeNode(node, options), 0);
}

app.registerExtension({
  name: "Shaobkj.MultiPromptEnhancerUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      initializeNode(this, { conservative: true });
      schedulePolicy(this);
      setTimeout(() => void refreshModels(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      schedulePolicy(this);
      setTimeout(() => void refreshModels(this), 0);
      return result;
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function () {
      const result = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
      schedulePolicy(this);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      schedulePolicy(this);
      return result;
    };
  },
  nodeCreated(node) {
    if (node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS) initializeNode(node, { conservative: true });
  },
  loadedGraphNode(node) {
    if (node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS) schedulePolicy(node);
  },
});
