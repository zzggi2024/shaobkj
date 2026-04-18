
import { app } from "/scripts/app.js";

const DYNAMIC_NODES = [
    "Shaobkj_APINode",
    "🤖图像生成",
    "Shaobkj_Sora_Video", 
    "Shaobkj_Veo_Video",
    "Shaobkj_SD20_Video",
    "🤖 Shaobkj -Sora视频",
    "🤖 Shaobkj -Veo视频",
    "🎬 SD_2.0视频",
    "Shaobkj_ConcurrentImageEdit_Sender",
    "🤖并发-编辑-图像驱动",
    "Shaobkj_GroupedConcurrentImageEdit",
    "🧩组合并发",
    "Shaobkj_APINode_Batch",
    "🤖并发-编辑-文本驱动",
    "Shaobkj_LLM_App",
    "🤖LLM应用",
    "Shaobkj_NanoBanana_Prompt",
    "🤖香蕉专属提示词",
];
const SHAOBKJ_NODE_TYPES = [
    "Shaobkj_APINode",
    "Shaobkj_APINode_Batch",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
    "Shaobkj_SD20_Video",
    "Shaobkj_ConcurrentImageEdit_Sender",
    "Shaobkj_GroupedConcurrentImageEdit",
    "Shaobkj_LLM_App",
    "Shaobkj_NanoBanana_Prompt",
    "Shaobkj_Load_Image_Path",
    "Shaobkj_Load_Batch_Images",
    "Shaobkj_Image_Save",
    "Shaobkj_Fixed_Seed",
    "Shaobkj_LoadImageListFromDir",
    "Shaobkj_Text_Process",
];
const MIN_INPUTS = 2;
let started = false;
const LONG_SIDE_WIDGET_NAME = "长边设置";
const LONG_SIDE_WIDGET_LABEL = "输入图像-长边设置";
const UPLOAD_LABEL_TEXT = "选择上传图像";
const SEED_WIDGET_NAME = "seed";
const CONTROL_WIDGET_NAME = "control_after_generate";

// 🎨 Shaobkj Cyber-Spectrum Theme Definition
const THEME_CONFIG = {
    // 🔮 创世系列 (图像生成) - Electric Violet
    "Shaobkj_APINode": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "Shaobkj_APINode_Batch": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "文本-图像生成": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "🤖图像生成": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "🤖并发-编辑-文本驱动": { color: "#7D24A6", bgcolor: "#1E0A29" },

    // 🎬 导演系列 (视频生成) - Future Blue
    "Shaobkj_Sora_Video": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj_Veo_Video": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj_SD20_Video": { color: "#0091EA", bgcolor: "#001A2E" },
    "🤖 Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "🤖 Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "🎬 SD_2.0视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },

    // ⚡ 极速系列 (效率与工具) - Matrix Green
    "Shaobkj_ConcurrentImageEdit_Sender": { color: "#00C853", bgcolor: "#003311" },
    "🤖并发-编辑-图像驱动": { color: "#00C853", bgcolor: "#003311" },
    "Shaobkj_GroupedConcurrentImageEdit": { color: "#00C853", bgcolor: "#003311" },
    "🧩组合并发": { color: "#00C853", bgcolor: "#003311" }
};

const DEFAULT_THEME = { color: "#006600", bgcolor: "#003300" }; // Fallback
const SHAOBKJ_TITLE_TEXT_COLOR = "#FF0000";

function getThemeForNode(node) {
    const type = node.comfyClass || node.type;
    const title = node.title;
    
    // Try by type/comfyClass
    if (type && THEME_CONFIG[type]) return THEME_CONFIG[type];
    
    // Try by title
    if (title && THEME_CONFIG[title]) return THEME_CONFIG[title];
    
    // Fallback logic for aliases not explicitly in map but having known keywords
    if (title) {
        if (title.includes("Sora") || title.includes("Veo") || title.includes("SD_2.0")) return THEME_CONFIG["Shaobkj_Sora_Video"];
        if (title.includes("反推") || title.includes("编辑")) return THEME_CONFIG["Shaobkj_ConcurrentImageEdit"];
        if (title.includes("图像生成")) return THEME_CONFIG["Shaobkj_APINode"];
        if (title.includes("桥接")) return THEME_CONFIG["Shaobkj_HTTP_Load_Image"];
    }
    
    return DEFAULT_THEME;
}

function shouldManageDynamicInputsByNodeData(nodeData) {
    const name = nodeData?.name || "";
    const displayName = nodeData?.display_name || nodeData?.displayName || "";
    if (DYNAMIC_NODES.includes(name)) {
        return true;
    }
    if (displayName && DYNAMIC_NODES.includes(displayName)) {
        return true;
    }
    return false;
}

function shouldManageDynamicInputsByNode(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    if (t && DYNAMIC_NODES.includes(t)) {
        return true;
    }
    if (title && typeof title === "string" && DYNAMIC_NODES.includes(title)) {
        return true;
    }
    return false;
}

function getDynamicInputSpec(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    const k = `${t} ${title}`.toLowerCase();
    if (k.includes("video_edit") || k.includes("视频编辑")) {
        return { prefix: "video_", slotType: "VIDEO" };
    }
    return { prefix: "image_", slotType: "IMAGE" };
}

function isShaobkjRuntimeNode(node) {
    const t = node?.type;
    const title = node?.title;
    
    // Debug logging
    if (t && (SHAOBKJ_NODE_TYPES.includes(t) || DYNAMIC_NODES.includes(t))) {
        console.log("[Shaobkj Debug] Node matched by type:", t);
        return true;
    }
    if (title && typeof title === "string" && title.toLowerCase().includes("shaobkj")) {
        console.log("[Shaobkj Debug] Node matched by title:", title);
        return true;
    }
    
    // Debug: Log what types we're checking against
    if (t && !SHAOBKJ_NODE_TYPES.includes(t) && !DYNAMIC_NODES.includes(t)) {
        console.log("[Shaobkj Debug] Node NOT matched:", t, "SHAOBKJ_NODE_TYPES:", SHAOBKJ_NODE_TYPES, "DYNAMIC_NODES:", DYNAMIC_NODES);
    }
    
    return false;
}

function isShaobkjLoadImageNode(node) {
    const t = node?.type;
    const title = node?.title;
    return t === "Shaobkj_Load_Image_Path" || title === "🤖加载图像";
}

function isShaobkjLoadImageListNode(node) {
    const t = node?.type;
    const title = node?.title;
    return t === "Shaobkj_LoadImageListFromDir" || title === "🤖加载图像列表(路径)";
}

function isShaobkjLoadBatchImagesNode(node) {
    const t = node?.type;
    const title = node?.title;
    return t === "Shaobkj_Load_Batch_Images" || title === "🤖批量加载图像(路径)" || title === "🤖批量加载图片 (Path)";
}

function isShaobkjNanoBananaNode(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    return t === "Shaobkj_NanoBanana_Prompt" || (typeof title === "string" && title.includes("香蕉专属提示词"));
}

function isShaobkjImageSaveNode(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    return t === "Shaobkj_Image_Save" || (typeof title === "string" && title.includes("图像保存"));
}

function isShaobkjTextProcessNode(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    return t === "Shaobkj_Text_Process" || (typeof title === "string" && title.includes("文本处理"));
}

function isShaobkjLoopTriggerNode(node) {
    const t = node?.type || "";
    const title = node?.title || "";
    return t === "Shaobkj_Loop_Trigger" || (typeof title === "string" && title.includes("循环触发"));
}

function findWidgetByNames(node, names) {
    if (!node?.widgets || !Array.isArray(node.widgets)) return null;
    for (const name of names) {
        const widget = node.widgets.find((w) => w && (w.name === name || w.label === name));
        if (widget) return widget;
    }
    return null;
}

function setWidgetDisabledState(widget, disabled) {
    let changed = false;
    if (widget.disabled !== disabled) {
        widget.disabled = disabled;
        changed = true;
    }
    if (!widget.options) {
        widget.options = {};
        changed = true;
    }
    if (widget.options.disabled !== disabled) {
        widget.options.disabled = disabled;
        changed = true;
    }
    if (widget.options.readOnly !== disabled) {
        widget.options.readOnly = disabled;
        changed = true;
    }
    return changed;
}

function setWidgetHiddenState(widget, hidden) {
    if (!widget) return false;
    let changed = false;
    const nextType = hidden ? "hidden" : (widget.__shaobkjOriginalType || widget.type || "number");
    if (!widget.__shaobkjOriginalType) {
        widget.__shaobkjOriginalType = widget.type || "number";
    }
    if (!widget.__shaobkjOriginalComputeSize && typeof widget.computeSize === "function") {
        widget.__shaobkjOriginalComputeSize = widget.computeSize;
    }
    if (!widget.__shaobkjHiddenComputeSize) {
        widget.__shaobkjHiddenComputeSize = () => [0, -4];
    }
    if (hidden) {
        if (widget.type !== nextType) {
            widget.type = nextType;
            changed = true;
        }
        if (widget.computeSize !== widget.__shaobkjHiddenComputeSize) {
            widget.computeSize = widget.__shaobkjHiddenComputeSize;
            changed = true;
        }
    } else {
        if (widget.type !== nextType) {
            widget.type = nextType;
            changed = true;
        }
        const originalComputeSize = widget.__shaobkjOriginalComputeSize;
        if (originalComputeSize && widget.computeSize !== originalComputeSize) {
            widget.computeSize = originalComputeSize;
            changed = true;
        }
    }
    return changed;
}

function getLinkedTextValue(node, inputName) {
    try {
        if (!node || !node.graph || !Array.isArray(node.inputs)) {
            return null;
        }
        const input = node.inputs.find((item) => item && item.name === inputName);
        if (!input || input.link == null) {
            return null;
        }
        const link = node.graph.links && node.graph.links[input.link];
        if (!link) {
            return null;
        }
        const originNode = node.graph.getNodeById ? node.graph.getNodeById(link.origin_id) : null;
        if (!originNode || !Array.isArray(originNode.widgets)) {
            return null;
        }
        for (const widget of originNode.widgets) {
            if (typeof widget?.value === "string") {
                return widget.value;
            }
        }
        return null;
    } catch {
        return null;
    }
}

function getTextProcessSourceText(node) {
    const linkedText = getLinkedTextValue(node, "文本");
    if (typeof linkedText === "string" && linkedText !== "") {
        return linkedText;
    }
    const textWidget = findWidgetByNames(node, ["文本"]);
    return textWidget ? String(textWidget.value ?? "") : "";
}

function getTextProcessMaxLineCount(node) {
    const textValue = getTextProcessSourceText(node);
    if (!textValue) return 0;
    const lines = textValue.split(/\r?\n/);
    return lines.filter((line) => String(line).trim() !== "").length;
}

function initializeTextProcessState(node) {
    if (!isShaobkjTextProcessNode(node) || !node.widgets) return false;
    const startWidget = findWidgetByNames(node, ["计数开始"]);
    const endWidget = findWidgetByNames(node, ["计数结束"]);
    const stateWidget = findWidgetByNames(node, ["当前执行编号状态"]);
    if (!startWidget || !endWidget || !stateWidget) return false;
    const maxLines = getTextProcessMaxLineCount(node);
    const startValue = Math.max(0, Number(startWidget.value ?? 0));
    const nextState = maxLines > 0 ? Math.min(startValue, maxLines - 1) : 0;
    let changed = false;
    if (Number(endWidget.value ?? 0) !== maxLines) {
        endWidget.value = maxLines;
        changed = true;
    }
    if (stateWidget.value !== nextState) {
        stateWidget.value = nextState;
        changed = true;
    }
    node.__shaobkjTextProcessLastText = getTextProcessSourceText(node);
    node.__shaobkjTextProcessInitialized = true;
    node.__shaobkjTextProcessListEnabled = Boolean(findWidgetByNames(node, ["列表"])?.value);
    node.__shaobkjTextProcessLastStartValue = startValue;
    if (changed) {
        node.onResize?.(node.size);
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupNanoBananaEditingMode(node) {
    if (!isShaobkjNanoBananaNode(node)) return false;
    if (!node.widgets) return false;
    const taskTypeWidget = findWidgetByNames(node, ["任务类型", "Task Type"]);
    if (!taskTypeWidget) return false;
    const taskValue = String(taskTypeWidget.value || "");
    const isEditingMode = taskValue.includes("Editing") || taskValue.includes("编辑模式");
    const styleWidget = findWidgetByNames(node, ["场景风格", "Scene Style"]);
    const brandWidget = findWidgetByNames(node, ["品牌名称", "Brand Name"]);
    let changed = false;
    if (styleWidget) {
        changed = setWidgetDisabledState(styleWidget, isEditingMode) || changed;
        if (styleWidget.options && Array.isArray(styleWidget.options.values)) {
            if (!styleWidget.__shaobkjOriginalValues) {
                styleWidget.__shaobkjOriginalValues = [...styleWidget.options.values];
            }
            if (isEditingMode) {
                const onlyCurrent = [styleWidget.value];
                const same = styleWidget.options.values.length === 1 && styleWidget.options.values[0] === styleWidget.value;
                if (!same) {
                    styleWidget.options.values = onlyCurrent;
                    changed = true;
                }
            } else if (Array.isArray(styleWidget.__shaobkjOriginalValues)) {
                const origin = styleWidget.__shaobkjOriginalValues;
                const same = styleWidget.options.values.length === origin.length && styleWidget.options.values.every((v, i) => v === origin[i]);
                if (!same) {
                    styleWidget.options.values = [...origin];
                    changed = true;
                }
            }
        }
    }
    if (brandWidget) {
        changed = setWidgetDisabledState(brandWidget, isEditingMode) || changed;
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupImageSaveCustomSizeMode(node) {
    if (!isShaobkjImageSaveNode(node)) return false;
    if (!node.widgets) return false;
    const enableWidget = findWidgetByNames(node, ["自定义尺寸"]);
    const widthWidget = findWidgetByNames(node, ["宽"]);
    const heightWidget = findWidgetByNames(node, ["高"]);
    if (!enableWidget || !widthWidget || !heightWidget) return false;
    const enabled = Boolean(enableWidget.value);
    let changed = false;
    changed = setWidgetDisabledState(widthWidget, !enabled) || changed;
    changed = setWidgetDisabledState(heightWidget, !enabled) || changed;
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupTextProcessListMode(node) {
    if (!isShaobkjTextProcessNode(node)) return false;
    if (!node.widgets || !Array.isArray(node.outputs) || node.outputs.length < 3) return false;
    const listWidget = findWidgetByNames(node, ["列表"]);
    const startWidget = findWidgetByNames(node, ["计数开始"]);
    const endWidget = findWidgetByNames(node, ["计数结束"]);
    const modeWidget = findWidgetByNames(node, ["mode"]);
    const stateWidget = findWidgetByNames(node, ["当前执行编号状态"]);
    const textWidget = findWidgetByNames(node, ["文本"]);
    if (!listWidget) return false;
    const totalOutput = node.outputs[1];
    const currentOutput = node.outputs[2];
    if (!totalOutput || !currentOutput) return false;
    const enabled = Boolean(listWidget.value);
    const totalOutputName = enabled ? "输出列表数" : "输出列表数（已禁用）";
    const currentOutputName = enabled ? "当前执行编号" : "当前执行编号（已禁用）";
    let changed = false;
    const startValue = Number(startWidget?.value ?? 0);
    if (totalOutput.name !== totalOutputName) {
        totalOutput.name = totalOutputName;
        changed = true;
    }
    if (totalOutput.label !== totalOutputName) {
        totalOutput.label = totalOutputName;
        changed = true;
    }
    if (totalOutput.disabled !== !enabled) {
        totalOutput.disabled = !enabled;
        changed = true;
    }
    if (currentOutput.name !== currentOutputName) {
        currentOutput.name = currentOutputName;
        changed = true;
    }
    if (currentOutput.label !== currentOutputName) {
        currentOutput.label = currentOutputName;
        changed = true;
    }
    if (currentOutput.disabled !== !enabled) {
        currentOutput.disabled = !enabled;
        changed = true;
    }
    if (startWidget) {
        changed = setWidgetDisabledState(startWidget, !enabled) || changed;
    }
    if (modeWidget) {
        changed = setWidgetDisabledState(modeWidget, !enabled) || changed;
        if (modeWidget.label !== "计数模式") {
            modeWidget.label = "计数模式";
            changed = true;
        }
    }
    if (endWidget) {
        changed = setWidgetDisabledState(endWidget, !enabled) || changed;
    }
    if (stateWidget) {
        changed = setWidgetHiddenState(stateWidget, true) || changed;
    }
    const previousEnabled = node.__shaobkjTextProcessListEnabled;
    const previousStartValue = node.__shaobkjTextProcessLastStartValue;
    if (stateWidget && startWidget) {
        const stateValue = Number(stateWidget.value ?? 0);
        const shouldInitializeState = node.__shaobkjTextProcessInitialized !== true
            || previousEnabled !== enabled
            || previousStartValue !== startValue;
        if (!Number.isNaN(startValue) && (shouldInitializeState || stateValue < startValue)) {
            stateWidget.value = startValue;
            changed = true;
        }
    }
    node.__shaobkjTextProcessInitialized = true;
    node.__shaobkjTextProcessListEnabled = enabled;
    node.__shaobkjTextProcessLastStartValue = startValue;
    if (typeof node.__shaobkjTextProcessLastText !== "string") {
        node.__shaobkjTextProcessLastText = getTextProcessSourceText(node);
    }
    if (changed) {
        node.onResize?.(node.size);
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupLoadImageListLocalization(node) {
    if (!isShaobkjLoadImageListNode(node) || !node.widgets) return false;
    let changed = false;
    const labelMap = {
        directory: "目录路径",
        image_load_cap: "加载数量上限",
        start_index: "起始索引",
        load_always: "每次重载",
        sort_method: "排序方式",
        include_subdirs: "包含子文件夹",
    };
    const sortValueMap = {
        numerical: "数字顺序",
        alphabetical: "字母顺序",
        date: "修改时间",
    };
    for (const w of node.widgets) {
        const name = typeof w.name === "string" ? w.name : "";
        const nextLabel = labelMap[name];
        if (nextLabel && w.label !== nextLabel) {
            w.label = nextLabel;
            changed = true;
        }
        if (name === "sort_method" && w.options && Array.isArray(w.options.values)) {
            const nextValues = ["数字顺序", "字母顺序", "修改时间"];
            const current = String(w.value ?? "");
            const mapped = sortValueMap[current] || current;
            const sameValues = w.options.values.length === nextValues.length && w.options.values.every((v, i) => v === nextValues[i]);
            if (!sameValues) {
                w.options.values = nextValues;
                changed = true;
            }
            if (nextValues.includes(mapped) && w.value !== mapped) {
                w.value = mapped;
                changed = true;
            }
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupLoadBatchImagesLocalization(node) {
    if (!isShaobkjLoadBatchImagesNode(node) || !node.widgets) return false;
    let changed = false;
    const labelMap = {
        directory: "目录路径",
        image_load_cap: "加载数量上限",
        start_index: "起始索引",
        load_always: "每次重载",
        sort_method: "排序方式",
    };
    const sortValueMap = {
        numerical: "数字顺序",
        alphabetical: "字母顺序",
        date: "修改时间",
    };
    const nextValues = ["数字顺序", "字母顺序", "修改时间"];
    for (const w of node.widgets) {
        const name = typeof w.name === "string" ? w.name : "";
        const nextLabel = labelMap[name];
        if (nextLabel && w.label !== nextLabel) {
            w.label = nextLabel;
            changed = true;
        }
        if (name === "sort_method" && w.options && Array.isArray(w.options.values)) {
            const current = String(w.value ?? "");
            const mapped = sortValueMap[current] || current;
            const sameValues = w.options.values.length === nextValues.length && w.options.values.every((v, i) => v === nextValues[i]);
            if (!sameValues) {
                w.options.values = nextValues;
                changed = true;
            }
            if (nextValues.includes(mapped) && w.value !== mapped) {
                w.value = mapped;
                changed = true;
            }
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

let shaobkjTitleHookInstalled = false;
function installShaobkjTitleColorHook() {
    if (shaobkjTitleHookInstalled) return true;
    const proto =
        globalThis?.LGraphNode?.prototype ||
        globalThis?.LiteGraph?.LGraphNode?.prototype;
    if (!proto) return false;

    const candidates = ["drawTitle", "draw_title", "_drawTitle", "drawTitleBar", "draw_title_bar"];
    for (const name of candidates) {
        const original = proto?.[name];
        if (typeof original !== "function") continue;
        if (original.__shaobkj_wrapped) {
            shaobkjTitleHookInstalled = true;
            return true;
        }

        const wrapped = function (ctx) {
            if (!isShaobkjRuntimeNode(this) || !ctx || typeof ctx.fillText !== "function") {
                return original.apply(this, arguments);
            }

            const originalFillText = ctx.fillText;
            ctx.fillText = function () {
                const old = ctx.fillStyle;
                ctx.fillStyle = SHAOBKJ_TITLE_TEXT_COLOR;
                const r = originalFillText.apply(this, arguments);
                ctx.fillStyle = old;
                return r;
            };
            try {
                return original.apply(this, arguments);
            } finally {
                ctx.fillText = originalFillText;
            }
        };
        wrapped.__shaobkj_wrapped = true;
        proto[name] = wrapped;
        shaobkjTitleHookInstalled = true;
        return true;
    }
    return false;
}

let shaobkjTooltipHookInstalled = false;
function installShaobkjTooltipHook() {
    // Always try to install/update the hook on the active canvas instance
    const canvas = app.canvas || (globalThis.LGraphCanvas && globalThis.LGraphCanvas.active_canvas);
    if (!canvas) return;

    // Avoid double wrapping the SAME instance
    if (canvas.__shaobkj_hook_active) return;
    
    const originalShowTooltip = canvas.showTooltip;

    // Hook ShowTooltip: Reposition tooltips for Shaobkj nodes to avoid blocking controls
    canvas.showTooltip = function(text, x, y) {
        // If we are over a Shaobkj node, reposition the tooltip to avoid blocking controls
        if (this.node_over && isShaobkjRuntimeNode(this.node_over)) {
            // Force use of Mouse Coordinates (this.last_mouse) if available
            // This overrides LiteGraph's default calculation which might place it to the right
            let targetX = x;
            let targetY = y;

            if (this.last_mouse) {
                targetX = this.last_mouse[0];
                targetY = this.last_mouse[1];
            }

            // Shift Y down by 30px to appear below the cursor
            // This prevents blocking the current widget's input area
            const finalX = targetX;
            const finalY = targetY + 30;
            
            // console.log("[Shaobkj Tooltip] Repositioned to below mouse: ", finalX, finalY);
            return originalShowTooltip.call(this, text, finalX, finalY);
        }
        
        // Otherwise, use original position
        return originalShowTooltip.apply(this, arguments);
    };

    canvas.__shaobkj_hook_active = true;
    shaobkjTooltipHookInstalled = true;
    console.log("[Shaobkj] Tooltip hook installed on active canvas");
}

function manageInputs(node, onlyAdd = false) {
    try {
        if (!node.inputs) {
            node.inputs = [];
        }

        let changed = false;
        const spec = getDynamicInputSpec(node);
        if (!spec) return;
        const { prefix, slotType } = spec;

        const imageInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (inp && inp.name && (inp.name.startsWith(prefix) || (prefix && inp.name.indexOf(prefix) === 0))) {
                imageInputs.push(inp);
            }
        }

        imageInputs.sort((a, b) => {
            const nameA = String(a.name || "");
            const nameB = String(b.name || "");
            const idxA = parseInt(String(nameA).replace(prefix, "") || "0");
            const idxB = parseInt(String(nameB).replace(prefix, "") || "0");
            return idxA - idxB;
        });

        let highestConnectedIndex = 0;
        for (const input of imageInputs) {
            const name = String(input.name || "");
            const idx = parseInt(String(name).replace(prefix, "") || "0");
            if (input.link !== null && input.link !== undefined && input.link !== -1) {
                if (idx > highestConnectedIndex) {
                    highestConnectedIndex = idx;
                }
            }
        }

        let targetCount = Math.max(highestConnectedIndex + 1, MIN_INPUTS);
        
        for (let i = 1; i <= targetCount; i++) {
            const name = `${prefix}${i}`;
            const existingIndex = node.findInputSlot ? node.findInputSlot(name) : -1;
            
            if (existingIndex === -1) {
                node.addInput(name, slotType);
                changed = true;
            }
        }
        
        // During initialization (onlyAdd=true), we should NOT remove any inputs.
        if (onlyAdd) {
            if (setupLinkWidget(node)) changed = true;
            if (setupLongSideWidget(node)) changed = true;
            if (setupNodeStyle(node)) changed = true;
            if (changed) {
                node.onResize?.(node.size);
                node.setDirtyCanvas(true, true);
            }
            return;
        }
        
        let currentMaxIndex = 0;
        if (imageInputs.length > 0) {
            const currentInputs = node.inputs.filter(inp => inp && inp.name && inp.name.startsWith(prefix));
            currentInputs.sort((a, b) => {
                 const nameA = String(a.name || "");
                 const nameB = String(b.name || "");
                 const idxA = parseInt(String(nameA).replace(prefix, "") || "0");
                 const idxB = parseInt(String(nameB).replace(prefix, "") || "0");
                 return idxA - idxB;
            });
            
            if (currentInputs.length > 0) {
                const lastName = String(currentInputs[currentInputs.length - 1].name || "");
                currentMaxIndex = parseInt(String(lastName).replace(prefix, "") || "0");
            }
        }

        if (currentMaxIndex > targetCount) {
            for (let i = currentMaxIndex; i > targetCount; i--) {
                const name = `${prefix}${i}`;
                const inputIndex = node.findInputSlot ? node.findInputSlot(name) : -1;
                
                if (inputIndex !== -1) {
                    const input = node.inputs[inputIndex];
                    if (input && input.link === null) {
                        node.removeInput(inputIndex);
                        changed = true;
                    }
                }
            }
        }

        if (setupLinkWidget(node)) {
            changed = true;
        }

        if (setupLongSideWidget(node)) {
            changed = true;
        }
        if (setupNodeStyle(node)) {
            changed = true;
        }
        if (changed) {
            node.onResize?.(node.size);
            node.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error("[Shaobkj] Error in manageInputs:", e);
    }
}

function cleanupDynamicInputs(node) {
    if (!node.inputs || !Array.isArray(node.inputs) || !node.inputs.length) return;
    let changed = false;
    // Iterate backwards to safely remove
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const n = node.inputs[i]?.name || "";
        if (n.startsWith("image_") || n.startsWith("video_")) {
            node.removeInput(i);
            changed = true;
        }
    }
    if (changed) {
        node.onResize?.(node.size);
        node.setDirtyCanvas(true, true);
    }
}

function setupLinkWidget(node) {
    if (!node.widgets) {
        node.widgets = [];
    }
    const nodeType = node?.type || "";
    const nodeTitle = node?.title || "";
    if (isShaobkjTextProcessNode(node)) {
        const legacyIndex = node.widgets.findIndex(w => w.name === "API申请地址");
        if (legacyIndex >= 0) {
            node.widgets.splice(legacyIndex, 1);
        }
        const index = node.widgets.findIndex(w => w.name === "初始化");
        if (index === -1) {
            const newWidget = node.addWidget("button", "初始化", "Init", () => {
                initializeTextProcessState(node);
            });
            newWidget.name = "初始化";
            newWidget.label = "⟳ 初始化";
            newWidget.tooltip = "按当前文本重新计算去空行后的最大列表行数，并初始化计数";
            newWidget.serialize = false;
            node.setDirtyCanvas(true, true);
            return true;
        }
        const widget = node.widgets[index];
        const isLast = index === node.widgets.length - 1;
        const isCorrect = widget.type === "button" && widget.label === "⟳ 初始化" && widget.callback;
        widget.callback = () => {
            initializeTextProcessState(node);
        };
        widget.tooltip = "按当前文本重新计算去空行后的最大列表行数，并初始化计数";
        if (isCorrect && isLast) {
            return false;
        }
        if (isCorrect && !isLast) {
            node.widgets.splice(index, 1);
            node.widgets.push(widget);
            node.setDirtyCanvas(true, true);
            return true;
        }
        node.widgets.splice(index, 1);
        const newWidget = node.addWidget("button", "初始化", "Init", () => {
            initializeTextProcessState(node);
        });
        newWidget.name = "初始化";
        newWidget.label = "⟳ 初始化";
        newWidget.tooltip = "按当前文本重新计算去空行后的最大列表行数，并初始化计数";
        newWidget.serialize = false;
        node.setDirtyCanvas(true, true);
        return true;
    }
    if (isShaobkjLoopTriggerNode(node)) {
        const existingIndex = node.widgets.findIndex(w => w.name === "API申请地址");
        if (existingIndex >= 0) {
            node.widgets.splice(existingIndex, 1);
            node.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    }
    if (nodeType === "Shaobkj_FreeColor" || (typeof nodeTitle === "string" && nodeTitle.includes("自由调色"))) {
        const existingIndex = node.widgets.findIndex(w => w.name === "API申请地址");
        if (existingIndex >= 0) {
            node.widgets.splice(existingIndex, 1);
            node.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    }
    if (nodeType === "Shaobkj_QuickMark" || (typeof nodeTitle === "string" && nodeTitle.includes("快速标记"))) {
        const existingIndex = node.widgets.findIndex(w => w.name === "API申请地址");
        if (existingIndex >= 0) {
            node.widgets.splice(existingIndex, 1);
            node.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    }
    if (nodeType === "Shaobkj_FontStyleSelector" || (typeof nodeTitle === "string" && nodeTitle.includes("字体风格提示词选择器"))) {
        const existingIndex = node.widgets.findIndex(w => w.name === "API申请地址");
        if (existingIndex >= 0) {
            node.widgets.splice(existingIndex, 1);
            node.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    }
    const index = node.widgets.findIndex(w => w.name === "API申请地址");
    const defaultUrl = "https://yhmx.work/login?expired=true";

    // Case 1: Not found - Add it
    if (index === -1) {
        const newWidget = node.addWidget("button", "API申请地址", "Open URL", () => {
            window.open(defaultUrl, "_blank");
        });
        newWidget.name = "API申请地址";
        newWidget.label = "🔗 打开 API 申请地址";
        newWidget.tooltip = "打开 API 申请页面";
        newWidget.serialize = false;
        node.setDirtyCanvas(true, true);
        return true;
    }

    // Case 2: Found - Check properties and position
    const widget = node.widgets[index];
    const isLast = index === node.widgets.length - 1;
    const isCorrect = widget.type === "button" && widget.label === "🔗 打开 API 申请地址" && widget.callback;

    // If correct and at the bottom, do nothing
    if (isCorrect && isLast) {
        return false;
    }

    // If correct but not at bottom, move to end
    if (isCorrect && !isLast) {
        node.widgets.splice(index, 1);
        node.widgets.push(widget);
        node.setDirtyCanvas(true, true);
        return true;
    }

    // Case 3: Incorrect properties - Replace and ensure at bottom
    const urlValue = typeof widget.value === "string" ? widget.value.trim() : "";
    const url = urlValue && (urlValue.startsWith("http://") || urlValue.startsWith("https://")) ? urlValue : defaultUrl;
    node.widgets.splice(index, 1);
    const newWidget = node.addWidget("button", "API申请地址", "Open URL", () => {
        window.open(url, "_blank");
    });

    newWidget.name = "API申请地址";
    newWidget.label = "🔗 打开 API 申请地址";
    newWidget.tooltip = "打开 API 申请页面";
    newWidget.serialize = false;

    node.setDirtyCanvas(true, true);
    return true;
}

function setupLongSideWidget(node) {
    if (!node.widgets) return;
    if (!isShaobkjRuntimeNode(node)) return;
    const widget = findWidget(node, LONG_SIDE_WIDGET_NAME);
    if (!widget) return;
    if (widget.label !== LONG_SIDE_WIDGET_LABEL) {
        widget.label = LONG_SIDE_WIDGET_LABEL;
        return true;
    }
    return false;
}

function setupUploadButtonLabel(node) {
    if (!isShaobkjLoadImageNode(node)) return false;
    if (!node.widgets) return false;
    let changed = false;
    for (const w of node.widgets) {
        const label = typeof w.label === "string" ? w.label : "";
        const name = typeof w.name === "string" ? w.name : "";
        const value = typeof w.value === "string" ? w.value : "";
        const text = (label || name || value).trim().toLowerCase();
        if (text === "upload") {
            if (w.label !== UPLOAD_LABEL_TEXT) {
                w.label = UPLOAD_LABEL_TEXT;
                changed = true;
            }
            if (w.name === "upload") {
                w.name = UPLOAD_LABEL_TEXT;
                changed = true;
            }
            if (w.value === "upload") {
                w.value = UPLOAD_LABEL_TEXT;
                changed = true;
            }
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function setupNodeStyle(node) {
    if (!isShaobkjRuntimeNode(node)) return;
    
    const theme = getThemeForNode(node);
    
    let changed = false;
    if (node.color !== theme.color) {
        node.color = theme.color;
        changed = true;
    }
    if (node.bgcolor !== theme.bgcolor) {
        node.bgcolor = theme.bgcolor;
        changed = true;
    }
    if (node.title_text_color !== SHAOBKJ_TITLE_TEXT_COLOR) {
        node.title_text_color = SHAOBKJ_TITLE_TEXT_COLOR;
        changed = true;
    }
    return changed;
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function getWidgetControlMode() {
    const settings = app?.ui?.settings;
    if (settings && typeof settings.getSettingValue === "function") {
        const keys = [
            "widget_control_mode",
            "node_widget_control_mode",
            "component_control_mode",
            "control_mode",
        ];
        for (const k of keys) {
            const v = settings.getSettingValue(k);
            if (v === "before" || v === "after") return v;
        }
    }
    const raw = localStorage.getItem("Comfy.Settings");
    if (raw) {
        const obj = JSON.parse(raw);
        for (const k of Object.keys(obj || {})) {
            const key = String(k).toLowerCase();
            const v = obj[k];
            if (key.includes("control") && key.includes("mode")) {
                if (v === "before" || v === "after") return v;
            }
        }
    }
    return "before";
}

function syncSeedControl(node) {
    if (!node.widgets) return false;
    let changed = false;
    const mode = getWidgetControlMode();
    const seedWidget = findWidget(node, SEED_WIDGET_NAME);
    if (seedWidget) {
        if (!seedWidget.options) seedWidget.options = {};
        const shouldAfter = mode === "after";
        if (seedWidget.options.control_after_generate !== shouldAfter) {
            seedWidget.options.control_after_generate = shouldAfter;
            changed = true;
        }
    }
    const controlWidget = findWidget(node, CONTROL_WIDGET_NAME);
    if (controlWidget && controlWidget.options && Array.isArray(controlWidget.options.values)) {
        const label = mode === "after" ? "生成后控制" : "生成前控制";
        let nextValue = mode;
        if (controlWidget.options.values.includes(label)) {
            nextValue = label;
        } else if (!controlWidget.options.values.includes(mode)) {
            nextValue = controlWidget.value;
        }
        if (controlWidget.value !== nextValue) {
            controlWidget.value = nextValue;
            changed = true;
        }
        if (controlWidget.label !== label) {
            controlWidget.label = label;
            changed = true;
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

function sanitizeNumericWidgets(node) {
    if (!node.widgets) return false;
    let changed = false;
    for (const w of node.widgets) {
        const type = typeof w.type === "string" ? w.type : "";
        const isNumberWidget = type === "number" || type === "slider";
        const raw = w.value;
        const num = typeof raw === "number" ? raw : (typeof raw === "string" && raw.trim() ? Number(raw) : NaN);
        const isNaNValue = Number.isNaN(num) || !Number.isFinite(num);
        const isForceNumeric = w.name === "并发间隔";
        if ((isNumberWidget || isForceNumeric) && isNaNValue) {
            const opts = w.options || {};
            let next = opts.default;
            if (next === undefined && isForceNumeric) next = 1.0;
            if (next === undefined) next = opts.min;
            if (next === undefined) next = 0;
            w.value = next;
            changed = true;
        }
    }
    if (changed) {
        node.setDirtyCanvas(true, true);
    }
    return changed;
}

app.registerExtension({
    name: "Shaobkj.DynamicInputs",
    async setup(app) {
        if (app.__shaobkjDynamicInputsInstalled) {
            return;
        }
        if (started) {
            return;
        }
        app.__shaobkjDynamicInputsInstalled = true;

        started = true;
        const ensureTitleHook = () => {
            installShaobkjTooltipHook();
            if (!installShaobkjTitleColorHook()) {
                requestAnimationFrame(ensureTitleHook);
            }
        };
        requestAnimationFrame(ensureTitleHook);
        const tick = () => {
            const graph = app?.graph;
            const nodes = graph?._nodes;
            if (!nodes || !Array.isArray(nodes)) {
                return;
            }
            for (const node of nodes) {
                if (isShaobkjRuntimeNode(node)) {
                    installShaobkjTitleColorHook();
                    setupNodeStyle(node);
                    setupLinkWidget(node);
                    setupLongSideWidget(node);
                    setupUploadButtonLabel(node);
                    setupLoadImageListLocalization(node);
                    setupLoadBatchImagesLocalization(node);
                    setupNanoBananaEditingMode(node);
                    setupTextProcessListMode(node);
                    if (isShaobkjTextProcessNode(node)) {
                        const currentSourceText = getTextProcessSourceText(node);
                        if (node.__shaobkjTextProcessLastText !== currentSourceText) {
                            initializeTextProcessState(node);
                        }
                    }
                    syncSeedControl(node);
                    sanitizeNumericWidgets(node);
                    if (shouldManageDynamicInputsByNode(node)) {
                        manageInputs(node);
                    } else {
                        cleanupDynamicInputs(node);
                    }
                }
            }
        };
        setTimeout(tick, 200);
        window.setInterval(tick, 400);

        import("/scripts/api.js").then(({ api }) => {
            api.addEventListener("shaobkj.llm.warning", (evt) => {
                const msg = (evt && evt.detail && evt.detail.message) ? evt.detail.message : "⚠️ LLM 输出为空";
                if (app?.ui?.dialog?.show) {
                    app.ui.dialog.show(msg);
                    setTimeout(() => { app.ui.dialog.close(); }, 1500);
                } else {
                    alert(msg);
                }
            });
            api.addEventListener("shaobkj.image_save.warning", (evt) => {
                const msg = (evt && evt.detail && evt.detail.message) ? evt.detail.message : "⚠️ 保存格式与颜色模式不兼容，已自动转换为 RGB";
                if (app?.ui?.dialog?.show) {
                    app.ui.dialog.show(msg);
                    setTimeout(() => { app.ui.dialog.close(); }, 2200);
                } else {
                    alert(msg);
                }
            });
            api.addEventListener("shaobkj.node_feedback", (evt) => {
                const detail = evt && evt.detail ? evt.detail : null;
                const nodeId = detail && detail.node_id ? String(detail.node_id) : "";
                const widgetName = detail && detail.widget_name ? String(detail.widget_name) : "";
                const value = detail ? detail.value : undefined;
                const node = app?.graph?._nodes_by_id?.[nodeId];
                if (!node || !Array.isArray(node.widgets)) {
                    return;
                }
                const widget = node.widgets.find((w) => w && w.name === widgetName);
                if (!widget) {
                    return;
                }
                widget.value = value;
                node.setDirtyCanvas?.(true, true);
            });
            api.addEventListener("shaobkj.add_queue", async () => {
                await app.queuePrompt(0, 1);
            });
        }).catch(() => {
        });
    },
    async init(app) {
        return this.setup(app);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isShaobkjCategory = nodeData?.category && nodeData.category.startsWith("🤖shaobkj-APIbox");
        const needsDynamicInputs = shouldManageDynamicInputsByNodeData(nodeData);
        if (isShaobkjCategory || needsDynamicInputs) {
            // Apply theme prototype
            const theme = getThemeForNode({ comfyClass: nodeData.name, title: nodeData.display_name });
            nodeType.prototype.color = theme.color;
            nodeType.prototype.bgcolor = theme.bgcolor;
            nodeType.prototype.title_text_color = SHAOBKJ_TITLE_TEXT_COLOR;
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Run manageInputs immediately to ensure slots exist for link restoration
                // Use onlyAdd=true to prevent removing slots during initial load
                if (needsDynamicInputs) {
                    manageInputs(this, true);
                } else {
                    cleanupDynamicInputs(this);
                }

                setTimeout(() => {
                    setupNodeStyle(this);
                    setupLinkWidget(this);
                    setupLongSideWidget(this);
                    setupUploadButtonLabel(this);
                    setupLoadImageListLocalization(this);
                    setupLoadBatchImagesLocalization(this);
                    setupNanoBananaEditingMode(this);
                    setupImageSaveCustomSizeMode(this);
                    setupTextProcessListMode(this);
                    initializeTextProcessState(this);
                    syncSeedControl(this);
                    sanitizeNumericWidgets(this);
                    // Check again, still onlyAdd=true to be safe during potential heavy load
                    if (needsDynamicInputs) {
                        manageInputs(this, true);
                    } else {
                        cleanupDynamicInputs(this);
                    }
                }, 50);
                
                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, slot) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                if (type === 1) {
                    setTimeout(() => {
                        if (needsDynamicInputs) {
                            manageInputs(this);
                        } else {
                            cleanupDynamicInputs(this);
                        }
                    }, 50);
                }
                
                return r;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupNanoBananaEditingMode(this);
                    setupImageSaveCustomSizeMode(this);
                    setupTextProcessListMode(this);
                    setupLoadBatchImagesLocalization(this);
                    if (needsDynamicInputs) {
                        manageInputs(this);
                    } else {
                        cleanupDynamicInputs(this);
                    }
                }, 50);
                
                return r;
            };

            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, oldValue, widget) {
                const r = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
                if (name === "计数开始") {
                    const stateWidget = findWidgetByNames(this, ["当前执行编号状态"]);
                    if (stateWidget && stateWidget.value !== value) {
                        stateWidget.value = value;
                    }
                }
                if (name === "文本") {
                    const textValue = String(value ?? "");
                    if (this.__shaobkjTextProcessLastText !== textValue) {
                        initializeTextProcessState(this);
                    }
                }
                setupNanoBananaEditingMode(this);
                setupImageSaveCustomSizeMode(this);
                setupTextProcessListMode(this);
                setupLoadBatchImagesLocalization(this);
                return r;
            };
        }
    }
});
