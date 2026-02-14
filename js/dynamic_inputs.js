
import { app } from "/scripts/app.js";

const DYNAMIC_NODES = [
    "Shaobkj_APINode",
    "🤖图像生成",
    "Shaobkj_Sora_Video", 
    "Shaobkj_Veo_Video",
    "🤖 Shaobkj -Sora视频",
    "🤖 Shaobkj -Veo视频",
    "Shaobkj_ConcurrentImageEdit_Sender",
    "🤖并发-编辑-图像驱动",
    "Shaobkj_APINode_Batch",
    "🤖并发-编辑-文本驱动",
    "Shaobkj_LLM_App",
    "🤖LLM应用",
];
const SHAOBKJ_NODE_TYPES = [
    "Shaobkj_APINode",
    "Shaobkj_APINode_Batch",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
    "Shaobkj_ConcurrentImageEdit_Sender",
    "Shaobkj_LLM_App",
    "Shaobkj_Load_Image_Path",
    "Shaobkj_Load_Batch_Images",
    "Shaobkj_Image_Save",
    "Shaobkj_Fixed_Seed",
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
    "🤖 Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "🤖 Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },

    // ⚡ 极速系列 (效率与工具) - Matrix Green
    "Shaobkj_ConcurrentImageEdit_Sender": { color: "#00C853", bgcolor: "#003311" },
    "🤖并发-编辑-图像驱动": { color: "#00C853", bgcolor: "#003311" }
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
        if (title.includes("Sora") || title.includes("Veo")) return THEME_CONFIG["Shaobkj_Sora_Video"];
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
                    syncSeedControl(node);
                    if (shouldManageDynamicInputsByNode(node)) {
                        manageInputs(node);
                    } else {
                        cleanupDynamicInputs(node);
                    }
                }
            }
        };
        setTimeout(tick, 200);

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
                    syncSeedControl(this);
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
                    if (needsDynamicInputs) {
                        manageInputs(this);
                    } else {
                        cleanupDynamicInputs(this);
                    }
                }, 50);
                
                return r;
            }
        }
    }
});
