
import { app } from "/scripts/app.js";

const DYNAMIC_NODES = [
    "Shaobkj_APINode",
    "文本-图像生成",
    "🤖图像生成",
    "Shaobkj_Reverse_Node",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
    "🤖 Shaobkj 反推",
    "🤖 Shaobkj -Sora视频",
    "🤖 Shaobkj -Veo视频",
    "Shaobkj 反推",
    "Shaobkj -Sora视频",
    "Shaobkj -Veo视频",
    "Shaobkj_ConcurrentImageEdit_Sender",
    "🤖并发-编辑-发送端",
    "Shaobkj_APINode_Batch",
    "🤖并发-文本-图像生成",
];
const SHAOBKJ_NODE_TYPES = [
    "Shaobkj_APINode",
    "Shaobkj_APINode_Batch",
    "Shaobkj_Reverse_Node",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
    "Shaobkj_ConcurrentImageEdit_Sender",
];
const MIN_INPUTS = 2;
let started = false;
const LONG_SIDE_WIDGET_NAME = "长边设置";
const LONG_SIDE_WIDGET_LABEL = "输入图像-长边设置";

// 🎨 Shaobkj Cyber-Spectrum Theme Definition
const THEME_CONFIG = {
    // 🔮 创世系列 (图像生成) - Electric Violet
    "Shaobkj_APINode": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "Shaobkj_APINode_Batch": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "文本-图像生成": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "🤖图像生成": { color: "#7D24A6", bgcolor: "#1E0A29" },
    "🤖并发-文本-图像生成": { color: "#7D24A6", bgcolor: "#1E0A29" },

    // 🎬 导演系列 (视频生成) - Future Blue
    "Shaobkj_Sora_Video": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj_Veo_Video": { color: "#0091EA", bgcolor: "#001A2E" },
    "🤖 Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "🤖 Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Sora视频": { color: "#0091EA", bgcolor: "#001A2E" },
    "Shaobkj -Veo视频": { color: "#0091EA", bgcolor: "#001A2E" },

    // ⚡ 极速系列 (效率与工具) - Matrix Green
    "Shaobkj_ConcurrentImageEdit_Sender": { color: "#00C853", bgcolor: "#003311" },
    "Shaobkj_Reverse_Node": { color: "#00C853", bgcolor: "#003311" },
    "🤖 Shaobkj 反推": { color: "#00C853", bgcolor: "#003311" },
    "🤖并发-编辑-发送端": { color: "#00C853", bgcolor: "#003311" },
    "Shaobkj 反推": { color: "#00C853", bgcolor: "#003311" }
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
    if (t && (SHAOBKJ_NODE_TYPES.includes(t) || DYNAMIC_NODES.includes(t))) {
        return true;
    }
    if (title && typeof title === "string" && title.toLowerCase().includes("shaobkj")) {
        return true;
    }
    return false;
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

function setupLinkWidget(node) {
    if (!node.widgets) return;
    const index = node.widgets.findIndex(w => w.name === "API申请地址");
    if (index === -1) return;

    const widget = node.widgets[index];
    if (widget.type === "button" && widget.label === "🔗 打开 API 申请地址" && widget.callback) {
        return false;
    }

    const defaultUrl = "https://yhmx.work/login?expired=true";
    const urlValue = typeof widget.value === "string" ? widget.value.trim() : "";
    const url = urlValue && (urlValue.startsWith("http://") || urlValue.startsWith("https://")) ? urlValue : defaultUrl;
    node.widgets.splice(index, 1);
    const newWidget = node.addWidget("button", "API申请地址", null, () => {
        window.open(url, "_blank");
    });

    newWidget.name = "API申请地址";
    newWidget.label = "🔗 打开 API 申请地址";
    newWidget.tooltip = "打开 API 申请页面";
    newWidget.serialize = false;

    const lastIndex = node.widgets.length - 1;
    if (lastIndex !== index) {
        node.widgets.splice(lastIndex, 1);
        node.widgets.splice(index, 0, newWidget);
    }

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
                    if (shouldManageDynamicInputsByNode(node)) {
                        manageInputs(node);
                    } else if (node.inputs && Array.isArray(node.inputs) && node.inputs.length) {
                        let changed = false;
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
                }
            }
        };
        setTimeout(tick, 200);
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
                if (needsDynamicInputs) manageInputs(this, true);

                setTimeout(() => {
                    setupNodeStyle(this);
                    setupLinkWidget(this);
                    setupLongSideWidget(this);
                    // Check again, still onlyAdd=true to be safe during potential heavy load
                    if (needsDynamicInputs) manageInputs(this, true);
                }, 50);
                
                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, slot) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                if (needsDynamicInputs && type === 1) {
                    setTimeout(() => {
                        manageInputs(this);
                    }, 50);
                }
                
                return r;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                if (needsDynamicInputs) {
                    setTimeout(() => {
                        manageInputs(this);
                    }, 50);
                }
                
                return r;
            }
        }
    }
});
