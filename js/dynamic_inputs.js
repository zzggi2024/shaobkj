
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
];
const SHAOBKJ_NODE_TYPES = [
    "Shaobkj_APINode",
    "Shaobkj_APINode_Batch",
    "Shaobkj_Reverse_Node",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
];
const MIN_INPUTS = 2;
let started = false;
const LONG_SIDE_WIDGET_NAME = "长边设置";
const LONG_SIDE_WIDGET_LABEL = "输入图像-长边设置";
const SHAOBKJ_NODE_COLOR = "#6C88B8";
const SHAOBKJ_NODE_BGCOLOR = "#1E2633";
const SHAOBKJ_TITLE_TEXT_COLOR = "#FF0000";

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

function manageInputs(node) {
    if (!node.inputs) {
        node.inputs = [];
    }

    let changed = false;

    const { prefix, slotType } = getDynamicInputSpec(node);

    const imageInputs = [];
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name && node.inputs[i].name.startsWith(prefix)) {
            imageInputs.push(node.inputs[i]);
        }
    }

    imageInputs.sort((a, b) => {
        const idxA = parseInt(a.name.replace(prefix, ""));
        const idxB = parseInt(b.name.replace(prefix, ""));
        return idxA - idxB;
    });

    let highestConnectedIndex = 0;
    for (const input of imageInputs) {
        const idx = parseInt(input.name.replace(prefix, ""));
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
    
    // Fix: Protect existing links from being lost during recreation
    // Sometimes addInput/removeInput might shift indices, but LiteGraph usually handles it.
    // However, if we are too aggressive, we might break things.
    // The logic above only removes inputs if they are > targetCount AND have no link.
    // So it should be safe.
    // But let's double check if "recreate node" triggers something else.
    // If "Recreate Node" is used, the node is destroyed and a new one created.
    // The new node might not have inputs initially, or have default inputs.
    // manageInputs adds them back.
    // If links are restored by ComfyUI app, they are restored by index or name.
    // If we change inputs async (setTimeout), link restoration might fail if slots don't exist yet.
    // But onNodeCreated calls manageInputs immediately (in setTimeout 50ms).
    // ComfyUI restores links after node creation.
    // If we wait 50ms, links might try to connect to non-existent slots?
    // Let's try to run manageInputs synchronously if possible, or very short timeout.
    
    let currentMaxIndex = 0;
    if (imageInputs.length > 0) {
        const currentInputs = node.inputs.filter(inp => inp.name.startsWith(prefix));
        currentInputs.sort((a, b) => {
             const idxA = parseInt(a.name.replace(prefix, ""));
             const idxB = parseInt(b.name.replace(prefix, ""));
             return idxA - idxB;
        });
        
        if (currentInputs.length > 0) {
            currentMaxIndex = parseInt(currentInputs[currentInputs.length - 1].name.replace(prefix, ""));
        }
    }

    if (currentMaxIndex > targetCount) {
        for (let i = currentMaxIndex; i > targetCount; i--) {
            const name = `${prefix}${i}`;
            const inputIndex = node.findInputSlot ? node.findInputSlot(name) : -1;
            
            if (inputIndex !== -1) {
                const input = node.inputs[inputIndex];
                if (input.link === null) {
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
    let changed = false;
    if (node.color !== SHAOBKJ_NODE_COLOR) {
        node.color = SHAOBKJ_NODE_COLOR;
        changed = true;
    }
    if (node.bgcolor !== SHAOBKJ_NODE_BGCOLOR) {
        node.bgcolor = SHAOBKJ_NODE_BGCOLOR;
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
            nodeType.prototype.color = SHAOBKJ_NODE_COLOR;
            nodeType.prototype.bgcolor = SHAOBKJ_NODE_BGCOLOR;
            nodeType.prototype.title_text_color = SHAOBKJ_TITLE_TEXT_COLOR;
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Run manageInputs immediately to ensure slots exist for link restoration
                // But keep setTimeout for style updates just in case
                if (needsDynamicInputs) manageInputs(this);

                setTimeout(() => {
                    setupNodeStyle(this);
                    setupLinkWidget(this);
                    setupLongSideWidget(this);
                    // Check again
                    if (needsDynamicInputs) manageInputs(this);
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
