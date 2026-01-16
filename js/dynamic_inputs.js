
import { app } from "/scripts/app.js";

const DYNAMIC_NODES = [
    "Shaobkj_APINode",
    "Shaobkj_Reverse_Node",
    "Shaobkj_Sora_Video",
    "Shaobkj_Veo_Video",
    "🤖 Shaobkj API Generator",
    "🤖 Shaobkj 反推",
    "🤖 Shaobkj -Sora视频",
    "🤖 Shaobkj -Veo视频",
    "Shaobkj API Generator",
    "Shaobkj 反推",
    "Shaobkj -Sora视频",
    "Shaobkj -Veo视频",
];
const INPUT_PREFIX = "image_";
const MIN_INPUTS = 2;
let started = false;
const LONG_SIDE_WIDGET_NAME = "长边设置";
const LONG_SIDE_WIDGET_LABEL = "输入图像-长边设置";
const SHAOBKJ_NODE_COLOR = "#6C88B8";
const SHAOBKJ_NODE_BGCOLOR = "#1E2633";

function isShaobkjRuntimeNode(node) {
    const t = node?.type;
    const title = node?.title;
    if (t && DYNAMIC_NODES.includes(t)) {
        return true;
    }
    if (title && typeof title === "string" && title.toLowerCase().includes("shaobkj")) {
        return true;
    }
    return false;
}

function manageInputs(node) {
    if (!node.inputs) {
        node.inputs = [];
    }

    const imageInputs = [];
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name && node.inputs[i].name.startsWith(INPUT_PREFIX)) {
            imageInputs.push(node.inputs[i]);
        }
    }

    imageInputs.sort((a, b) => {
        const idxA = parseInt(a.name.replace(INPUT_PREFIX, ""));
        const idxB = parseInt(b.name.replace(INPUT_PREFIX, ""));
        return idxA - idxB;
    });

    let highestConnectedIndex = 0;
    for (const input of imageInputs) {
        const idx = parseInt(input.name.replace(INPUT_PREFIX, ""));
        if (input.link !== null && input.link !== undefined && input.link !== -1) {
            if (idx > highestConnectedIndex) {
                highestConnectedIndex = idx;
            }
        }
    }

    let targetCount = Math.max(highestConnectedIndex + 1, MIN_INPUTS);
    
    for (let i = 1; i <= targetCount; i++) {
        const name = `${INPUT_PREFIX}${i}`;
        const existingIndex = node.findInputSlot ? node.findInputSlot(name) : -1;
        
        if (existingIndex === -1) {
            node.addInput(name, "IMAGE");
        }
    }
    
    let currentMaxIndex = 0;
    if (imageInputs.length > 0) {
        const currentInputs = node.inputs.filter(inp => inp.name.startsWith(INPUT_PREFIX));
        currentInputs.sort((a, b) => {
             const idxA = parseInt(a.name.replace(INPUT_PREFIX, ""));
             const idxB = parseInt(b.name.replace(INPUT_PREFIX, ""));
             return idxA - idxB;
        });
        
        if (currentInputs.length > 0) {
            currentMaxIndex = parseInt(currentInputs[currentInputs.length - 1].name.replace(INPUT_PREFIX, ""));
        }
    }

    if (currentMaxIndex > targetCount) {
        for (let i = currentMaxIndex; i > targetCount; i--) {
            const name = `${INPUT_PREFIX}${i}`;
            const inputIndex = node.findInputSlot ? node.findInputSlot(name) : -1;
            
            if (inputIndex !== -1) {
                const input = node.inputs[inputIndex];
                if (input.link === null) {
                    node.removeInput(inputIndex);
                }
            }
        }
    }

    node.onResize?.(node.size);
    node.setDirtyCanvas(true, true);
    setupLinkWidget(node);
    setupLongSideWidget(node);
    setupNodeStyle(node);
}

function setupLinkWidget(node) {
    if (!node.widgets) return;
    const index = node.widgets.findIndex(w => w.name === "API申请地址");
    if (index === -1) return;

    const widget = node.widgets[index];
    if (widget.type === "button" && widget.label === "🔗 打开 API 申请地址" && widget.callback) {
        return;
    }

    const url = "https://yhmx.work/login?expired=true";
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
}

function setupLongSideWidget(node) {
    if (!node.widgets) return;
    if (!isShaobkjRuntimeNode(node)) return;
    const widget = findWidget(node, LONG_SIDE_WIDGET_NAME);
    if (!widget) return;
    widget.label = LONG_SIDE_WIDGET_LABEL;
}

function setupNodeStyle(node) {
    if (!isShaobkjRuntimeNode(node)) return;
    if (node.color !== SHAOBKJ_NODE_COLOR) node.color = SHAOBKJ_NODE_COLOR;
    if (node.bgcolor !== SHAOBKJ_NODE_BGCOLOR) node.bgcolor = SHAOBKJ_NODE_BGCOLOR;
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
        const tick = () => {
            const graph = app?.graph;
            const nodes = graph?._nodes;
            if (!nodes || !Array.isArray(nodes)) {
                return;
            }
            for (const node of nodes) {
                if (isShaobkjRuntimeNode(node)) {
                    manageInputs(node);
                }
            }
        };
        setInterval(tick, 250);
        setTimeout(tick, 200);
    },
    async init(app) {
        return this.setup(app);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isShaobkjNode =
            DYNAMIC_NODES.includes(nodeData?.name) ||
            (nodeData?.category && nodeData.category.startsWith("🤖shaobkj-APIbox"));
        if (isShaobkjNode) {
            nodeType.prototype.color = SHAOBKJ_NODE_COLOR;
            nodeType.prototype.bgcolor = SHAOBKJ_NODE_BGCOLOR;
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    setupNodeStyle(this);
                    manageInputs(this);
                }, 50);
                
                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, slot) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                if (type === 1) {
                    setTimeout(() => {
                        manageInputs(this);
                    }, 50);
                }
                
                return r;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    manageInputs(this);
                }, 50);
                
                return r;
            }
        }
    }
});
