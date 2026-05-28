import { app } from "../../scripts/app.js";

const NODE_NAME = "Shaobkj_GPT2Edits_Node";
const MODEL_WIDGET_NAME = "模型选择";
const DEFAULT_MODEL = "gpt-image-2-all";
const BUTTON_NAME = "默认模型";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setupDefaultModelButton(node) {
    if (!node.widgets || findWidget(node, BUTTON_NAME)) {
        return;
    }

    const button = node.addWidget("button", BUTTON_NAME, "Default", () => {
        const modelWidget = findWidget(node, MODEL_WIDGET_NAME);
        if (modelWidget) {
            modelWidget.value = DEFAULT_MODEL;
        }
        node.setDirtyCanvas?.(true, true);
    });
    button.label = "默认模型";
    button.serialize = false;
    button.tooltip = "点击后将模型选择恢复为默认模型 gpt-image-2-all";
}

app.registerExtension({
    name: "Shaobkj.GPT2EditsDefaultModel",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupDefaultModelButton(this);
            return result;
        };
    },
});
