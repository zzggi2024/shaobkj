import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "Shaobkj_GPT2Test_Node";
const API_KEY_WIDGET = "API密钥";
const API_BASE_WIDGET = "API地址";
const MODEL_WIDGET = "模型选择";
const MODEL_SELECT_WIDGET = "可用模型列表";
const FETCH_BUTTON = "获取模型列表";
const DEFAULT_MODELS = ["gpt-image-2"];
let lastPointer = { x: window.innerWidth / 2, y: 72 };

window.addEventListener("pointerdown", (event) => {
	lastPointer = { x: event.clientX, y: event.clientY };
}, true);

function findWidget(node, name) {
	return node.widgets?.find((widget) => widget.name === name);
}

function showToast(message, isError = false) {
	const toast = document.createElement("div");
	toast.textContent = message;
	const left = Math.min(Math.max(lastPointer.x + 12, 12), window.innerWidth - 220);
	const top = Math.min(Math.max(lastPointer.y + 12, 12), window.innerHeight - 60);
	toast.style.cssText = [
		"position:fixed",
		`left:${left}px`,
		`top:${top}px`,
		"z-index:99999",
		"padding:8px 14px",
		"border-radius:8px",
		"font-size:13px",
		"color:#fff",
		`background:${isError ? "rgba(220,38,38,.94)" : "rgba(22,163,74,.94)"}`,
		"box-shadow:0 8px 24px rgba(0,0,0,.25)",
	].join(";");
	document.body.appendChild(toast);
	setTimeout(() => toast.remove(), 1000);
}

function setWidgetValue(widget, value) {
	widget.value = value;
	if (widget.inputEl) {
		widget.inputEl.value = value;
	}
}

function setModelOptions(node, models, selectedValue, updateModelWidget = false) {
	const modelWidget = findWidget(node, MODEL_WIDGET);
	const selectWidget = findWidget(node, MODEL_SELECT_WIDGET);
	const uniqueModels = Array.from(new Set((models || []).filter(Boolean)));
	const options = uniqueModels.length ? uniqueModels : DEFAULT_MODELS;
	const currentModel = String(modelWidget?.value || "").trim();
	const selected = selectedValue || (currentModel && options.includes(currentModel) ? currentModel : options[0]) || "gpt-image-2";

	if (selectWidget) {
		selectWidget.options = selectWidget.options || {};
		selectWidget.options.values = options;
		setWidgetValue(selectWidget, selected);
	}
	if (updateModelWidget && modelWidget) {
		setWidgetValue(modelWidget, selected);
	}
}

function getLinkedWidgetValue(node, inputName) {
	const input = node.inputs?.find((item) => item.name === inputName);
	const linkId = input?.link;
	const link = linkId != null ? app.graph.links?.[linkId] : null;
	const sourceNode = link ? app.graph.getNodeById(link.origin_id) : null;
	if (!sourceNode) {
		return "";
	}

	const sourceOutput = sourceNode.outputs?.[link.origin_slot];
	const sourceNames = [sourceOutput?.name, inputName].filter(Boolean);
	for (const name of sourceNames) {
		const widget = findWidget(sourceNode, name);
		const value = String(widget?.value || "").trim();
		if (value) {
			return value;
		}
	}

	const values = (sourceNode.widgets || []).map((widget) => String(widget.value || "").trim()).filter(Boolean);
	if (inputName === API_BASE_WIDGET) {
		return values.find((value) => /^https?:\/\//i.test(value)) || "";
	}
	if (inputName === API_KEY_WIDGET) {
		return values.find((value) => !/^https?:\/\//i.test(value)) || "";
	}
	return values[0] || "";
}

function getWidgetOrLinkedValue(node, name) {
	const widgetValue = String(findWidget(node, name)?.value || "").trim();
	return widgetValue || getLinkedWidgetValue(node, name);
}

async function fetchModels(node, buttonWidget) {
	const modelWidget = findWidget(node, MODEL_WIDGET);
	const apiKey = getWidgetOrLinkedValue(node, API_KEY_WIDGET);
	const apiBase = getWidgetOrLinkedValue(node, API_BASE_WIDGET);
	const useProxy = false;
	if (!modelWidget) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-gpt-2-测试] 未找到模型选择控件");
		return;
	}
	if (!apiKey || !apiBase) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-gpt-2-测试] API密钥或API地址为空", { apiKeySet: Boolean(apiKey), apiBase });
		return;
	}

	const oldName = buttonWidget.name;
	buttonWidget.name = "正在获取...";
	node.setDirtyCanvas?.(true, true);

	try {
		const response = await api.fetchApi("/shaobkj/test_api/models", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ api_key: apiKey, api_base: apiBase, use_proxy: useProxy }),
		});
		const data = await response.json();
		if (!response.ok) {
			throw new Error(data?.error || `模型列表获取失败：HTTP ${response.status}`);
		}
		const models = Array.isArray(data.models) ? data.models.filter(Boolean) : [];
		if (!models.length) {
			throw new Error("请到后台查看具体错误");
		}
		const currentModel = String(modelWidget.value || "").trim();
		const selectedModel = currentModel && models.includes(currentModel) ? currentModel : models[0];
		setModelOptions(node, models, selectedModel, !currentModel || !models.includes(currentModel));
		showToast("模型更新成功");
	} catch (error) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-gpt-2-测试]", error);
	} finally {
		buttonWidget.name = oldName;
		node.setDirtyCanvas?.(true, true);
	}
}

function setupModelFetcher(node) {
	const modelWidget = findWidget(node, MODEL_WIDGET);
	if (!modelWidget || node.__shaobkjGpt2TestModelFetcherReady) {
		return;
	}
	node.__shaobkjGpt2TestModelFetcherReady = true;
	setModelOptions(node, DEFAULT_MODELS, modelWidget.value || "gpt-image-2", false);

	const buttonWidget = node.addWidget("button", FETCH_BUTTON, null, () => fetchModels(node, buttonWidget));
	buttonWidget.serialize = false;
	const selectWidget = node.addWidget("combo", MODEL_SELECT_WIDGET, modelWidget.value || DEFAULT_MODELS[0], (value) => {
		setWidgetValue(modelWidget, value);
	}, { values: DEFAULT_MODELS });
	selectWidget.serialize = false;
}

app.registerExtension({
	name: "Shaobkj.GPT2TestModelList",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_NAME) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			setupModelFetcher(this);
			return result;
		};
	},
});
