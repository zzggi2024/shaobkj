import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const API_KEY_WIDGET = "API密钥";
const FETCH_BUTTON = "获取模型列表";
const MODEL_SELECT_WIDGET = "可用模型列表";
const FIXED_API_BASE = "https://yhmx.work";
const STORAGE_PREFIX = "Shaobkj.FixedApiModelList";

const GEMINI_IMAGE_MODEL_FILTER = (model) => {
	const name = String(model || "").toLowerCase();
	return name.includes("gemini") && (name.includes("image") || name.includes("preview"));
};

const NODE_CONFIGS = {

	Shaobkj_APINode: { endpoint: "/shaobkj/test_api/models", modelWidget: "模型选择", defaults: ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "智能加载"], filter: GEMINI_IMAGE_MODEL_FILTER },
	Shaobkj_APINode_Batch: { endpoint: "/shaobkj/test_api/models", modelWidget: "模型选择", defaults: ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "智能加载"], filter: GEMINI_IMAGE_MODEL_FILTER },
	Shaobkj_GPT2Edits_Node: { endpoint: "/shaobkj/gpt_image/models", modelWidget: "模型选择", defaults: ["gpt-image-2", "gpt-image-2-all"] },
	Shaobkj_GPTImage2_Batch_Node: { endpoint: "/shaobkj/gpt_image/models", modelWidget: "模型选择", defaults: ["gpt-image-2", "gpt-image-2-all"] },

	Shaobkj_ConcurrentImageEdit_Sender: { endpoint: "/shaobkj/test_api/models", modelWidget: "模型选择", defaults: ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "智能加载"], filter: GEMINI_IMAGE_MODEL_FILTER },
	Shaobkj_GroupedConcurrentImageEdit: { endpoint: "/shaobkj/test_api/models", modelWidget: "模型选择", defaults: ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "智能加载"], filter: GEMINI_IMAGE_MODEL_FILTER },
	Shaobkj_Doubao_Image: { endpoint: "/shaobkj/doubao_image/models", modelWidget: "模型选择", defaults: ["doubao-seedream-5-0-260128", "doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128"] },

	Shaobkj_LLM_App: { endpoint: "/shaobkj/llm_test/models", modelWidget: "模型选择", defaults: ["gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gpt-5.4-mini"] },
	Shaobkj_Media_Reverse_Prompt: { endpoint: "/shaobkj/media_reverse/models", modelWidget: "模型名称", defaults: ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"] },
	Shaobkj_NanoBanana_Prompt: { endpoint: "/shaobkj/llm_test/models", modelWidget: "模型选择", defaults: ["gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"] },

	Shaobkj_Grok_Video: { endpoint: "/shaobkj/grok_video/models", modelWidget: "模型", defaults: ["grok-imagine-video-1.5-preview", "grok-imagine-1.0-video", "grok-imagine-1.0-video-20s", "grok-imagine-1.0-video-30s"] },


	Shaobkj_SD20_Video: { endpoint: "/shaobkj/sd20_video/models", modelWidget: "模型", defaults: ["doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"] },
	Shaobkj_Seedance_Video: { endpoint: "/shaobkj/sd20_video/models", modelWidget: "模型", defaults: ["seedance-2.0-standard-t2v", "seedance-2.0-fast-t2v", "seedance-2.0-mini-t2v", "seedance-2.0-standard-i2v", "seedance-2.0-fast-i2v", "seedance-2.0-mini-i2v", "seedance-2.0-standard-multi", "seedance-2.0-fast-multi", "seedance-2.0-mini-multi", "seedance-2.0-global-standard-t2v", "seedance-2.0-global-fast-t2v", "seedance-2.0-global-mini-t2v", "seedance-2.0-global-standard-i2v", "seedance-2.0-global-fast-i2v", "seedance-2.0-global-mini-i2v", "seedance-2.0-global-standard-multi", "seedance-2.0-global-fast-multi", "seedance-2.0-global-mini-multi"] },

};

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

function updateComboValues(widget, values) {
	widget.options = widget.options || {};
	widget.options.values = values;
	if (widget.comboEl) {
		widget.comboEl.options.length = 0;
		for (const value of values) {
			widget.comboEl.add(new Option(value, value));
		}
	}
}

function getStorageKey(nodeTypeName) {
	return `${STORAGE_PREFIX}.${nodeTypeName}`;
}

function loadStoredModels(nodeTypeName) {
	try {
		const raw = localStorage.getItem(getStorageKey(nodeTypeName));
		const models = raw ? JSON.parse(raw) : [];
		return Array.isArray(models) ? models.filter(Boolean) : [];
	} catch (error) {
		console.warn("[Shaobkj-模型列表] 读取持久化模型列表失败", error);
		return [];
	}
}

function saveStoredModels(nodeTypeName, models) {
	const values = Array.from(new Set((models || []).filter(Boolean)));
	if (!values.length) {
		return;
	}
	localStorage.setItem(getStorageKey(nodeTypeName), JSON.stringify(values));
}


function filterModels(config, models) {
	const values = (models || []).filter(Boolean);
	return config.filter ? values.filter(config.filter) : values;
}

function setModelOptions(node, config, models, selectedValue) {

	const modelWidget = findWidget(node, config.modelWidget);
	const selectWidget = findWidget(node, MODEL_SELECT_WIDGET);
	const options = Array.from(new Set(filterModels(config, models)));
	const values = options.length ? options : config.defaults;
	const currentModel = String(modelWidget?.value || "").trim();
	const selected = selectedValue || (currentModel && values.includes(currentModel) ? currentModel : values[0]);

	if (selectWidget) {
		updateComboValues(selectWidget, values);
		setWidgetValue(selectWidget, selected);
	}
	if (modelWidget && selected) {
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
	const widgets = sourceNode.widgets || [];
	const preferredNames = [inputName, "API密钥", "api_key", "API Key", "key", "value", "字符串", "文本"];
	for (const name of preferredNames) {
		const widget = widgets.find((item) => item.name === name);
		const value = String(widget?.value || "").trim();
		if (value) {
			return value;
		}
	}
	if (widgets.length === 1) {
		return String(widgets[0].value || "").trim();
	}
	return "";
}


function hasLinkedInput(node, name) {
	const input = node.inputs?.find((item) => item.name === name);
	return input?.link != null;
}

function getWidgetOrLinkedValue(node, name) {
	const linkedValue = hasLinkedInput(node, name) ? getLinkedWidgetValue(node, name) : "";
	if (linkedValue) {
		return linkedValue;
	}
	return String(findWidget(node, name)?.value || "").trim();
}


async function fetchModels(node, config, buttonWidget, nodeTypeName) {
	if (node.__shaobkjFetchingModels) {
		return;
	}
	const modelWidget = findWidget(node, config.modelWidget);

	const apiKey = getWidgetOrLinkedValue(node, API_KEY_WIDGET);
	const proxyWidget = findWidget(node, "使用系统代理");
	const useProxy = proxyWidget ? Boolean(proxyWidget.value) : true;
	if (!modelWidget) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-模型列表] 未找到模型控件", config.modelWidget);
		return;
	}
	if (!apiKey) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-模型列表] API密钥为空");
		return;
	}

	const oldName = buttonWidget.name;
	node.__shaobkjFetchingModels = true;
	buttonWidget.name = "正在获取...";
	node.setDirtyCanvas?.(true, true);


	try {
		const response = await api.fetchApi(config.endpoint, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ api_key: apiKey, api_base: FIXED_API_BASE, use_proxy: useProxy }),
		});
		const data = await response.json();
		if (!response.ok) {
			throw new Error(data?.error || `模型列表获取失败：HTTP ${response.status}`);
		}
		const models = Array.isArray(data.models) ? data.models.filter(Boolean) : [];
		const filteredModels = filterModels(config, models);
		if (!filteredModels.length) {
			throw new Error("请到后台查看具体错误");
		}
		saveStoredModels(nodeTypeName, filteredModels);

		setModelOptions(node, config, filteredModels, filteredModels[0]);
		showToast("模型更新成功");

	} catch (error) {
		showToast("请到后台查看具体错误", true);
		console.warn("[Shaobkj-模型列表]", error);
	} finally {
		node.__shaobkjFetchingModels = false;
		buttonWidget.name = oldName;
		node.setDirtyCanvas?.(true, true);
	}

}

function applyStoredModelOptions(node, config, nodeTypeName) {
	const storedModels = loadStoredModels(nodeTypeName);
	if (!storedModels.length) {
		return;
	}
	setModelOptions(node, config, storedModels);
}

function setupModelFetcher(node, config, nodeTypeName) {

	const modelWidget = findWidget(node, config.modelWidget);
	if (!modelWidget || node.__shaobkjFixedApiModelFetcherReady) {
		return;
	}
	node.__shaobkjFixedApiModelFetcherReady = true;
	const storedModels = loadStoredModels(nodeTypeName);

	const values = storedModels.length ? storedModels : config.defaults;
	const selected = modelWidget.value || values[0];
	setModelOptions(node, config, values, selected);

	const buttonWidget = node.addWidget("button", FETCH_BUTTON, null, () => fetchModels(node, config, buttonWidget, nodeTypeName));

	buttonWidget.serialize = false;
	const selectWidget = node.addWidget("combo", MODEL_SELECT_WIDGET, selected, (value) => {
		setWidgetValue(modelWidget, value);
	}, { values });
	selectWidget.serialize = false;
}



app.registerExtension({
	name: "Shaobkj.FixedApiModelList",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		const config = NODE_CONFIGS[nodeData.name];
		if (!config) {
			return;
		}

		const nodeTypeName = nodeData.name;
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			setupModelFetcher(this, config, nodeTypeName);
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
			setTimeout(() => applyStoredModelOptions(this, config, nodeTypeName), 50);
			return result;
		};

	},

});
