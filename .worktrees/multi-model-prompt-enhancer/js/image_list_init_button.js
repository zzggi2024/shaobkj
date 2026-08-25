import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODES = new Set(["Shaobkj_LoadImageListFromDir", "Shaobkj_BatchInput"]);
const INIT_VALUE_NAME = "初始化";
const INIT_BUTTON_NAME = "⟳ 初始化";
const OLD_WIDGET_NAMES = new Set(["每次重载", "load_always", "打开 API 申请地址", "打开API申请地址"]);

function removeOldWidgets(node) {
	if (!Array.isArray(node?.widgets)) {
		return;
	}
	node.widgets = node.widgets.filter((widget) => !OLD_WIDGET_NAMES.has(widget?.name) && widget?.label !== "打开 API 申请地址" && widget?.label !== "打开API申请地址");
}

function findWidget(node, name) {
	return Array.isArray(node?.widgets) ? node.widgets.find((widget) => widget?.name === name) : null;
}

function showMessage(message, duration = 3000) {
	if (app?.ui?.dialog?.show) {
		app.ui.dialog.show(message);
		setTimeout(() => app.ui.dialog.close(), duration);
		return;
	}
	alert(message);
}

function collectStringValues(value, result = []) {
	if (typeof value === "string" && value.trim()) {
		result.push(value.trim());
	} else if (Array.isArray(value)) {
		for (const item of value) {
			collectStringValues(item, result);
		}
	}
	return result;
}

function looksLikePath(value) {
	const text = String(value ?? "").trim();
	return Boolean(text) && (/[a-zA-Z]:[\\/]/.test(text) || text.includes("\\") || text.includes("/"));
}

function getLinkedTextValue(node, inputName) {
	try {
		const input = node.inputs?.find((item) => item?.name === inputName);
		if (!input || input.link == null) {
			return "";
		}
		const link = node.graph?.links?.[input.link];
		const originNode = link && node.graph?.getNodeById ? node.graph.getNodeById(link.origin_id) : null;
		if (!originNode) {
			return "";
		}
		const values = [];
		for (const widget of originNode.widgets || []) {
			collectStringValues(widget?.value, values);
		}
		collectStringValues(originNode.widgets_values, values);
		return values.find(looksLikePath) || values[0] || "";
	} catch {
		return "";
	}
}

function hasLinkedInput(node, inputName) {
	const input = node.inputs?.find((item) => item?.name === inputName);
	return Boolean(input && input.link != null);
}

function resolveFolderPath(node) {
	const inputNames = node?.comfyClass === "Shaobkj_LoadImageListFromDir" ? ["directory", "目录路径"] : ["目录路径", "directory"];
	if (node?.comfyClass === "Shaobkj_LoadImageListFromDir") {
		for (const inputName of inputNames) {
			if (hasLinkedInput(node, inputName)) {
				const linkedValue = getLinkedTextValue(node, inputName);
				if (linkedValue) {
					return linkedValue;
				}
			}
		}
	}
	const pathWidget = findWidget(node, "目录路径") || findWidget(node, "directory");
	const widgetValue = String(pathWidget?.value ?? "").trim();
	if (widgetValue) {
		return widgetValue;
	}
	for (const inputName of inputNames) {
		const linkedValue = getLinkedTextValue(node, inputName);
		if (linkedValue) {
			return linkedValue;
		}
	}
	return "";
}

function getIncludeSubdirsValue(node) {
	const widget = findWidget(node, "包含子文件夹") || findWidget(node, "include_subdirs");
	return widget ? Boolean(widget.value) : true;
}

async function scanFolder(path, includeSubdirs = true) {
	const response = await api.fetchApi("/shaobkj/loop_trigger/scan", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ path, include_subdirs: includeSubdirs }),
	});
	const data = await response.json();
	if (!response.ok || data?.status !== "success") {
		throw new Error(data?.message || "扫描文件夹失败");
	}
	return data;
}

function ensureInitButton(node) {
	if (!node?.addWidget) {
		return;
	}

	let initValueWidget = findWidget(node, INIT_VALUE_NAME);
	if (!initValueWidget) {
		initValueWidget = node.addWidget("number", INIT_VALUE_NAME, 0, null, { min: 0, max: 0xffffffffffffffff, step: 1 });
	}
	initValueWidget.type = "number";
	initValueWidget.computeSize = () => [0, -4];
	initValueWidget.serialize = true;

	let initButton = findWidget(node, INIT_BUTTON_NAME);
	if (!initButton) {
		initButton = node.addWidget("button", INIT_BUTTON_NAME, "", async () => {
			initValueWidget.value = Number(initValueWidget.value || 0) + 1;
			node.setDirtyCanvas?.(true, true);
			try {
				const folderPath = resolveFolderPath(node);
				if (!folderPath) {
					throw new Error("请先填写目录路径");
				}
				const result = await scanFolder(folderPath, getIncludeSubdirsValue(node));
				showMessage(`成功加载 ${Number(result.total ?? 0)} 张图片`, 3000);
			} catch (error) {
				showMessage(String(error?.message || error), 3000);
			}
		});
	}
	initButton.label = INIT_BUTTON_NAME;
	initButton.serialize = false;
}

function enhanceInitWidget(node) {
	removeOldWidgets(node);
	ensureInitButton(node);
	node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
	name: "Shaobkj.ImageListInitButton",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!TARGET_NODES.has(nodeData?.name)) {
			return;
		}

		const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
			enhanceInitWidget(this);
			return result;
		};

		const originalOnConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
			enhanceInitWidget(this);
			return result;
		};
	},
});
