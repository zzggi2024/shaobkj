import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS_NAME = "Shaobkj_Loop_Trigger";
const watchedNodes = new Map();
let pollTimer = null;

function findWidget(node, name) {
	return Array.isArray(node?.widgets) ? node.widgets.find((widget) => widget?.name === name) : null;
}

function setWidgetValue(node, name, value) {
	const widget = findWidget(node, name);
	if (!widget) {
		return;
	}
	widget.value = value;
	node.setDirtyCanvas?.(true, true);
}

function setWidgetHiddenState(widget, hidden) {
	if (!widget) {
		return;
	}
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
		widget.type = "hidden";
		widget.computeSize = widget.__shaobkjHiddenComputeSize;
	} else if (widget.__shaobkjOriginalComputeSize) {
		widget.type = widget.__shaobkjOriginalType;
		widget.computeSize = widget.__shaobkjOriginalComputeSize;
	}
}

function getLinkedTextValue(node, inputName) {
	try {
		if (!node || !node.graph || !Array.isArray(node.inputs)) {
			return "";
		}
		const input = node.inputs.find((item) => item && item.name === inputName);
		if (!input || input.link == null) {
			return "";
		}
		const link = node.graph.links && node.graph.links[input.link];
		if (!link) {
			return "";
		}
		const originNode = node.graph.getNodeById ? node.graph.getNodeById(link.origin_id) : null;
		if (!originNode || !Array.isArray(originNode.widgets)) {
			return "";
		}
		for (const widget of originNode.widgets) {
			if (typeof widget?.value === "string" && widget.value.trim()) {
				return widget.value.trim();
			}
		}
		return "";
	} catch {
		return "";
	}
}

function resolveFolderPath(node) {
	const pathWidget = findWidget(node, "文件夹路径");
	const widgetValue = String(pathWidget?.value ?? "").trim();
	if (widgetValue) {
		return widgetValue;
	}
	return getLinkedTextValue(node, "文件夹路径");
}

async function scanFolder(path) {
	const response = await api.fetchApi("/shaobkj/loop_trigger/scan", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ path }),
	});
	const data = await response.json();
	if (!response.ok || data?.status !== "success") {
		throw new Error(data?.message || "扫描文件夹失败");
	}
	return data;
}

function showMessage(message) {
	if (app?.ui?.dialog?.show) {
		app.ui.dialog.show(message);
		setTimeout(() => app.ui.dialog.close(), 1800);
		return;
	}
	alert(message);
}

async function initializeFolder(node, path) {
	const response = await api.fetchApi("/shaobkj/loop_trigger/initialize", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			path,
			unique_id: String(node?.id ?? ""),
		}),
	});
	const data = await response.json();
	if (!response.ok || data?.status !== "success") {
		throw new Error(data?.message || "初始化失败");
	}
	return data;
}

async function syncNodeFolderState(node, options = {}) {
	const { fromInit = false, silent = false } = options;
	const pathWidget = findWidget(node, "文件夹路径");
	const totalWidget = findWidget(node, "总数");
	const forceWidget = findWidget(node, "强制循环数");

	if (!pathWidget || !totalWidget || !forceWidget) {
		return;
	}

	const folderPath = resolveFolderPath(node);
	if (!folderPath) {
		if (fromInit && !silent) {
			throw new Error("请先填写文件夹路径");
		}
		return;
	}

	const result = fromInit ? await initializeFolder(node, folderPath) : await scanFolder(folderPath);
	if (!result?.exists || !result?.is_dir) {
		if (fromInit && !silent) {
			throw new Error("文件夹不存在或路径无效");
		}
		setWidgetValue(node, "总数", 0);
		setWidgetValue(node, "强制循环数", 0);
		setWidgetValue(node, "当前执行编号状态", 0);
		watchedNodes.set(String(node.id), {
			path: folderPath,
			signature: "",
			total: 0,
			initialized: false,
		});
		return;
	}

	const total = Number(result.total ?? 0);
	const previousTotal = Number(totalWidget.value ?? 0);
	const currentForceValue = Number(forceWidget.value ?? 0);
	if (Number(totalWidget.value ?? 0) !== total) {
		totalWidget.value = total;
	}
	if (currentForceValue <= 0 || currentForceValue === previousTotal) {
		forceWidget.value = total;
	}
	if (fromInit) {
		setWidgetValue(node, "当前执行编号状态", 0);
	}

	node.setDirtyCanvas?.(true, true);
	watchedNodes.set(String(node.id), {
		path: folderPath,
		signature: String(result.signature ?? ""),
		total,
		initialized: true,
	});

	if (!silent && fromInit) {
		showMessage(`已初始化，检测到 ${total} 张图片`);
	}
}

async function pollLoopTriggerNodes() {
	const nodesById = app?.graph?._nodes_by_id ?? {};
	for (const [nodeId, state] of Array.from(watchedNodes.entries())) {
		const node = nodesById[nodeId];
		if (!node) {
			watchedNodes.delete(nodeId);
			continue;
		}

		const stateInitialized = Boolean(state?.initialized);
		if (!stateInitialized) {
			continue;
		}

		const folderPath = resolveFolderPath(node);
		if (!folderPath) {
			continue;
		}

		try {
			const result = await scanFolder(folderPath);
			const signature = String(result.signature ?? "");
			const total = Number(result.total ?? 0);
			if (signature !== state.signature || total !== state.total || folderPath !== state.path) {
				setWidgetValue(node, "总数", total);
				watchedNodes.set(nodeId, {
					path: folderPath,
					signature,
					total,
					initialized: true,
				});
			}
		} catch (error) {
			console.warn("[Shaobkj] 循环触发扫描失败:", error);
		}
	}
}

function startPolling() {
	if (pollTimer !== null) {
		return;
	}
	pollTimer = window.setInterval(() => {
		pollLoopTriggerNodes().catch((error) => {
			console.warn("[Shaobkj] 循环触发轮询失败:", error);
		});
	}, 2000);
}

function enhanceLoopTriggerNode(node) {
	if (node.__shaobkjLoopTriggerEnhanced) {
		return;
	}
	node.__shaobkjLoopTriggerEnhanced = true;

	const pathWidget = findWidget(node, "文件夹路径");
	const legacyApiWidgetIndex = Array.isArray(node.widgets) ? node.widgets.findIndex((widget) => widget?.name === "API申请地址") : -1;
	if (legacyApiWidgetIndex >= 0) {
		node.widgets.splice(legacyApiWidgetIndex, 1);
	}
	const legacyCountWidgetIndex = Array.isArray(node.widgets) ? node.widgets.findIndex((widget) => widget?.name === "计数") : -1;
	if (legacyCountWidgetIndex >= 0) {
		node.widgets.splice(legacyCountWidgetIndex, 1);
	}
	const legacyStartWidgetIndex = Array.isArray(node.widgets) ? node.widgets.findIndex((widget) => widget?.name === "计数开始") : -1;
	if (legacyStartWidgetIndex >= 0) {
		node.widgets.splice(legacyStartWidgetIndex, 1);
	}
	const legacyEndWidgetIndex = Array.isArray(node.widgets) ? node.widgets.findIndex((widget) => widget?.name === "计数结束") : -1;
	if (legacyEndWidgetIndex >= 0) {
		node.widgets.splice(legacyEndWidgetIndex, 1);
	}
	const internalStateWidget = findWidget(node, "当前执行编号状态");
	if (internalStateWidget) {
		setWidgetHiddenState(internalStateWidget, true);
	}

	const existingInitButton = findWidget(node, "初始化");
	if (!existingInitButton) {
		const initButton = node.addWidget("button", "初始化", "Init", () => {
			syncNodeFolderState(node, { fromInit: true }).catch((error) => {
				showMessage(String(error?.message || error));
			});
		});
		initButton.name = "初始化";
		initButton.label = "⟳ 初始化";
		initButton.serialize = false;
		initButton.tooltip = "首次点击后读取文件夹图片总数，并同步强制循环数默认值";
	} else {
		existingInitButton.label = "⟳ 初始化";
		existingInitButton.serialize = false;
		existingInitButton.callback = () => {
			syncNodeFolderState(node, { fromInit: true }).catch((error) => {
				showMessage(String(error?.message || error));
			});
		};
	}

	if (pathWidget) {
		const originalPathCallback = pathWidget.callback;
		pathWidget.callback = async (...args) => {
			if (typeof originalPathCallback === "function") {
				await originalPathCallback.apply(pathWidget, args);
			}
			const state = watchedNodes.get(String(node.id));
			if (Boolean(state?.initialized)) {
				try {
					await syncNodeFolderState(node, { silent: true });
				} catch (error) {
					console.warn("[Shaobkj] 循环触发路径更新失败:", error);
				}
			}
		};
	}

	const originalOnRemoved = node.onRemoved;
	node.onRemoved = function (...args) {
		watchedNodes.delete(String(node.id));
		if (typeof originalOnRemoved === "function") {
			return originalOnRemoved.apply(this, args);
		}
		return undefined;
	};

	watchedNodes.set(String(node.id), {
		path: resolveFolderPath(node),
		signature: "",
		total: Number(findWidget(node, "总数")?.value ?? 0),
		initialized: false,
	});

	node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
	name: "Shaobkj.LoopTrigger",
	async setup() {
		startPolling();
	},
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_CLASS_NAME) {
			return;
		}

		const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
			enhanceLoopTriggerNode(this);
			return result;
		};
	},
});
