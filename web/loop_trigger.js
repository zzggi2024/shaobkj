import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS_NAME = "Shaobkj_Loop_Trigger";
const watchedNodes = new Map();
const stoppedNodes = new Set();
let pollTimer = null;
let queueListenerBound = false;
let feedbackListenerBound = false;
let loopQueueRunning = false;

function sleep(ms) {
	return new Promise((resolve) => window.setTimeout(resolve, ms));
}

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

function looksLikePath(value) {
	const text = String(value ?? "").trim();
	return Boolean(text) && (/[a-zA-Z]:[\\/]/.test(text) || text.includes("\\") || text.includes("/"));
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
		if (!originNode) {
			return "";
		}
		const values = [];
		if (Array.isArray(originNode.widgets)) {
			for (const widget of originNode.widgets) {
				collectStringValues(widget?.value, values);
			}
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
	if (hasLinkedInput(node, "文件夹路径")) {
		const linkedValue = getLinkedTextValue(node, "文件夹路径");
		if (linkedValue) {
			return linkedValue;
		}
	}
	const pathWidget = findWidget(node, "文件夹路径");
	const widgetValue = String(pathWidget?.value ?? "").trim();
	if (widgetValue) {
		return widgetValue;
	}
	return getLinkedTextValue(node, "文件夹路径");
}

function getIncludeSubdirsValue(node) {
	const widget = findWidget(node, "包含子文件夹");
	return widget ? Boolean(widget.value) : true;
}

function getCurrentIndexValue(node) {
	const widget = findWidget(node, "当前执行编号");
	return Number(widget?.value ?? 0);
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

function showMessage(message, duration = 1800) {
	if (app?.ui?.dialog?.show) {
		app.ui.dialog.show(message);
		setTimeout(() => app.ui.dialog.close(), duration);
		return;
	}
	alert(message);
}

async function stopLoopTasks(node) {
	const nodeId = String(node?.id ?? "");
	if (nodeId) {
		stoppedNodes.add(nodeId);
	}
	setWidgetValue(node, "mode", false);
	setWidgetValue(node, "当前执行编号", 0);
	try {
		await api.fetchApi("/queue", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ clear: true }),
		});
	} catch (error) {
		console.warn("[Shaobkj] 停止未执行任务失败:", error);
	}
	showMessage("已停止未执行的循环任务");
}

function resetLoopStoppedState(node) {
	const nodeId = String(node?.id ?? "");
	if (nodeId) {
		stoppedNodes.delete(nodeId);
	}
}

async function initializeFolder(node, path, includeSubdirs = true) {
	const response = await api.fetchApi("/shaobkj/loop_trigger/initialize", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			path,
			include_subdirs: includeSubdirs,
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

	const includeSubdirs = getIncludeSubdirsValue(node);
	if (fromInit) {
		pathWidget.value = folderPath;
	}
	const result = fromInit ? await initializeFolder(node, folderPath, includeSubdirs) : await scanFolder(folderPath, includeSubdirs);
	if (!result?.exists || !result?.is_dir) {
		if (fromInit && !silent) {
			throw new Error("文件夹不存在或路径无效");
		}
		setWidgetValue(node, "总数", 0);
		setWidgetValue(node, "强制循环数", 0);
		setWidgetValue(node, "当前执行编号", 0);
		watchedNodes.set(String(node.id), {
			path: folderPath,
			signature: "",
			total: 0,
			initialized: false,
			includeSubdirs,
		});
		return;
	}

	const total = Number(result.total ?? 0);
	const previousTotal = Number(totalWidget.value ?? 0);
	const totalChanged = previousTotal !== total;
	if (totalChanged) {
		totalWidget.value = total;
	}
	if (Number(forceWidget.value ?? 0) !== total) {
		forceWidget.value = total;
	}

	if (fromInit) {
		setWidgetValue(node, "当前执行编号", 0);
	}

	node.setDirtyCanvas?.(true, true);
	watchedNodes.set(String(node.id), {
		path: folderPath,
		signature: String(result.signature ?? ""),
		total,
		initialized: true,
		includeSubdirs,
	});

	if (!silent && fromInit) {
		console.log("[Shaobkj] 循环触发初始化诊断:", result);
		showMessage(`成功加载 ${total} 张图片`, 3000);
	}
}

async function pollLoopTriggerNodes() {
	const nodesById = app?.graph?._nodes_by_id ?? {};
	for (const [nodeId] of Array.from(watchedNodes.entries())) {
		const node = nodesById[nodeId];
		if (!node) {
			watchedNodes.delete(nodeId);
			continue;
		}

		const folderPath = resolveFolderPath(node);
		if (!folderPath) {
			continue;
		}

		if (getCurrentIndexValue(node) > 0) {
			continue;
		}
		try {
			await syncNodeFolderState(node, { silent: true });
		} catch (error) {
			console.warn("[Shaobkj] 循环触发自动初始化失败:", error);
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

async function queueLoopPrompt(nodeId) {
	if (nodeId && stoppedNodes.has(nodeId)) {
		return;
	}
	if (loopQueueRunning) {
		return;
	}
	loopQueueRunning = true;
	try {
		for (let attempt = 1; attempt <= 3; attempt++) {
			if (nodeId && stoppedNodes.has(nodeId)) {
				return;
			}
			try {
				await sleep(800);
				if (nodeId && stoppedNodes.has(nodeId)) {
					return;
				}
				await app.queuePrompt(0);
				return;
			} catch (error) {
				console.warn(`[Shaobkj] 循环触发续队列失败，第 ${attempt} 次:`, error);
				if (attempt < 3) {
					await sleep(1000);
				}
			}
		}
	} finally {
		loopQueueRunning = false;
	}
}

function bindLoopTriggerEvents() {
	if (!feedbackListenerBound) {
		api.addEventListener("shaobkj.loop_trigger.feedback", (evt) => {
			const detail = evt?.detail ?? null;
			const nodeId = detail?.node_id ? String(detail.node_id) : "";
			const widgetName = detail?.widget_name ? String(detail.widget_name) : "";
			const value = detail?.value;
			const node = app?.graph?._nodes_by_id?.[nodeId];
			if (!node) {
				return;
			}
			setWidgetValue(node, widgetName, value);
		});
		feedbackListenerBound = true;
	}

	if (!queueListenerBound) {
		api.addEventListener("shaobkj.loop_trigger.add_queue", async (evt) => {
			const detail = evt?.detail ?? null;
			const nodeId = detail?.node_id ? String(detail.node_id) : "";
			await queueLoopPrompt(nodeId);
		});
		queueListenerBound = true;
	}
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
	const internalStateWidget = findWidget(node, "当前执行编号");
	if (internalStateWidget) {
		internalStateWidget.options = internalStateWidget.options || {};
		internalStateWidget.options.readOnly = true;
		setWidgetHiddenState(internalStateWidget, false);
	}

	const existingInitButton = findWidget(node, "初始化");
	if (!existingInitButton) {
		const initButton = node.addWidget("button", "初始化", "Init", () => {
			resetLoopStoppedState(node);
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
			resetLoopStoppedState(node);
			syncNodeFolderState(node, { fromInit: true }).catch((error) => {
				showMessage(String(error?.message || error));
			});
		};
	}

	const existingStopButton = findWidget(node, "停止任务");
	if (!existingStopButton) {
		const stopButton = node.addWidget("button", "停止任务", "Stop", () => {
			stopLoopTasks(node);
		});
		stopButton.name = "停止任务";
		stopButton.label = "■ 停止任务";
		stopButton.serialize = false;
		stopButton.tooltip = "停止当前循环触发，并清空未执行的队列任务";
	} else {
		existingStopButton.label = "■ 停止任务";
		existingStopButton.serialize = false;
		existingStopButton.callback = () => {
			stopLoopTasks(node);
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
		const nodeId = String(node.id);
		watchedNodes.delete(nodeId);
		stoppedNodes.delete(nodeId);
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
		includeSubdirs: getIncludeSubdirsValue(node),
	});

	node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
	name: "Shaobkj.LoopTrigger",
	async setup() {
		bindLoopTriggerEvents();
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
