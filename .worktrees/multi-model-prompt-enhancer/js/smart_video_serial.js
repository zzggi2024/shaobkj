import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS_NAME = "Shaobkj_SmartVideoSplit";
const PROGRESS_WIDGET_NAMES = ["当前段", "总段数"];

function findWidget(node, widgetName) {
	return node?.widgets?.find((item) => item?.name === widgetName);
}

function lockProgressWidgetValue(widget, value) {
	widget.options = widget.options || {};
	widget.options.readOnly = true;
	widget.options.min = value;
	widget.options.max = value;
	widget.value = value;
}

function setWidgetValue(nodeId, widgetName, value) {
	const node = app?.graph?._nodes_by_id?.[String(nodeId)];
	const widget = findWidget(node, widgetName);
	if (!widget) {
		return;
	}
	if (PROGRESS_WIDGET_NAMES.includes(widgetName)) {
		lockProgressWidgetValue(widget, value);
	} else {
		widget.value = value;
	}
	widget.callback?.(value);
	node.setDirtyCanvas?.(true, true);
}

function configureProgressWidgets(node) {
	for (const widgetName of PROGRESS_WIDGET_NAMES) {
		const widget = findWidget(node, widgetName);
		if (!widget) {
			continue;
		}
		lockProgressWidgetValue(widget, Number(widget.value ?? 0));
	}
}

function resetProgressWidgets(node) {
	for (const widgetName of PROGRESS_WIDGET_NAMES) {
		const widget = findWidget(node, widgetName);
		if (!widget) {
			continue;
		}
		lockProgressWidgetValue(widget, 0);
		widget.callback?.(0);
	}
	node?.setDirtyCanvas?.(true, true);
}

function resetAllProgressWidgets() {
	for (const node of app?.graph?._nodes || []) {
		if (node?.comfyClass === NODE_CLASS_NAME || node?.type === NODE_CLASS_NAME) {
			resetProgressWidgets(node);
		}
	}
}

app.registerExtension({
	name: "shaobkj.smart_video_serial",
	setup() {
		api.addEventListener("shaobkj.smart_video.feedback", (event) => {
			const detail = event?.detail;
			setWidgetValue(detail?.node_id, detail?.widget_name, detail?.value);
		});
		api.addEventListener("promptQueueing", resetAllProgressWidgets);
	},
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_CLASS_NAME) {
			return;
		}
		const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = originalOnNodeCreated
				? originalOnNodeCreated.apply(this, arguments)
				: undefined;
			configureProgressWidgets(this);
			return result;
		};
	},
});
