import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const PARAM_EXTRACT = "Shaobkj_ParamExtract";
const LAST_TYPE = Symbol("LastType");

app.registerExtension({
    name: "shaobkj.ParamExtract",
    init() {
        const graphConfigure = LGraph.prototype.configure;
        LGraph.prototype.configure = function () {
            const r = graphConfigure.apply(this, arguments);
            for (const n of app.graph._nodes) {
                if (n.type === PARAM_EXTRACT) {
                    n.onGraphConfigured();
                }
            }
            return r;
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        function addOutputHandler() {
            nodeType.prototype.getFirstReroutedOutput = function (slot) {
                const links = this.outputs[slot].links;
                if (!links) return null;

                const search = [];
                for (const l of links) {
                    const link = app.graph.links[l];
                    if (!link) continue;

                    const node = app.graph.getNodeById(link.target_id);
                    if (node.type !== PARAM_EXTRACT) {
                        return { node, link };
                    }
                    search.push({ node, link });
                }

                for (const { link, node } of search) {
                    const r = node.getFirstReroutedOutput(link.target_slot);
                    if (r) {
                        return r;
                    }
                }
            };
        }

        if (nodeData.name !== PARAM_EXTRACT) {
            return;
        }

        const configure = nodeType.prototype.configure || LGraphNode.prototype.configure;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        const onAdded = nodeType.prototype.onAdded;

        nodeType.title_mode = LiteGraph.NO_TITLE;

        function hasAnyInput(node) {
            for (const input of node.inputs) {
                if (input.link) {
                    return true;
                }
            }
            return false;
        }

        nodeType.prototype.onAdded = function () {
            onAdded?.apply(this, arguments);
            this.inputs[0].label = "";
            this.outputs[0].label = "value";
            this.setSize(this.computeSize());
        };

        nodeType.prototype.onGraphConfigured = function () {
            if (hasAnyInput(this)) return;

            const outputNode = this.getFirstReroutedOutput(0);
            if (outputNode) {
                this.checkPrimitiveWidget(outputNode);
            }
        };

        nodeType.prototype.checkPrimitiveWidget = function ({ node, link }) {
            let widgetType = link.type;
            let targetLabel = widgetType;
            const input = node.inputs[link.target_slot];
            const sourceWidget = input.widget;
            const realWidget = sourceWidget?.name && Array.isArray(node.widgets)
                ? node.widgets.find((w) => w?.name === sourceWidget.name)
                : null;
            if (sourceWidget?.config?.[0] instanceof Array) {
                targetLabel = sourceWidget.name;
                widgetType = "COMBO";
            }

            if (widgetType in ComfyWidgets) {
                const currentWidget = this.widgets?.[0];
                const shouldRebuild = !currentWidget || this[LAST_TYPE] !== widgetType || currentWidget.value === undefined;
                if (shouldRebuild && this.widgets) {
                    this.widgets.length = 0;
                }
                if (!this.widgets?.length) {
                    let v;
                    if (this.widgets_values?.length && this.widgets_values[0] !== undefined) {
                        v = this.widgets_values[0];
                    }
                    let config = [link.type, {}];
                    if (sourceWidget?.config) {
                        config = sourceWidget.config;
                    }
                    const { widget } = ComfyWidgets[widgetType](this, "value", config, app);
                    if (widgetType === "COMBO") {
                        const values = realWidget?.options?.values ?? sourceWidget?.options?.values ?? sourceWidget?.config?.[0] ?? [];
                        if (widget.options) {
                            widget.options.values = Array.isArray(values) ? [...values] : [];
                        }
                        const defaultValue = realWidget?.value ?? v ?? sourceWidget?.value ?? values?.[0];
                        if (defaultValue !== undefined) {
                            widget.value = defaultValue;
                        }
                    } else if (v !== undefined && (!this[LAST_TYPE] || this[LAST_TYPE] === widgetType)) {
                        widget.value = v;
                    }
                    this[LAST_TYPE] = widgetType;
                }
            } else if (this.widgets) {
                this.widgets.length = 0;
            }

            return targetLabel;
        };

        nodeType.prototype.getReroutedInputs = function (slot) {
            let nodes = [{ node: this }];
            let node = this;
            while (node?.type === PARAM_EXTRACT) {
                const input = node.inputs[slot];
                if (input.link) {
                    const link = app.graph.links[input.link];
                    node = app.graph.getNodeById(link.origin_id);
                    slot = link.origin_slot;
                    nodes.push({ node, link });
                } else {
                    node = null;
                }
            }
            return nodes;
        };

        addOutputHandler();

        nodeType.prototype.changeRerouteType = function (slot, type, label) {
            const color = LGraphCanvas.link_type_colors[type];
            const output = this.outputs[slot];
            this.inputs[slot].label = " ";
            output.label = label || (type === "*" ? "value" : type);
            output.type = type;

            for (const linkId of output.links || []) {
                const link = app.graph.links[linkId];
                if (!link) continue;
                link.color = color;
                const node = app.graph.getNodeById(link.target_id);
                if (node.changeRerouteType) {
                    node.changeRerouteType(link.target_slot, type, label);
                } else {
                    const theirType = node.inputs[link.target_slot].type;
                    if (theirType !== type && theirType !== "*") {
                        node.disconnectInput(link.target_slot);
                    }
                }
            }

            if (this.inputs[slot].link) {
                const link = app.graph.links[this.inputs[slot].link];
                if (link) link.color = color;
            }
        };

        let configuring = false;
        nodeType.prototype.configure = function () {
            configuring = true;
            const r = configure?.apply(this, arguments);
            configuring = false;
            return r;
        };

        Object.defineProperty(nodeType, "title_mode", {
            get() {
                return app.canvas.current_node?.widgets?.length ? LiteGraph.NORMAL_TITLE : LiteGraph.NO_TITLE;
            },
        });

        nodeType.prototype.onConnectionsChange = function (type, _, _connected, link_info) {
            if (configuring) return;

            const isInput = type === LiteGraph.INPUT;
            const slot = isInput ? link_info.target_slot : link_info.origin_slot;

            let targetLabel = null;
            let targetNode = null;
            let targetType = "*";
            let targetSlot = slot;

            const inputPath = this.getReroutedInputs(slot);
            const rootInput = inputPath[inputPath.length - 1];
            const outputNode = this.getFirstReroutedOutput(slot);
            if (rootInput.node.type === PARAM_EXTRACT) {
                if (outputNode) {
                    targetType = outputNode.link.type;
                } else if (rootInput.node.widgets) {
                    rootInput.node.widgets.length = 0;
                }
                targetNode = rootInput;
                targetSlot = rootInput.link?.target_slot ?? slot;
            } else {
                targetNode = inputPath[inputPath.length - 2];
                targetType = rootInput.node.outputs[rootInput.link.origin_slot].type;
                targetSlot = rootInput.link.target_slot;
            }

            if (this.widgets && inputPath.length > 1) {
                this.widgets.length = 0;
            }

            if (outputNode && rootInput.node.checkPrimitiveWidget) {
                targetLabel = rootInput.node.checkPrimitiveWidget(outputNode);
            }

            targetNode.node.changeRerouteType(targetSlot, targetType, targetLabel);
            return onConnectionsChange?.apply(this, arguments);
        };

        const computeSize = nodeType.prototype.computeSize || LGraphNode.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const r = computeSize.apply(this, arguments);
            if (this.flags?.collapsed) {
                return [1, 25];
            } else if (this.widgets?.length) {
                return r;
            } else {
                let w = 75;
                if (this.outputs?.[0]?.label) {
                    const t = LiteGraph.NODE_TEXT_SIZE * this.outputs[0].label.length * 0.6 + 30;
                    if (t > w) {
                        w = t;
                    }
                }
                return [w, r[1]];
            }
        };

        const collapse = nodeType.prototype.collapse || LGraphNode.prototype.collapse;
        nodeType.prototype.collapse = function () {
            collapse.apply(this, arguments);
            this.setSize(this.computeSize());
            requestAnimationFrame(() => {
                this.setDirtyCanvas(true, true);
            });
        };

        nodeType.prototype.onBounding = function (area) {
            if (this.flags?.collapsed) {
                area[1] -= 15;
            }
        };
    },
});
