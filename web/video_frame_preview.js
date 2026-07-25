import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "Shaobkj_VideoFrameNumber";
const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 360;
const PREVIEW_MIN_HEIGHT = 220;
const PREVIEW_NODE_CHROME_HEIGHT = 68;

function getPreviewWidgetHeight(node) {
    const nodeHeight = Number(node?.size?.[1]);
    const availableHeight = (Number.isFinite(nodeHeight) ? nodeHeight : MIN_NODE_HEIGHT)
        - PREVIEW_NODE_CHROME_HEIGHT;
    return Math.max(PREVIEW_MIN_HEIGHT, availableHeight);
}

function buildPreviewUrl(state, frameNumber) {
    const params = new URLSearchParams({
        node_id: state.nodeId,
        frame: String(frameNumber),
        version: String(state.version),
    });
    return api.apiURL(`/shaobkj/video_frame_preview?${params.toString()}`);
}

function createPreviewWidget(node) {
    const root = document.createElement("div");
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "8px";
    root.style.width = "100%";
    root.style.padding = "4px 8px 8px";
    root.style.boxSizing = "border-box";
    root.addEventListener("pointerdown", (event) => event.stopPropagation());

    const imageStage = document.createElement("div");
    imageStage.style.display = "flex";
    imageStage.style.alignItems = "center";
    imageStage.style.justifyContent = "center";
    imageStage.style.flex = "1";
    imageStage.style.minHeight = `${PREVIEW_MIN_HEIGHT}px`;
    imageStage.style.overflow = "hidden";
    imageStage.style.background = "#111";
    imageStage.style.border = "1px solid #4a4a4a";
    imageStage.style.borderRadius = "4px";

    const image = document.createElement("img");
    image.alt = "";
    image.draggable = false;
    image.style.display = "block";
    image.style.width = "100%";
    image.style.height = "100%";
    image.style.objectFit = "contain";
    image.style.userSelect = "none";
    imageStage.appendChild(image);

    const footer = document.createElement("div");
    footer.style.display = "grid";
    footer.style.gridTemplateColumns = "1fr auto";
    footer.style.alignItems = "center";
    footer.style.gap = "10px";

    const input = document.createElement("input");
    input.type = "range";
    input.min = "1";
    input.max = "1";
    input.step = "1";
    input.value = "1";
    input.disabled = true;
    input.style.width = "100%";
    input.style.accentColor = "#55a7ff";
    input.title = "选择预览帧";

    const counter = document.createElement("span");
    counter.textContent = "F000001 / 1";
    counter.style.minWidth = "108px";
    counter.style.color = "#ddd";
    counter.style.fontSize = "12px";
    counter.style.textAlign = "right";
    counter.style.whiteSpace = "nowrap";

    footer.append(input, counter);
    root.append(imageStage, footer);

    const state = {
        nodeId: "",
        frameCount: 1,
        version: 0,
        pendingFrame: 1,
        animationFrame: 0,
    };

    const renderFrame = (frameNumber) => {
        const clamped = Math.max(1, Math.min(state.frameCount, Number(frameNumber) || 1));
        input.value = String(clamped);
        counter.textContent = `F${String(clamped).padStart(6, "0")} / ${state.frameCount}`;
        if (state.nodeId) {
            image.src = buildPreviewUrl(state, clamped);
        }
    };

    input.addEventListener("input", () => {
        state.pendingFrame = Number(input.value);
        if (state.animationFrame) {
            return;
        }
        state.animationFrame = requestAnimationFrame(() => {
            state.animationFrame = 0;
            renderFrame(state.pendingFrame);
        });
    });

    const widget = node.addDOMWidget(
        "视频帧预览",
        "shaobkj_video_frame_preview",
        root,
        { serialize: false }
    );
    widget.serialize = false;
    widget.computeSize = (width) => [
        Math.max(Number(width) || 0, MIN_NODE_WIDTH),
        getPreviewWidgetHeight(node),
    ];
    const syncSize = () => {
        root.style.height = `${getPreviewWidgetHeight(node)}px`;
        node.setDirtyCanvas?.(true, true);
    };
    const originalOnResize = node.onResize;
    node.onResize = function () {
        const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
        syncSize();
        return result;
    };
    node.__shaobkjVideoFramePreview = {
        state,
        input,
        image,
        counter,
        renderFrame,
        syncSize,
    };
    syncSize();
    return widget;
}

app.registerExtension({
    name: "Shaobkj.VideoFramePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!this.__shaobkjVideoFramePreview) {
                createPreviewWidget(this);
            }
            this.setSize([
                Math.max(Number(this.size?.[0]) || 0, MIN_NODE_WIDTH),
                Math.max(Number(this.size?.[1]) || 0, MIN_NODE_HEIGHT),
            ]);
            this.__shaobkjVideoFramePreview?.syncSize();
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const raw = message?.frame_preview;
            const preview = Array.isArray(raw) ? raw[0] : raw;
            const controls = this.__shaobkjVideoFramePreview;
            if (!preview || !controls) {
                return result;
            }

            controls.state.nodeId = String(preview.node_id ?? this.id ?? "");
            controls.state.frameCount = Math.max(1, Number(preview.frame_count) || 1);
            controls.state.version = Number(preview.version) || 0;
            controls.input.min = "1";
            controls.input.max = String(controls.state.frameCount);
            controls.input.disabled = controls.state.frameCount <= 1;
            controls.renderFrame(1);
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});
