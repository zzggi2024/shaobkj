import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_CLASS = "Shaobkj_MaskRecognition5";
const MARK_WIDGET = "标记数据";
const HIDDEN_WIDGET_NAMES = [
    // Keep the normal ComfyUI parameter widgets outside the annotation panel.
    // Only the two panel toggles and the serialized annotation payload are
    // internal controls.
    "复用标记", "标记特征", MARK_WIDGET,
];
// Native external controls retained: 指定文字、遮罩扩展、抽取帧数、随机种。

function previewUrl(nodeId, frame) {
    const params = new URLSearchParams({ node_id: String(nodeId), frame: String(frame) });
    return api.apiURL(`/shaobkj/mask_recognition5_preview?${params.toString()}`);
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget?.name === name);
}

function hideNativeWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
}

function hideAllNativeWidgets(node) {
    for (const name of HIDDEN_WIDGET_NAMES) hideNativeWidget(getWidget(node, name));
}

function createPanel(node) {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:4px 6px 8px;box-sizing:border-box;width:100%;background:#171717;color:#ddd;";
    root.addEventListener("pointerdown", (event) => event.stopPropagation());

    const panelHeader = document.createElement("div");
    panelHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;min-height:28px;padding:0 2px;";
    const reuseToggle = document.createElement("button");
    reuseToggle.type = "button";
    reuseToggle.style.cssText = "border:1px solid #666;border-radius:4px;background:#333;color:#eee;padding:4px 9px;cursor:pointer;";
    const featureToggle = document.createElement("button");
    featureToggle.type = "button";
    featureToggle.style.cssText = "border:1px solid #e5b94f;border-radius:4px;background:#b78619;color:#fff4c7;padding:4px 10px;cursor:pointer;font-weight:600;";
    panelHeader.append(reuseToggle, featureToggle);

    const content = document.createElement("div");
    content.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex;gap:5px;align-items:center;";
    const modeButton = document.createElement("button");
    modeButton.textContent = "画笔";
    const rectButton = document.createElement("button");
    rectButton.textContent = "矩形选框";
    const undoButton = document.createElement("button");
    undoButton.textContent = "撤销";
    const changeButton = document.createElement("button");
    changeButton.textContent = "换一批";
    const doneButton = document.createElement("button");
    doneButton.textContent = "✓ 选好了";
    doneButton.style.background = "#347d4a";
    const resetButton = document.createElement("button");
    resetButton.textContent = "重新标记";
    resetButton.style.background = "#6b3f3f";
    [modeButton, rectButton, undoButton, changeButton, resetButton, doneButton].forEach((button) => {
        button.style.cssText += "border:1px solid #666;border-radius:4px;background:#333;color:#eee;padding:3px 8px;cursor:pointer;";
    });
    toolbar.append(modeButton, rectButton, undoButton, changeButton, resetButton, doneButton);

    const thumbs = document.createElement("div");
    thumbs.style.cssText = "display:flex;gap:5px;overflow-x:auto;min-height:58px;";
    const stage = document.createElement("div");
    stage.style.cssText = "display:flex;align-items:center;justify-content:center;width:100%;min-height:80px;max-height:420px;overflow:hidden;background:#0b0b0b;border:1px solid #444;border-radius:4px;box-sizing:border-box;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;max-width:100%;max-height:100%;margin:0 auto;cursor:crosshair;";
    const status = document.createElement("div");
    status.style.cssText = "font-size:11px;color:#aaa;white-space:pre-wrap;";
    const helper = document.createElement("div");
    helper.textContent = "代表帧 · 点击缩略图切换，标记可保存在本帧";
    helper.style.cssText = "font-size:10px;color:#999;padding:0 2px;";
    const frameFooter = document.createElement("div");
    frameFooter.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:26px;";
    const confirmFrameButton = document.createElement("button");
    confirmFrameButton.textContent = "确定";
    confirmFrameButton.style.cssText = "border:1px solid #5d9b68;border-radius:4px;background:#2f6840;color:#fff;padding:3px 14px;cursor:pointer;";
    const markedCount = document.createElement("span");
    markedCount.style.cssText = "font-size:11px;color:#a9d7b1;";
    stage.appendChild(canvas);
    frameFooter.append(markedCount, confirmFrameButton);
    const annotationSection = document.createElement("div");
    annotationSection.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
    annotationSection.append(helper, toolbar, thumbs, stage, frameFooter, status);
    content.append(annotationSection);
    root.append(panelHeader, content);

    const ctx = canvas.getContext("2d");
    const state = {
        nodeId: "",
        previewToken: "",
        frameCount: 0,
        candidates: [],
        batchOffset: 0,
        batchSize: 5,
        currentFrame: 1,
        image: null,
        drawing: false,
        mode: "paint",
        tool: "brush",
        currentStroke: null,
        rectStart: null,
        marks: new Map(),
        confirmedFrames: new Set(),
        // Before submission this is the node's unmarked representative batch;
        // after “选好了” it becomes the confirmed marked-frame batch.
        showMarkedOnly: false,
        enabled: true,
        userResized: false,
        programmaticResize: false,
    };

    function syncCustomControls() {
        // The regular ComfyUI widgets are intentionally rendered outside this
        // panel, so there is no duplicate value synchronisation here.
    }

    function currentStrokes() {
        return state.marks.get(state.currentFrame) || [];
    }

    function updateMarkedCount() {
        markedCount.textContent = `已标记 ${state.confirmedFrames.size} 张`;
        confirmFrameButton.disabled = currentStrokes().length === 0;
        confirmFrameButton.textContent = state.confirmedFrames.has(state.currentFrame)
            ? "已确定"
            : "确定";
        confirmFrameButton.style.opacity = confirmFrameButton.disabled ? "0.5" : "1";
        doneButton.disabled = state.confirmedFrames.size === 0;
        doneButton.style.opacity = doneButton.disabled ? "0.5" : "1";
    }

    function serializedFrames() {
        return [...state.marks.entries()]
            .filter(([frame, strokes]) => strokes.length && state.confirmedFrames.has(frame))
            .map(([frame, strokes]) => ({ frame, strokes }));
    }

    function persistMarks(submitted = false) {
        const reuse = Boolean(getWidget(node, "复用标记")?.value);
        if (!reuse) return;
        const frames = serializedFrames();
        const payload = {
            submitted: Boolean(submitted && frames.length),
            token: submitted ? state.previewToken : "",
            frames,
        };
        const encoded = JSON.stringify(payload);
        node.properties = node.properties || {};
        node.properties.maskRecognitionMarksDraft = encoded;
        node.properties.maskRecognitionMarks = encoded;
        const marksWidget = getWidget(node, MARK_WIDGET);
        if (marksWidget) marksWidget.value = encoded;
        node.setDirtyCanvas?.(true, true);
    }

    function restorePersistedMarks() {
        const reuse = Boolean(getWidget(node, "复用标记")?.value);
        if (!reuse) return;
        const marksWidget = getWidget(node, MARK_WIDGET);
        const encoded = node.properties?.maskRecognitionMarksDraft
            || node.properties?.maskRecognitionMarks
            || marksWidget?.value
            || "";
        if (!encoded) return;
        try {
            const payload = typeof encoded === "string" ? JSON.parse(encoded) : encoded;
            if (!payload || !Array.isArray(payload.frames)) return;
            state.marks.clear();
            state.confirmedFrames.clear();
            for (const item of payload.frames) {
                const frame = Number(item?.frame);
                const strokes = Array.isArray(item?.strokes) ? item.strokes : [];
                if (frame >= 1 && strokes.length) {
                    state.marks.set(frame, strokes);
                    state.confirmedFrames.add(frame);
                }
            }
            if (payload.token) state.previewToken = String(payload.token);
            state.showMarkedOnly = Boolean(payload.submitted && state.confirmedFrames.size);
            updateMarkedCount();
            draw();
            renderThumbs();
        } catch (error) {
            console.warn("[shaobkj] 无法恢复遮罩识别标记", error);
        }
    }

    function updateFeatureToggle() {
        featureToggle.textContent = state.enabled ? "启动标记特征：开启" : "启动标记特征：关闭";
        featureToggle.style.background = state.enabled ? "#b78619" : "#5f512e";
        const reuse = getWidget(node, "复用标记");
        reuseToggle.textContent = reuse?.value ? "复用标记：开启" : "复用标记：关闭";
        reuseToggle.style.background = reuse?.value ? "#3b6d4a" : "#333";
    }

    function draw() {
        if (!state.image) return;
        canvas.width = state.image.naturalWidth || 1;
        canvas.height = state.image.naturalHeight || 1;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "rgba(0,255,90,.9)";
        ctx.lineWidth = Math.max(3, Math.min(canvas.width, canvas.height) * 0.012);
        for (const stroke of currentStrokes()) {
            if (!stroke.points.length) continue;
            if ((stroke.mode === "rect" || stroke.mode === "lasso") && stroke.points.length >= 3) {
                const first = stroke.points[0];
                const last = stroke.points.reduce((point, current) => [
                    Math.max(point[0], current[0]),
                    Math.max(point[1], current[1]),
                ], stroke.points[0]);
                const left = stroke.points.reduce((value, point) => Math.min(value, point[0]), 1);
                const top = stroke.points.reduce((value, point) => Math.min(value, point[1]), 1);
                ctx.save();
                if (stroke.mode === "lasso") {
                    ctx.beginPath();
                    stroke.points.forEach((point, index) => {
                        const x = point[0] * canvas.width;
                        const y = point[1] * canvas.height;
                        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    });
                    ctx.closePath();
                    ctx.fillStyle = "rgba(0,255,90,.22)";
                    ctx.fill();
                }
                ctx.setLineDash([8, 5]);
                if (stroke.mode === "rect") {
                    ctx.strokeRect(left * canvas.width, top * canvas.height,
                        (last[0] - left) * canvas.width, (last[1] - top) * canvas.height);
                } else {
                    ctx.beginPath();
                    stroke.points.forEach((point, index) => {
                        const x = point[0] * canvas.width;
                        const y = point[1] * canvas.height;
                        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    });
                    ctx.closePath();
                    ctx.stroke();
                }
                ctx.restore();
                continue;
            }
            ctx.beginPath();
            stroke.points.forEach((point, index) => {
                const x = point[0] * canvas.width;
                const y = point[1] * canvas.height;
                if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
    }

    function loadFrame(frame) {
        state.currentFrame = Number(frame) || 1;
        const image = new Image();
        image.onload = () => {
            state.image = image;
            syncCanvasLayout();
            draw();
            syncNodeSize();
        };
        image.src = previewUrl(state.nodeId, state.currentFrame);
        status.textContent = `当前帧 F${String(state.currentFrame).padStart(6, "0")} · 可在画布上涂抹需要去除的字幕/水印`;
        renderThumbs();
        updateMarkedCount();
    }

    function syncCanvasLayout() {
        if (!state.image) return;
        const maxWidth = Math.max(1, content.clientWidth - 12);
        const maxHeight = 412;
        const naturalWidth = Math.max(1, state.image.naturalWidth);
        const naturalHeight = Math.max(1, state.image.naturalHeight);
        const scale = Math.min(
            1,
            maxWidth / naturalWidth,
            maxHeight / naturalHeight,
        );
        const displayWidth = Math.max(1, Math.round(naturalWidth * scale));
        const displayHeight = Math.max(1, Math.round(naturalHeight * scale));
        stage.style.aspectRatio = `${naturalWidth} / ${naturalHeight}`;
        stage.style.height = `${displayHeight + 8}px`;
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;
    }

    function syncNodeSize(force = false) {
        if (typeof node.computeSize !== "function" || typeof node.setSize !== "function") return;
        if (state.userResized && !force) return;
        const computed = node.computeSize();
        if (!Array.isArray(computed) || computed.length < 2) return;
        const width = Math.max(420, Number(computed[0]) || 420);
        const height = Math.max(180, Number(computed[1]) || 180);
        const current = node.size || [];
        if (Math.abs(Number(current[0]) - width) > 1 || Math.abs(Number(current[1]) - height) > 1) {
            state.programmaticResize = true;
            node.setSize([width, height]);
            state.programmaticResize = false;
        }
        node.setDirtyCanvas?.(true, true);
    }

    function renderThumbs() {
        thumbs.replaceChildren();
        const visibleFrames = state.showMarkedOnly
            ? [...state.confirmedFrames]
            : [...new Set(state.candidates)];
        const visible = visibleFrames.slice(state.batchOffset, state.batchOffset + state.batchSize);
        for (const frame of visible) {
            const button = document.createElement("button");
            button.title = `F${String(frame).padStart(6, "0")}`;
            button.style.cssText = `position:relative;flex:0 0 92px;height:54px;padding:0;overflow:hidden;background:#111;border:2px solid ${state.confirmedFrames.has(frame) ? "#55b86a" : frame === state.currentFrame ? "#f0a04b" : "#555"};border-radius:3px;cursor:pointer;`;
            const image = document.createElement("img");
            image.src = previewUrl(state.nodeId, frame);
            image.alt = `F${String(frame).padStart(6, "0")}`;
            image.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;";
            const overlay = document.createElement("canvas");
            overlay.width = 92;
            overlay.height = 54;
            overlay.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";
            const overlayContext = overlay.getContext("2d");
            const drawThumbMarks = () => {
                if (!overlayContext) return;
                overlayContext.clearRect(0, 0, overlay.width, overlay.height);
                overlayContext.strokeStyle = "rgba(0,255,90,.95)";
                overlayContext.fillStyle = "rgba(0,255,90,.2)";
                overlayContext.lineWidth = 2;
                for (const stroke of state.marks.get(frame) || []) {
                    if (!stroke.points?.length) continue;
                    overlayContext.beginPath();
                    stroke.points.forEach((point, index) => {
                        const x = point[0] * overlay.width;
                        const y = point[1] * overlay.height;
                        if (index === 0) overlayContext.moveTo(x, y); else overlayContext.lineTo(x, y);
                    });
                    if (stroke.mode === "rect" || stroke.mode === "lasso") {
                        overlayContext.closePath();
                        if (stroke.mode === "lasso") overlayContext.fill();
                    }
                    overlayContext.stroke();
                }
            };
            image.addEventListener("load", drawThumbMarks, { once: true });
            button.append(image, overlay);
            button.addEventListener("click", () => loadFrame(frame));
            thumbs.appendChild(button);
        }
    }

    function pointerPoint(event) {
        const rect = canvas.getBoundingClientRect();
        return [
            Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))),
            Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height))),
        ];
    }

    canvas.addEventListener("pointerdown", (event) => {
        if (!state.image) return;
        canvas.setPointerCapture(event.pointerId);
        state.drawing = true;
        const point = pointerPoint(event);
        state.rectStart = point;
        state.currentStroke = {
            mode: "lasso",
            radius: state.tool === "rect" ? 0.008 : 0.012,
            points: [point],
        };
        if (state.tool === "rect") state.currentStroke.mode = "rect";
        const strokes = currentStrokes();
        state.marks.set(state.currentFrame, [...strokes, state.currentStroke]);
        draw();
        updateMarkedCount();
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!state.drawing || !state.currentStroke) return;
        const point = pointerPoint(event);
        if (state.tool === "rect" && state.rectStart) {
            const [x0, y0] = state.rectStart;
            state.currentStroke.points = [
                [x0, y0], [point[0], y0], [point[0], point[1]],
                [x0, point[1]], [x0, y0],
            ];
        } else {
            state.currentStroke.points.push(point);
        }
        draw();
        updateMarkedCount();
    });
    const stopDrawing = () => {
        if (state.currentStroke?.mode === "lasso") {
            const points = state.currentStroke.points;
            if (points.length >= 3) {
                const first = points[0];
                const last = points[points.length - 1];
                if (Math.abs(first[0] - last[0]) > 1e-4 || Math.abs(first[1] - last[1]) > 1e-4) {
                    points.push([...first]);
                }
            }
        }
        state.drawing = false;
        state.currentStroke = null;
        state.rectStart = null;
    };
    canvas.addEventListener("pointerup", stopDrawing);
    canvas.addEventListener("pointercancel", stopDrawing);

    modeButton.addEventListener("click", () => {
        state.tool = "brush";
        state.mode = state.mode === "paint" ? "erase" : "paint";
        modeButton.textContent = state.mode === "paint" ? "画笔" : "擦除";
        rectButton.style.background = "#333";
    });
    rectButton.addEventListener("click", () => {
        state.tool = state.tool === "rect" ? "brush" : "rect";
        rectButton.style.background = state.tool === "rect" ? "#347d4a" : "#333";
        if (state.tool === "rect") modeButton.textContent = "画笔";
    });
    confirmFrameButton.addEventListener("click", () => {
        if (!currentStrokes().length) return;
        state.confirmedFrames.add(state.currentFrame);
        persistMarks(false);
        status.textContent = `当前帧 F${String(state.currentFrame).padStart(6, "0")} 已确认标记`;
        updateMarkedCount();
    });
    resetButton.addEventListener("click", () => {
        state.marks.clear();
        state.confirmedFrames.clear();
        state.showMarkedOnly = false;
        state.candidates = state.candidates.length ? [...state.candidates] : [];
        state.currentStroke = null;
        const marksWidget = getWidget(node, MARK_WIDGET);
        if (marksWidget) marksWidget.value = "";
        node.properties = node.properties || {};
        delete node.properties.maskRecognitionMarks;
        delete node.properties.maskRecognitionMarksDraft;
        status.textContent = "已清空标记，请重新圈选字幕或水印特征";
        draw();
        updateMarkedCount();
        renderThumbs();
        node.setDirtyCanvas?.(true, true);
    });
    undoButton.addEventListener("click", () => {
        const strokes = currentStrokes().slice(0, -1);
        state.marks.set(state.currentFrame, strokes);
        if (!strokes.length) state.confirmedFrames.delete(state.currentFrame);
        draw();
        updateMarkedCount();
    });
    changeButton.addEventListener("click", () => {
        const visibleFrames = state.showMarkedOnly
            ? [...state.confirmedFrames]
            : [...new Set(state.candidates)];
        if (!visibleFrames.length) return;
        state.batchOffset = (state.batchOffset + state.batchSize) % visibleFrames.length;
        loadFrame(visibleFrames[state.batchOffset]);
    });
    doneButton.addEventListener("click", async () => {
        const widget = getWidget(node, MARK_WIDGET);
        if (!widget) return;
        const frames = [...state.marks.entries()]
            .filter(([frame, strokes]) => strokes.length && state.confirmedFrames.has(frame))
            .map(([frame, strokes]) => ({ frame, strokes }));
        if (!frames.length) return;
        widget.value = JSON.stringify({ submitted: true, token: state.previewToken, frames });
        node.properties = node.properties || {};
        node.properties.maskRecognitionMarks = widget.value;
        node.properties.maskRecognitionMarksDraft = widget.value;
        state.showMarkedOnly = true;
        state.candidates = [...state.confirmedFrames];
        renderThumbs();
        node.setDirtyCanvas?.(true, true);
        status.textContent = `已提交 ${frames.length} 帧标记，正在自动开始第二阶段…`;
        await app.queuePrompt(0, 1);
    });

    const widget = node.addDOMWidget("示例帧标记", "shaobkj_mask_recognition5_annotation", root, { serialize: false });
    widget.serialize = false;
    const panelHeight = () => state.enabled
        ? Math.max(220, panelHeader.offsetHeight + annotationSection.offsetHeight + 18)
        : Math.max(84, panelHeader.offsetHeight + 18);
    const setEnabled = (enabled) => {
        state.enabled = Boolean(enabled);
        content.style.display = "flex";
        annotationSection.style.display = state.enabled ? "flex" : "none";
        updateFeatureToggle();
        widget.computeSize = (width) => [
            Math.max(Number(width) || 0, 420),
            panelHeight(),
        ];
        node.setDirtyCanvas?.(true, true);
    };
    featureToggle.addEventListener("click", () => {
        const widget = getWidget(node, "标记特征");
        const next = !state.enabled;
        if (widget) widget.value = next;
        setEnabled(next);
        syncNodeSize(true);
    });
    reuseToggle.addEventListener("click", () => {
        const widget = getWidget(node, "复用标记");
        if (!widget) return;
        widget.value = !Boolean(widget.value);
        updateFeatureToggle();
        node.setDirtyCanvas?.(true, true);
    });
    node.__shaobkjMaskRecognitionAnnotation = {
        state, loadFrame, renderThumbs, draw, setEnabled, syncCustomControls, updateFeatureToggle, syncNodeSize, persistMarks, restorePersistedMarks,
    };
    if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => {
            syncCanvasLayout();
        });
        observer.observe(root);
        node.__shaobkjMaskRecognitionAnnotation.resizeObserver = observer;
    }
    setEnabled(Boolean(getWidget(node, "标记特征")?.value ?? true));
    restorePersistedMarks();
    return widget;
}

app.registerExtension({
    name: "Shaobkj.MaskRecognition5Annotation",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated ? originalCreated.apply(this, arguments) : undefined;
            if (!this.__shaobkjMaskRecognitionAnnotation) createPanel(this);
            const controls = this.__shaobkjMaskRecognitionAnnotation;
            const originalResize = this.onResize;
            this.onResize = function () {
                if (!controls?.state.programmaticResize) controls.state.userResized = true;
                return originalResize ? originalResize.apply(this, arguments) : undefined;
            };
            const marksWidget = getWidget(this, MARK_WIDGET);
            if (marksWidget) {
                marksWidget.computeSize = () => [0, -4];
                marksWidget.serialize = true;
            }
            const featureWidget = getWidget(this, "标记特征");
            if (featureWidget) {
                featureWidget.hidden = true;
                featureWidget.computeSize = () => [0, -4];
            }
            hideAllNativeWidgets(this);
            if (typeof this.computeSize === "function" && typeof this.setSize === "function") {
                if (controls) controls.state.programmaticResize = true;
                this.setSize(this.computeSize());
                if (controls) controls.state.programmaticResize = false;
            }
            return result;
        };
        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured ? originalConfigured.apply(this, arguments) : undefined;
            hideAllNativeWidgets(this);
            const controls = this.__shaobkjMaskRecognitionAnnotation;
            if (controls) {
                // onConfigure runs before onExecuted after a ComfyUI restart.
                // Set the stable node id first so persisted thumbnails resolve
                // from disk instead of requesting node_id="" (404).
                controls.state.nodeId = String(this.id || "");
                controls.restorePersistedMarks();
                if (controls.state.confirmedFrames.size) {
                    controls.state.candidates = [...controls.state.confirmedFrames];
                    controls.loadFrame(controls.state.candidates[0]);
                }
            }
            return result;
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalExecuted ? originalExecuted.apply(this, arguments) : undefined;
            hideAllNativeWidgets(this);
            const preview = message?.mask_recognition5_preview?.[0];
            const controls = this.__shaobkjMaskRecognitionAnnotation;
            if (!controls) return result;
            controls.syncCustomControls();
            controls.updateFeatureToggle();
            const enabled = Boolean(getWidget(this, "标记特征")?.value ?? true);
            controls.setEnabled(enabled);
            if (!preview || !enabled) return result;
            const { state } = controls;
            state.nodeId = String(preview.node_id || this.id || "");
            state.previewToken = String(preview.token || "");
            state.frameCount = Math.max(1, Number(preview.frame_count) || 1);
            state.candidates = (preview.candidate_frames || preview.representative_frames || [1]).map(Number);
            state.batchSize = Math.max(1, Number(getWidget(this, "抽取帧数")?.value) || 5);
            state.batchOffset = 0;
            const reuse = Boolean(getWidget(this, "复用标记")?.value);
            if (preview.marked && !reuse) {
                state.marks.clear();
                state.confirmedFrames.clear();
                const marksWidget = getWidget(this, MARK_WIDGET);
                if (marksWidget) marksWidget.value = "";
                if (this.properties) delete this.properties.maskRecognitionMarks;
            }
            state.showMarkedOnly = Boolean(preview.marked && reuse && state.confirmedFrames.size);
            if (state.showMarkedOnly) state.candidates = [...state.confirmedFrames];
            const firstMarkedFrame = [...state.confirmedFrames][0];
            controls.loadFrame(firstMarkedFrame || state.candidates[0]);
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});
