import { app } from "../../scripts/app.js";

const QUICKMARK_BASE64_TOKEN_PREFIX = "__SHAOBKJ_QUICKMARK_BASE64__:";

function getQuickMarkBase64Store() {
    if (!window.__shaobkj_quickmark_base64_store || typeof window.__shaobkj_quickmark_base64_store !== "object") {
        window.__shaobkj_quickmark_base64_store = {};
    }
    return window.__shaobkj_quickmark_base64_store;
}

function installQuickMarkPromptPatch() {
    if (window.__shaobkj_quickmark_prompt_patch_installed) return;
    window.__shaobkj_quickmark_prompt_patch_installed = true;

    const originalFetch = window.fetch?.bind(window);
    if (typeof originalFetch !== "function") return;

    window.fetch = async function (input, init) {
        try {
            const url = typeof input === "string" ? input : input?.url;
            const body = init?.body;
            if (typeof url === "string" && url.includes("/api/prompt") && typeof body === "string" && body.includes(QUICKMARK_BASE64_TOKEN_PREFIX)) {
                const store = getQuickMarkBase64Store();
                const payload = JSON.parse(body);
                const visit = value => {
                    if (typeof value === "string") {
                        if (value.startsWith(QUICKMARK_BASE64_TOKEN_PREFIX)) {
                            const nodeId = value.slice(QUICKMARK_BASE64_TOKEN_PREFIX.length);
                            return typeof store[nodeId] === "string" ? store[nodeId] : "";
                        }
                        return value;
                    }
                    if (!value || typeof value !== "object") return value;
                    if (Array.isArray(value)) return value.map(visit);
                    for (const key of Object.keys(value)) {
                        value[key] = visit(value[key]);
                    }
                    return value;
                };
                visit(payload);
                init = { ...(init || {}), body: JSON.stringify(payload) };
            }
        } catch (_error) {
        }
        return originalFetch(input, init);
    };
}

const DEFAULT_LAYOUT = {
    shiftLeft: 80,
    shiftRight: 80,
    panelHeight: 0
};

const WIDGET_NAMES = {
    BRUSH_DATA: "brush_data",
    BRUSH_SIZE: "brush_size",
    IMAGE_WIDTH: "image_width",
    IMAGE_HEIGHT: "image_height",
    IMAGE_BASE64: "image_base64"
};

class ShaobkjQuickMarkController {
    constructor(node) {
        this.node = node;
        this.state = createInitialState(node);
        initUIBindings(node, this.state);
        initInteractionBindings(node, this.state);
    }
}

function createInitialState(node) {
    if (!node.properties) {
        node.properties = {};
    }

    const defaults = {
        brushPaths: [],
        isDrawing: false,
        currentPath: [],
        brushSize: 4,
        brushOpacity: 1.0,
        brushMode: "brush",
        brushType: "free",
        brushColor: "255,0,0",
        eraserColor: "255,50,50",
        backgroundImage: null,
        imageWidth: 512,
        imageHeight: 512,
        colorPalette: null,
        markerPalette: null,
        activeMarker: null,
        imageBase64Data: "",
        brushTypeButtons: null,
        actionButtons: null
    };

    node.properties = {
        ...defaults,
        ...node.properties
    };

    delete node.properties.brushTypeButtons;
    delete node.properties.actionButtons;
    delete node.properties.colorButtonGroup;

    node.size = node.size || [500, 500];

    return {
        layout: { ...DEFAULT_LAYOUT },
        fontSize: LiteGraph?.NODE_SUBTEXT_SIZE ?? 10
    };
}

function initUIBindings(node, state) {
    const { shiftLeft, shiftRight, panelHeight } = state.layout;

    setupHiddenWidgets(node);
    installQuickMarkPromptPatch();

    const originalOnSerialize = node.onSerialize;
    node.onSerialize = function (o) {
        if (originalOnSerialize) {
            originalOnSerialize.apply(this, arguments);
        }
        try {
            if (o && Array.isArray(o.widgets_values) && Array.isArray(this.widgets)) {
                const base64Idx = this.widgets.findIndex(w => w?.name === WIDGET_NAMES.IMAGE_BASE64);
                if (base64Idx >= 0 && base64Idx < o.widgets_values.length) {
                    const current = this.widgets[base64Idx]?.value;
                    o.widgets_values[base64Idx] = typeof current === "string" ? current : "";
                }
            }
            if (o && o.properties && typeof o.properties === "object") {
                if ("imageBase64Data" in o.properties) {
                    o.properties.imageBase64Data = "";
                }
                delete o.properties.buttons;
                delete o.properties.inputs;
                delete o.properties.brushTypeButtons;
                delete o.properties.actionButtons;
                delete o.properties.colorButtonGroup;
            }
        } catch (_error) {
        }
    };

    const originalOnResize = node.onResize;
    node.onResize = function () {
        if (originalOnResize) {
            originalOnResize.apply(this, arguments);
        }
        if (this.min_size) {
            this.size[0] = Math.max(this.size[0], this.min_size[0] || 0);
            this.size[1] = Math.max(this.size[1], this.min_size[1] || 0);
        }
    };

    node.initButtons = function () {
        if (this.widgets[1] && this.widgets[1].value) {
            this.properties.brushSize = this.widgets[1].value || 10;
        }

        const buttonX = 12;
        let buttonY = 10;
        const buttonWidth = 56;
        const buttonHeight = 16;
        const buttonSpacing = 4;

        this._buttons = [
            {
                text: "加载",
                label: "加载图片",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                action: () => this.loadImageFromFile()
            }
        ];

        buttonY += buttonHeight + buttonSpacing + 10;
        this._brushTypeButtons = {
            free: {
                text: "画笔",
                label: "画笔",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: true,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "free";
                    this.properties.activeMarker = null;
                    this._brushTypeButtons.free.active = true;
                    this._brushTypeButtons.box.active = false;
                    this._brushTypeButtons.square.active = false;
                    this.updateThisNodeGraph?.();
                }
            },
            box: {
                text: "方框",
                label: "方框",
                x: buttonX,
                y: buttonY + buttonHeight + buttonSpacing,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: false,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "box";
                    this.properties.activeMarker = null;
                    this._brushTypeButtons.free.active = false;
                    this._brushTypeButtons.box.active = true;
                    this._brushTypeButtons.square.active = false;
                    this.updateThisNodeGraph?.();
                }
            },
            square: {
                text: "色块",
                label: "色块",
                x: buttonX,
                y: buttonY + (buttonHeight + buttonSpacing) * 2,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: false,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "square";
                    this.properties.activeMarker = null;
                    this._brushTypeButtons.free.active = false;
                    this._brushTypeButtons.box.active = false;
                    this._brushTypeButtons.square.active = true;
                    this.updateThisNodeGraph?.();
                }
            }
        };

        buttonY += (buttonHeight + buttonSpacing) * 3 + 10;
        this._actionButtons = [
            {
                text: "清除",
                label: "清除",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                action: () => {
                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];
                    this.updateThisNodeGraph?.();
                }
            },
            {
                text: "撤销",
                label: "撤销",
                x: buttonX,
                y: buttonY + buttonHeight + buttonSpacing,
                width: buttonWidth,
                height: buttonHeight,
                action: () => {
                    if (this.properties.brushPaths.length > 0) {
                        this.properties.brushPaths.pop();
                        this.updateThisNodeGraph?.();
                    }
                }
            }
        ];

        buttonY += (buttonHeight + buttonSpacing) * 2 + 10;
        const sliderHeight = 8;
        const labelHeight = 12;
        const sliderSpacing = 8;
        this.properties.sliders = [
            {
                label: "大小",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                labelHeight,
                sliderHeight,
                type: "size",
                min: 1,
                max: 100,
                value: this.properties.brushSize || 10,
                isDragging: false
            },
            {
                label: "透明度",
                x: buttonX,
                y: buttonY + labelHeight + sliderHeight + sliderSpacing,
                width: buttonWidth,
                labelHeight,
                sliderHeight,
                type: "opacity",
                min: 10,
                max: 100,
                value: Math.round((this.properties.brushOpacity || 1.0) * 100),
                isDragging: false
            }
        ];

        buttonY += (labelHeight + sliderHeight + sliderSpacing) * 2 + 10;
        const colorLabelHeight = 14;
        const swatchSize = 16;
        const swatchGap = 4;
        const swatchRows = 2;
        const swatchCols = 3;
        const swatchGridWidth = swatchCols * swatchSize + (swatchCols - 1) * swatchGap;
        this.properties.colorPalette = {
            label: "颜色",
            x: buttonX,
            y: buttonY,
            width: swatchGridWidth,
            labelHeight: colorLabelHeight,
            swatchSize,
            swatchGap,
            rows: swatchRows,
            cols: swatchCols,
            options: [
                { label: "Black", text: "黑", value: "0,0,0", textColor: "#ffffff" },
                { label: "White", text: "白", value: "255,255,255", textColor: "#000000" },
                { label: "Red", text: "红", value: "255,0,0", textColor: "#ffffff" },
                { label: "Green", text: "绿", value: "0,255,0", textColor: "#000000" },
                { label: "Blue", text: "蓝", value: "0,0,255", textColor: "#ffffff" },
                { label: "Gray", text: "灰", value: "128,128,128", textColor: "#000000" }
            ]
        };

        const markerGridY = buttonY + colorLabelHeight + swatchRows * swatchSize + (swatchRows - 1) * swatchGap + 6;
        const markerSize = swatchSize;
        const markerGap = swatchGap;
        const markerRows = 2;
        const markerCols = 3;
        const markerGridWidth = markerCols * markerSize + (markerCols - 1) * markerGap;
        this.properties.markerPalette = {
            label: "标记",
            x: buttonX,
            y: markerGridY,
            labelHeight: colorLabelHeight,
            markerSize,
            markerGap,
            rows: markerRows,
            cols: markerCols,
            width: markerGridWidth,
            options: [
                { text: "1", value: "1" },
                { text: "2", value: "2" },
                { text: "3", value: "3" },
                { text: "4", value: "4" },
                { text: "5", value: "5" },
                { text: "6", value: "6" }
            ]
        };

        const markerGridHeight = markerRows * markerSize + (markerRows - 1) * markerGap;
        const minHeight = Math.max(320, markerGridY + this.properties.markerPalette.labelHeight + markerGridHeight + 8);
        if (!this.min_size) {
            this.min_size = [300, minHeight];
        } else {
            this.min_size[1] = Math.max(this.min_size[1] || 0, minHeight);
        }
        if (this.size[1] < minHeight) {
            this.size[1] = minHeight;
        }
    };

    node.onAdded = function () {
        this.initButtons?.();
    };

    node.onConfigure = function () {
        const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
        const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);

        if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
            this.updateImageSize(widthWidget.value, heightWidget.value);
        } else {
            this.updateImageSize(512, 512);
        }

        const brushSizeWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
        this.properties.brushSize = brushSizeWidget && brushSizeWidget.value !== undefined
            ? brushSizeWidget.value
            : 10;

        if (this.properties.brushOpacity === undefined) {
            this.properties.brushOpacity = 1.0;
        }
        if (this.properties.brushColor === undefined) {
            this.properties.brushColor = "255,0,0";
        }
        if (this.properties.brushType === undefined) {
            this.properties.brushType = "free";
        }
        if (this.properties.eraserColor === undefined) {
            this.properties.eraserColor = "255,50,50";
        }

        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                if (slider.type === "size") {
                    slider.value = this.properties.brushSize;
                } else if (slider.type === "opacity") {
                    slider.value = Math.round((this.properties.brushOpacity || 1.0) * 100);
                }
            }
        }

        if (this.properties.colorPalette && this.properties.colorPalette.options) {
            const colorValue = this.properties.brushColor || "255,0,0";
            const match = this.properties.colorPalette.options.find(option => option.value === colorValue);
            if (!match) {
                this.properties.brushColor = "255,0,0";
            }
        }
        if (this.properties.markerPalette && this.properties.activeMarker) {
            const match = this.properties.markerPalette.options?.some(option => option.value === this.properties.activeMarker);
            if (!match) {
                this.properties.activeMarker = null;
            }
        }

        const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
        if (imageBase64Widget && imageBase64Widget.value) {
            let base64Value = imageBase64Widget.value;
            if (Array.isArray(base64Value)) {
                base64Value = base64Value.find(v => typeof v === "string" && v.trim() !== "") ?? base64Value[0];
            }
            base64Value = base64Value ?? "";
            if (typeof base64Value !== "string") {
                if (base64Value && typeof base64Value === "object") {
                    if (typeof base64Value.base64 === "string") base64Value = base64Value.base64;
                    else if (typeof base64Value.data === "string") base64Value = base64Value.data;
                    else if (typeof base64Value.image_base64 === "string") base64Value = base64Value.image_base64;
                }
            }
            if (typeof base64Value !== "string") base64Value = String(base64Value);
            if (base64Value.startsWith(QUICKMARK_BASE64_TOKEN_PREFIX)) {
                const nodeId = base64Value.slice(QUICKMARK_BASE64_TOKEN_PREFIX.length);
                const store = getQuickMarkBase64Store();
                if (typeof store[nodeId] === "string" && store[nodeId].trim() !== "") {
                    this._imageBase64Data = store[nodeId];
                    this.loadBackgroundImageFromBase64(this._imageBase64Data);
                }
            } else {
                this._imageBase64Data = base64Value;
                imageBase64Widget.value = `${QUICKMARK_BASE64_TOKEN_PREFIX}${this.id}`;
                getQuickMarkBase64Store()[String(this.id)] = base64Value;
                this.loadBackgroundImageFromBase64(base64Value);
            }
        } else if (this._imageBase64Data) {
            this.loadBackgroundImageFromBase64(this._imageBase64Data);
        } else if (this.properties.imageBase64Data) {
            this._imageBase64Data = this.properties.imageBase64Data;
            this.properties.imageBase64Data = "";
            this.loadBackgroundImageFromBase64(this._imageBase64Data);
        }

        this.properties.brushPaths = [];
        const brushDataWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA) || this.widgets[2];
        if (brushDataWidget && brushDataWidget.value) {
            try {
                let brushData = brushDataWidget.value;
                if (Array.isArray(brushData)) {
                    brushData = brushData.find(v => typeof v === "string" && v.trim() !== "") ?? brushData[0];
                }
                brushData = brushData ?? "";
                if (typeof brushData !== "string") brushData = String(brushData);
                if (brushData.trim()) {
                    const strokes = brushData.split("|");
                    for (const stroke of strokes) {
                        if (!stroke.trim()) continue;
                        const parsed = parseStroke(stroke);
                        if (parsed.points.length > 0) {
                            this.properties.brushPaths.push(parsed);
                        }
                    }
                }
            } catch (_error) {
                this.properties.brushPaths = [];
            }
        }

        this.initButtons?.();
    };

    node.updateImageSize = function (width, height) {
        if (!width || !height || width <= 0 || height <= 0) {
            return;
        }

        this.properties.imageWidth = width;
        this.properties.imageHeight = height;

        const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
        const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
        if (widthWidget) widthWidget.value = width;
        if (heightWidget) heightWidget.value = height;

        const maxDisplaySize = 500;
        const scale = Math.min(
            maxDisplaySize / width,
            maxDisplaySize / height,
            1.0
        );

        const displayWidth = Math.max(300, Math.min(width * scale + shiftRight + shiftLeft, 800));
        const displayHeight = Math.max(420, this.min_size?.[1] || 0);

        this.size[0] = displayWidth;
        this.size[1] = displayHeight;

        this.updateThisNodeGraph?.();
    };

    node.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) {
            return false;
        }
        ctx.save();
        try {
            const canvasWidth = this.properties.imageWidth || 512;
            const canvasHeight = this.properties.imageHeight || 512;

            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(0, 0, shiftLeft, this.size[1]);

            const canvasAreaWidth = this.size[0] - shiftLeft - shiftRight;
            const canvasAreaHeight = this.size[1] - panelHeight;

            const scaleX = canvasAreaWidth / canvasWidth;
            const scaleY = canvasAreaHeight / canvasHeight;
            const scale = Math.min(scaleX, scaleY);

            const scaledWidth = canvasWidth * scale;
            const scaledHeight = canvasHeight * scale;
            const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
            const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

            ctx.fillStyle = "#3a3a3a";
            ctx.fillRect(shiftLeft, 0, canvasAreaWidth, this.size[1]);

            ctx.fillStyle = "rgba(100,100,100,0.3)";
            ctx.fillRect(offsetX, offsetY, scaledWidth, scaledHeight);
            ctx.strokeStyle = "rgba(150,150,150,0.3)";
            ctx.lineWidth = 1;
            const gridSize = 20;
            const gridScale = gridSize * scale;

            for (let x = offsetX; x <= offsetX + scaledWidth; x += gridScale) {
                ctx.beginPath();
                ctx.moveTo(x, offsetY);
                ctx.lineTo(x, offsetY + scaledHeight);
                ctx.stroke();
            }

            for (let y = offsetY; y <= offsetY + scaledHeight; y += gridScale) {
                ctx.beginPath();
                ctx.moveTo(offsetX, y);
                ctx.lineTo(offsetX + scaledWidth, y);
                ctx.stroke();
            }

            if (this._backgroundImageObj && this._backgroundImageObj.complete) {
                try {
                    ctx.globalAlpha = 1.0;
                    ctx.drawImage(this._backgroundImageObj, offsetX, offsetY, scaledWidth, scaledHeight);
                    ctx.globalAlpha = 1.0;
                } catch (_error) {
                }
            }

            ctx.lineCap = "round";
            ctx.lineJoin = "round";

            ctx.save();
            ctx.rect(offsetX, offsetY, scaledWidth, scaledHeight);
            ctx.clip();

            for (const pathObj of this.properties.brushPaths) {
                const path = pathObj.points || pathObj;
                const type = pathObj.type || "free";
                const pathSize = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
                const pathOpacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
                const pathColor = pathObj.color || this.properties.brushColor || "255,0,0";

                if (path.length < 2) continue;

                if (type === "square") {
                    const isMarker = !!pathObj.marker;
                    const fillColor = isMarker ? "255,255,0" : pathColor;
                    const rgb = fillColor.split(",").map(c => parseInt(c.trim()));
                    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;

                    const minX = Math.min(...path.map(point => point.x));
                    const maxX = Math.max(...path.map(point => point.x));
                    const minY = Math.min(...path.map(point => point.y));
                    const maxY = Math.max(...path.map(point => point.y));

                    ctx.fillRect(offsetX + minX * scale, offsetY + minY * scale, (maxX - minX) * scale, (maxY - minY) * scale);

                    if (pathObj.marker) {
                        const centerX = offsetX + (minX + maxX) * 0.5 * scale;
                        const centerY = offsetY + (minY + maxY) * 0.5 * scale;
                        const sideScaled = Math.min((maxX - minX) * scale, (maxY - minY) * scale);
                        const fontSize = Math.max(10, Math.min(200, sideScaled * 0.6));
                        ctx.fillStyle = "#000000";
                        ctx.font = `bold ${Math.floor(fontSize)}px Arial`;
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText(String(pathObj.marker), centerX, centerY);
                    }
                } else if (type === "box") {
                    const rgb = pathColor.split(",").map(c => parseInt(c.trim()));
                    ctx.lineWidth = pathSize * scale;
                    ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;

                    const minX = Math.min(...path.map(point => point.x));
                    const maxX = Math.max(...path.map(point => point.x));
                    const minY = Math.min(...path.map(point => point.y));
                    const maxY = Math.max(...path.map(point => point.y));

                    ctx.strokeRect(offsetX + minX * scale, offsetY + minY * scale, (maxX - minX) * scale, (maxY - minY) * scale);
                } else {
                    const rgb = pathColor.split(",").map(c => parseInt(c.trim()));
                    ctx.lineWidth = pathSize * scale;
                    ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;
                    ctx.beginPath();
                    for (let i = 0; i < path.length; i++) {
                        const x = offsetX + path[i].x * scale;
                        const y = offsetY + path[i].y * scale;
                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                }
            }

            ctx.restore();

            if (this.properties.currentPath.length > 0) {
                ctx.globalCompositeOperation = "source-over";

                ctx.save();
                ctx.rect(offsetX, offsetY, scaledWidth, scaledHeight);
                ctx.clip();

                if ((this.properties.brushType === "box" || this.properties.brushType === "square") && this.properties.boxStartPoint) {
                    const startPoint = this.properties.boxStartPoint;
                    const endPoint = this.properties.currentPath[this.properties.currentPath.length - 1];

                    let minX = Math.min(startPoint.x, endPoint.x);
                    let maxX = Math.max(startPoint.x, endPoint.x);
                    let minY = Math.min(startPoint.y, endPoint.y);
                    let maxY = Math.max(startPoint.y, endPoint.y);

                    let currentColor = this.properties.brushColor;
                    if (!currentColor) {
                        currentColor = "255,0,0";
                    }
                    const rgb = currentColor.split(",").map(c => parseInt(c.trim()));

                    if (this.properties.brushType === "square") {
                        const marker = this.properties.activeMarker;
                        const fillColor = marker ? "255,255,0" : currentColor;
                        const fillRgb = fillColor.split(",").map(c => parseInt(c.trim()));
                        ctx.fillStyle = `rgba(${fillRgb[0]},${fillRgb[1]},${fillRgb[2]},${this.properties.brushOpacity})`;
                        if (marker) {
                            const dx = endPoint.x - startPoint.x;
                            const dy = endPoint.y - startPoint.y;
                            const side = Math.max(Math.abs(dx), Math.abs(dy));
                            const signX = dx >= 0 ? 1 : -1;
                            const signY = dy >= 0 ? 1 : -1;
                            const squareEndX = startPoint.x + signX * side;
                            const squareEndY = startPoint.y + signY * side;
                            minX = Math.min(startPoint.x, squareEndX);
                            maxX = Math.max(startPoint.x, squareEndX);
                            minY = Math.min(startPoint.y, squareEndY);
                            maxY = Math.max(startPoint.y, squareEndY);
                        }
                        ctx.fillRect(offsetX + minX * scale, offsetY + minY * scale, (maxX - minX) * scale, (maxY - minY) * scale);
                        if (marker) {
                            const centerX = offsetX + (minX + maxX) * 0.5 * scale;
                            const centerY = offsetY + (minY + maxY) * 0.5 * scale;
                            const sideScaled = Math.min((maxX - minX) * scale, (maxY - minY) * scale);
                            const fontSize = Math.max(10, Math.min(200, sideScaled * 0.6));
                            ctx.fillStyle = "#000000";
                            ctx.font = `bold ${Math.floor(fontSize)}px Arial`;
                            ctx.textAlign = "center";
                            ctx.textBaseline = "middle";
                            ctx.fillText(String(marker), centerX, centerY);
                        }
                    } else {
                        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${this.properties.brushOpacity})`;
                        ctx.lineWidth = this.properties.brushSize * scale;
                        ctx.strokeRect(offsetX + minX * scale, offsetY + minY * scale, (maxX - minX) * scale, (maxY - minY) * scale);
                    }
                } else if (this.properties.brushType === "free") {
                    ctx.lineWidth = this.properties.brushSize * scale;

                    let currentColor = this.properties.brushColor;
                    if (!currentColor) {
                        currentColor = "255,0,0";
                    }
                    const rgb = currentColor.split(",").map(c => parseInt(c.trim()));

                    ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${this.properties.brushOpacity})`;
                    ctx.beginPath();
                    for (let i = 0; i < this.properties.currentPath.length; i++) {
                        const x = offsetX + this.properties.currentPath[i].x * scale;
                        const y = offsetY + this.properties.currentPath[i].y * scale;
                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                }

                ctx.restore();
            }

            for (const button of (this._buttons || [])) {
                drawButton(ctx, button, this);
            }

            if (this._brushTypeButtons) {
                drawButton(ctx, this._brushTypeButtons.free, this);
                drawButton(ctx, this._brushTypeButtons.box, this);
                drawButton(ctx, this._brushTypeButtons.square, this);
            }

            if (this._actionButtons) {
                for (const button of this._actionButtons) {
                    drawButton(ctx, button, this);
                }
            }

            for (const slider of this.properties.sliders) {
                const sliderX = slider.x;
                const sliderY = slider.y + slider.labelHeight;
                const sliderWidth = slider.width;
                const sliderHeight = slider.sliderHeight;

                ctx.fillStyle = "#b0b0b0";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                const valueText = slider.type === "opacity" ? `${slider.value}%` : slider.value;
                ctx.fillText(`${slider.label}: ${valueText}`, sliderX, slider.y);

                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(sliderX, sliderY, sliderWidth, sliderHeight);

                ctx.strokeStyle = "#333333";
                ctx.lineWidth = 1;
                ctx.strokeRect(sliderX, sliderY, sliderWidth, sliderHeight);

                const range = slider.max - slider.min;
                const progress = (slider.value - slider.min) / range;
                const thumbX = sliderX + progress * (sliderWidth - 4);

                ctx.fillStyle = "#4CAF50";
                ctx.fillRect(sliderX + 2, sliderY + 2, thumbX - sliderX - 2, sliderHeight - 4);

                ctx.fillStyle = "#ffffff";
                ctx.fillRect(thumbX - 2, sliderY + 1, 4, sliderHeight - 2);
            }

            if (this.properties.colorPalette) {
                const palette = this.properties.colorPalette;
                const paletteX = palette.x;
                const paletteY = palette.y;
                const gridX = paletteX;
                const gridY = paletteY + palette.labelHeight;

                ctx.fillStyle = "#b0b0b0";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                ctx.fillText(palette.label, paletteX, paletteY);

                const selectedValue = this.properties.brushColor || "255,0,0";
                const { swatchSize, swatchGap, cols, options } = palette;

                for (let i = 0; i < options.length; i++) {
                    const row = Math.floor(i / cols);
                    const col = i % cols;
                    const x = gridX + col * (swatchSize + swatchGap);
                    const y = gridY + row * (swatchSize + swatchGap);

                    const option = options[i];
                    const rgb = option.value.split(",").map(c => parseInt(c.trim()));
                    ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},1.0)`;
                    ctx.fillRect(x, y, swatchSize, swatchSize);

                    const isSelected = option.value === selectedValue;
                    ctx.strokeStyle = isSelected ? "#4CAF50" : "#555555";
                    ctx.lineWidth = isSelected ? 2 : 1;
                    ctx.strokeRect(x + 0.5, y + 0.5, swatchSize - 1, swatchSize - 1);

                    ctx.fillStyle = option.textColor || "#ffffff";
                    ctx.font = "bold 10px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(option.text, x + swatchSize / 2, y + swatchSize / 2);
                }
            }

            if (this.properties.markerPalette) {
                const palette = this.properties.markerPalette;
                const paletteX = palette.x;
                const paletteY = palette.y;
                const gridX = paletteX;
                const gridY = paletteY + palette.labelHeight;

                ctx.fillStyle = "#b0b0b0";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                ctx.fillText(palette.label, paletteX, paletteY);

                const selectedValue = this.properties.activeMarker;
                const { markerSize, markerGap, cols, options } = palette;

                for (let i = 0; i < options.length; i++) {
                    const row = Math.floor(i / cols);
                    const col = i % cols;
                    const x = gridX + col * (markerSize + markerGap);
                    const y = gridY + row * (markerSize + markerGap);

                    const option = options[i];
                    const isSelected = option.value === selectedValue;

                    ctx.fillStyle = "#ffff00";
                    ctx.fillRect(x, y, markerSize, markerSize);
                    ctx.strokeStyle = isSelected ? "#4CAF50" : "#444444";
                    ctx.lineWidth = isSelected ? 2 : 1;
                    ctx.strokeRect(x + 0.5, y + 0.5, markerSize - 1, markerSize - 1);

                    ctx.fillStyle = "#000000";
                    ctx.font = "bold 10px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(option.text, x + markerSize / 2, y + markerSize / 2);
                }
            }
        } finally {
            ctx.restore();
        }
        syncBrushDataWidget(this);
    };

    function drawButton(ctx, button, nodeRef) {
        const isHover = nodeRef.mouseX >= button.x && nodeRef.mouseX <= button.x + button.width &&
            nodeRef.mouseY >= button.y && nodeRef.mouseY <= button.y + button.height;

        if (button.isToggle && button.active) {
            ctx.fillStyle = "#4CAF50";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#45a049";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        } else if (isHover) {
            ctx.fillStyle = "#555555";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#666666";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        } else {
            ctx.fillStyle = "#3a3a3a";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#4a4a4a";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        }

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(button.text, button.x + button.width / 2, button.y + button.height / 2);
    }
}

function syncBrushDataWidget(node) {
    const brushDataStrings = (node.properties.brushPaths || []).map(pathObj => {
        const path = pathObj.points || pathObj;
        const mode = pathObj.mode || "brush";
        const type = pathObj.type || "free";
        const size = pathObj.size !== undefined ? pathObj.size : node.properties.brushSize;
        const opacity = pathObj.opacity !== undefined ? pathObj.opacity : node.properties.brushOpacity;
        const color = pathObj.color || node.properties.brushColor || "255,0,0";
        const pointsStr = path.map(point => `${point.x},${point.y}`).join(";");
        if (pathObj.marker) {
            return `${mode}:${type}:${size}:${opacity}:${color}:${pathObj.marker}:${pointsStr}`;
        }
        return `${mode}:${type}:${size}:${opacity}:${color}:${pointsStr}`;
    });

    const brushDataWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA) || node.widgets[2];
    if (brushDataWidget) {
        brushDataWidget.value = brushDataStrings.join("|");
    }
}

function setupHiddenWidgets(node) {
    const brushDataWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA);
    if (brushDataWidget) {
        brushDataWidget.hidden = true;
    }

    const brushSizeWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
    if (brushSizeWidget) {
        brushSizeWidget.hidden = true;
    }

    let widthWidget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
    let heightWidget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
    let imageBase64Widget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);

    if (!widthWidget) {
        widthWidget = node.addWidget("number", WIDGET_NAMES.IMAGE_WIDTH, 512, () => { }, { min: 64, max: 4096 });
        widthWidget.hidden = true;
    }
    if (!heightWidget) {
        heightWidget = node.addWidget("number", WIDGET_NAMES.IMAGE_HEIGHT, 512, () => { }, { min: 64, max: 4096 });
        heightWidget.hidden = true;
    }
    if (!imageBase64Widget) {
        imageBase64Widget = node.addWidget("text", WIDGET_NAMES.IMAGE_BASE64, "", () => { });
    }
    if (imageBase64Widget) {
        imageBase64Widget.hidden = true;
    }

    node._backgroundImageObj = null;
    node._imageBase64Data = "";
    node.properties.imageBase64Data = "";
}

function parseStroke(stroke) {
    let mode = "brush";
    let type = "free";
    let size = 10;
    let opacity = 1.0;
    let color = "255,0,0";
    let marker = null;
    let pointsStr = stroke;

    if (stroke.includes(":")) {
        const parts = stroke.split(":");
        if (parts[0] === "brush" || parts[0] === "erase") {
            mode = parts[0];
            if (parts.length >= 6) {
                const maybeType = parts[1];
                if (maybeType === "free" || maybeType === "box" || maybeType === "square") {
                    type = maybeType;
                }
                size = parseFloat(parts[2]) || 10;
                opacity = parseFloat(parts[3]);
                if (!Number.isFinite(opacity)) opacity = 1.0;
                const rgbParts = (parts[4] || "").split(",").map(v => parseInt(v.trim()));
                if (rgbParts.length === 3 && isValidRGB(rgbParts[0], rgbParts[1], rgbParts[2])) {
                    color = `${rgbParts[0]},${rgbParts[1]},${rgbParts[2]}`;
                }
                const maybeMarker = parts[5];
                const looksLikePoint = typeof maybeMarker === "string" && maybeMarker.includes(",");
                const isMarkerToken = typeof maybeMarker === "string" && ["1", "2", "3", "4", "5", "6"].includes(maybeMarker);
                if (isMarkerToken && !looksLikePoint && parts.length >= 7) {
                    marker = maybeMarker;
                    pointsStr = parts.slice(6).join(":");
                } else {
                    pointsStr = parts.slice(5).join(":");
                }
            } else {
                pointsStr = parts.slice(1).join(":");
            }
        } else if (parts[0] === "free" || parts[0] === "box" || parts[0] === "square") {
            type = parts[0];
            pointsStr = parts.slice(1).join(":");
        }
    }

    const points = pointsStr.split(";").filter(point => point.trim() !== "");
    const path = [];
    for (const point of points) {
        if (!point.trim()) continue;
        const coords = point.split(",");
        if (coords.length === 2) {
            path.push({
                x: parseFloat(coords[0]),
                y: parseFloat(coords[1])
            });
        }
    }

    return {
        points: path,
        mode,
        type,
        size,
        opacity,
        color,
        marker
    };
}

function isValidRGB(r, g, b) {
    return [r, g, b].every(v => Number.isFinite(v) && v >= 0 && v <= 255);
}

function initInteractionBindings(node, state) {
    const { shiftLeft, shiftRight, panelHeight } = state.layout;

    node.onKeyDown = function () {
        if (!this.capture) {
            return false;
        }
        return false;
    };

    node.onMouseDown = function (e) {
        if (e.canvasY - this.pos[1] < 0) {
            return false;
        }

        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        if (mouseX <= 80) {
            if (this.properties.sliders) {
                for (const slider of this.properties.sliders) {
                    const sliderX = slider.x;
                    const sliderY = slider.y + slider.labelHeight;
                    const sliderWidth = slider.width;
                    const sliderHeight = slider.sliderHeight;

                    if (mouseX >= sliderX && mouseX <= sliderX + sliderWidth &&
                        mouseY >= sliderY && mouseY <= sliderY + sliderHeight) {
                        slider.isDragging = true;
                        this.capture = true;
                        this.captureInput(true);
                        return true;
                    }
                }
            }

            if (this.properties.colorPalette) {
                const palette = this.properties.colorPalette;
                const gridX = palette.x;
                const gridY = palette.y + palette.labelHeight;
                const gridWidth = palette.cols * palette.swatchSize + (palette.cols - 1) * palette.swatchGap;
                const gridHeight = palette.rows * palette.swatchSize + (palette.rows - 1) * palette.swatchGap;

                if (mouseX >= gridX && mouseX <= gridX + gridWidth &&
                    mouseY >= gridY && mouseY <= gridY + gridHeight) {
                    const col = Math.floor((mouseX - gridX) / (palette.swatchSize + palette.swatchGap));
                    const row = Math.floor((mouseY - gridY) / (palette.swatchSize + palette.swatchGap));

                    const cellX = gridX + col * (palette.swatchSize + palette.swatchGap);
                    const cellY = gridY + row * (palette.swatchSize + palette.swatchGap);
                    const inCell = mouseX >= cellX && mouseX <= cellX + palette.swatchSize &&
                        mouseY >= cellY && mouseY <= cellY + palette.swatchSize;
                    if (inCell) {
                        const idx = row * palette.cols + col;
                        const option = palette.options[idx];
                        if (option) {
                            this.properties.brushColor = option.value;
                            this.properties.brushMode = "brush";
                            this.updateThisNodeGraph?.();
                            return true;
                        }
                    }
                }
            }

            if (this.properties.markerPalette) {
                const palette = this.properties.markerPalette;
                const gridX = palette.x;
                const gridY = palette.y + palette.labelHeight;
                const gridWidth = palette.cols * palette.markerSize + (palette.cols - 1) * palette.markerGap;
                const gridHeight = palette.rows * palette.markerSize + (palette.rows - 1) * palette.markerGap;

                if (mouseX >= gridX && mouseX <= gridX + gridWidth &&
                    mouseY >= gridY && mouseY <= gridY + gridHeight) {
                    const col = Math.floor((mouseX - gridX) / (palette.markerSize + palette.markerGap));
                    const row = Math.floor((mouseY - gridY) / (palette.markerSize + palette.markerGap));

                    const cellX = gridX + col * (palette.markerSize + palette.markerGap);
                    const cellY = gridY + row * (palette.markerSize + palette.markerGap);
                    const inCell = mouseX >= cellX && mouseX <= cellX + palette.markerSize &&
                        mouseY >= cellY && mouseY <= cellY + palette.markerSize;
                    if (inCell) {
                        const idx = row * palette.cols + col;
                        const option = palette.options[idx];
                        if (option) {
                            this.properties.activeMarker = option.value;
                            this.properties.brushColor = "255,255,0";
                            this.properties.brushMode = "brush";
                            this.properties.brushType = "square";
                            if (this._brushTypeButtons) {
                                this._brushTypeButtons.free.active = false;
                                this._brushTypeButtons.box.active = false;
                                this._brushTypeButtons.square.active = false;
                            }
                            this.updateThisNodeGraph?.();
                            return true;
                        }
                    }
                }
            }
        }

        if (this._brushTypeButtons) {
            const freeBtn = this._brushTypeButtons.free;
            const boxBtn = this._brushTypeButtons.box;
            const squareBtn = this._brushTypeButtons.square;

            if (mouseX >= freeBtn.x && mouseX <= freeBtn.x + freeBtn.width &&
                mouseY >= freeBtn.y && mouseY <= freeBtn.y + freeBtn.height) {
                freeBtn.action();
                return true;
            }

            if (mouseX >= boxBtn.x && mouseX <= boxBtn.x + boxBtn.width &&
                mouseY >= boxBtn.y && mouseY <= boxBtn.y + boxBtn.height) {
                boxBtn.action();
                return true;
            }

            if (squareBtn && mouseX >= squareBtn.x && mouseX <= squareBtn.x + squareBtn.width &&
                mouseY >= squareBtn.y && mouseY <= squareBtn.y + squareBtn.height) {
                squareBtn.action();
                return true;
            }
        }

        if (this._actionButtons) {
            for (const button of this._actionButtons) {
                if (button.action && mouseX >= button.x && mouseX <= button.x + button.width &&
                    mouseY >= button.y && mouseY <= button.y + button.height) {
                    button.action();
                    return true;
                }
            }
        }

        if (this._buttons) {
            for (const button of this._buttons) {
                if (button.action && mouseX >= button.x && mouseX <= button.x + button.width &&
                    mouseY >= button.y && mouseY <= button.y + button.height) {
                    button.action();
                    return true;
                }
            }
        }

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        const canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        const canvasAreaHeight = this.size[1] - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        if (mouseX <= shiftLeft || mouseX >= this.size[0] - shiftRight) return false;

        if (e.button === 0) {
            const localX = e.canvasX - this.pos[0] - offsetX;
            const localY = e.canvasY - this.pos[1] - offsetY;

            const realX = localX / scale;
            const realY = localY / scale;

            this.properties.isDrawing = true;
            this.properties.currentPath = [{ x: realX, y: realY }];

            if (this.properties.brushType === "box" || this.properties.brushType === "square") {
                this.properties.boxStartPoint = { x: realX, y: realY };
            }

            this.capture = true;
            this.captureInput(true);
            return true;
        }

        return false;
    };

    node.onMouseMove = function (e, _pos, canvas) {
        if (!this.capture) {
            return;
        }

        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                if (slider.isDragging) {
                    const sliderX = slider.x;
                    const sliderY = slider.y + slider.labelHeight;
                    const sliderWidth = slider.width;
                    const sliderHeight = slider.sliderHeight;

                    const mouseX = e.canvasX - this.pos[0];
                    const mouseY = e.canvasY - this.pos[1];

                    if (mouseX >= sliderX && mouseX <= sliderX + sliderWidth &&
                        mouseY >= sliderY && mouseY <= sliderY + sliderHeight) {
                        const range = slider.max - slider.min;
                        const progress = (mouseX - sliderX) / sliderWidth;
                        const newValue = Math.round(slider.min + progress * range);
                        slider.value = Math.max(slider.min, Math.min(slider.max, newValue));

                        if (slider.type === "size") {
                            this.properties.brushSize = slider.value;
                            const brushSizeWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
                            if (brushSizeWidget) {
                                brushSizeWidget.value = this.properties.brushSize;
                            }
                        } else if (slider.type === "opacity") {
                            this.properties.brushOpacity = slider.value / 100;
                        }

                        this.updateThisNodeGraph?.();
                        return;
                    }
                }
            }
        }

        if (!this.properties.isDrawing) {
            return;
        }

        if (canvas.pointer.isDown === false) {
            this.onMouseUp(e);
            return;
        }
        this.valueUpdate(e);
    };

    node.valueUpdate = function (e) {
        if (!this.properties.isDrawing) {
            return;
        }

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        const canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        const canvasAreaHeight = this.size[1] - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        const mouseX = e.canvasX - this.pos[0] - offsetX;
        const mouseY = e.canvasY - this.pos[1] - offsetY;

        const realX = mouseX / scale;
        const realY = mouseY / scale;

        const lastPoint = this.properties.currentPath[this.properties.currentPath.length - 1];
        const dist = Math.sqrt(
            Math.pow(realX - lastPoint.x, 2) +
            Math.pow(realY - lastPoint.y, 2)
        );

        if (dist > 1) {
            this.properties.currentPath.push({ x: realX, y: realY });
            this.updateThisNodeGraph?.();
        }
    };

    node.onMouseUp = function () {
        if (!this.capture) {
            return;
        }

        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                slider.isDragging = false;
            }
        }

        if (this.properties.isDrawing && this.properties.currentPath.length > 0) {
            if ((this.properties.brushType === "box" || this.properties.brushType === "square") && this.properties.boxStartPoint) {
                const startPoint = this.properties.boxStartPoint;
                const endPoint = this.properties.currentPath[this.properties.currentPath.length - 1];

                let minX = Math.min(startPoint.x, endPoint.x);
                let maxX = Math.max(startPoint.x, endPoint.x);
                let minY = Math.min(startPoint.y, endPoint.y);
                let maxY = Math.max(startPoint.y, endPoint.y);

                const color = this.properties.brushColor;

                if (this.properties.brushType === "box") {
                    const boxPath = [
                        { x: minX, y: minY },
                        { x: maxX, y: minY },
                        { x: maxX, y: maxY },
                        { x: minX, y: maxY },
                        { x: minX, y: minY }
                    ];

                    this.properties.brushPaths.push({
                        points: boxPath,
                        mode: "brush",
                        type: "box",
                        size: this.properties.brushSize,
                        opacity: this.properties.brushOpacity,
                        color
                    });
                } else if (this.properties.brushType === "square") {
                    const marker = this.properties.activeMarker;
                    if (marker) {
                        const dx = endPoint.x - startPoint.x;
                        const dy = endPoint.y - startPoint.y;
                        const side = Math.max(Math.abs(dx), Math.abs(dy));
                        const signX = dx >= 0 ? 1 : -1;
                        const signY = dy >= 0 ? 1 : -1;
                        const squareEndX = startPoint.x + signX * side;
                        const squareEndY = startPoint.y + signY * side;
                        minX = Math.min(startPoint.x, squareEndX);
                        maxX = Math.max(startPoint.x, squareEndX);
                        minY = Math.min(startPoint.y, squareEndY);
                        maxY = Math.max(startPoint.y, squareEndY);
                    }
                    if (marker && Math.abs(maxX - minX) < 5 && Math.abs(maxY - minY) < 5) {
                        const defaultSize = 60;
                        maxX = minX + defaultSize;
                        maxY = minY + defaultSize;
                    }
                    const squarePath = [
                        { x: minX, y: minY },
                        { x: maxX, y: minY },
                        { x: maxX, y: maxY },
                        { x: minX, y: maxY },
                        { x: minX, y: minY }
                    ];

                    const colorForSquare = marker ? "255,255,0" : color;
                    this.properties.brushPaths.push({
                        points: squarePath,
                        mode: "brush",
                        type: "square",
                        size: this.properties.brushSize,
                        opacity: this.properties.brushOpacity,
                        color: colorForSquare,
                        marker
                    });
                }

                this.properties.boxStartPoint = null;
            } else {
                const color = this.properties.brushColor;

                this.properties.brushPaths.push({
                    points: [...this.properties.currentPath],
                    mode: "brush",
                    type: "free",
                    size: this.properties.brushSize,
                    opacity: this.properties.brushOpacity,
                    color
                });
            }

            this.properties.currentPath = [];
            syncBrushDataWidget(this);
        }

        this.properties.isDrawing = false;
        this.capture = false;
        this.captureInput(false);
        this.updateThisNodeGraph?.();
    };

    node.onSelected = function () {
        this.onMouseUp();
    };

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        if (originalOnConnectionsChange) {
            originalOnConnectionsChange.apply(this, arguments);
        }
    };

    const originalOnAfterExecuteNode = node.onAfterExecuteNode;
    node.onAfterExecuteNode = function (message) {
        if (originalOnAfterExecuteNode) {
            originalOnAfterExecuteNode.apply(this, arguments);
        }
        return message;
    };

    const originalOnWidgetChange = node.onWidgetChange;
    node.onWidgetChange = function (widget) {
        if (originalOnWidgetChange) {
            originalOnWidgetChange.apply(this, arguments);
        }

        if (!widget) {
            return;
        }

        if (widget.name === WIDGET_NAMES.BRUSH_SIZE) {
            this.properties.brushSize = widget.value || 10;
            this.updateThisNodeGraph?.();
        }

        if (widget.name === WIDGET_NAMES.IMAGE_WIDTH || widget.name === WIDGET_NAMES.IMAGE_HEIGHT) {
            const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
            const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
            if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
                this.updateImageSize(widthWidget.value, heightWidget.value);
            }
        }

        if (widget.name === WIDGET_NAMES.IMAGE_BASE64) {
            if (widget.value) {
                let base64Value = widget.value;
                if (Array.isArray(base64Value)) {
                    base64Value = base64Value.find(v => typeof v === "string" && v.trim() !== "") ?? base64Value[0];
                }
                base64Value = base64Value ?? "";
                if (typeof base64Value !== "string") {
                    if (base64Value && typeof base64Value === "object") {
                        if (typeof base64Value.base64 === "string") base64Value = base64Value.base64;
                        else if (typeof base64Value.data === "string") base64Value = base64Value.data;
                        else if (typeof base64Value.image_base64 === "string") base64Value = base64Value.image_base64;
                    }
                }
                if (typeof base64Value !== "string") base64Value = String(base64Value);
                if (base64Value.startsWith(QUICKMARK_BASE64_TOKEN_PREFIX)) {
                    widget.value = base64Value;
                    const nodeId = base64Value.slice(QUICKMARK_BASE64_TOKEN_PREFIX.length);
                    const store = getQuickMarkBase64Store();
                    if (typeof store[nodeId] === "string" && store[nodeId].trim() !== "") {
                        this._imageBase64Data = store[nodeId];
                        this.loadBackgroundImageFromBase64(this._imageBase64Data);
                    }
                } else {
                    this._imageBase64Data = base64Value;
                    getQuickMarkBase64Store()[String(this.id)] = base64Value;
                    widget.value = `${QUICKMARK_BASE64_TOKEN_PREFIX}${this.id}`;
                    this.loadBackgroundImageFromBase64(base64Value);
                }
            } else {
                this._backgroundImageObj = null;
                this._imageBase64Data = "";
                this.properties.imageBase64Data = "";
                this.updateThisNodeGraph?.();
            }
        }
    };

    node.loadImageFromFile = function () {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = event => {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = loadEvent => {
                try {
                    const dataURL = loadEvent.target.result;
                    let base64String = dataURL;
                    if (dataURL.includes(",")) {
                        base64String = dataURL.split(",")[1];
                    }

                    this._imageBase64Data = base64String;
                    getQuickMarkBase64Store()[String(this.id)] = base64String;

                    const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
                    if (imageBase64Widget) {
                        imageBase64Widget.value = `${QUICKMARK_BASE64_TOKEN_PREFIX}${this.id}`;
                    }

                    this.loadBackgroundImageFromBase64(dataURL);
                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];
                } catch (error) {
                    alert(`加载图片失败: ${error.message}`);
                }
            };
            reader.onerror = () => {
                alert("读取文件失败");
            };
            reader.readAsDataURL(file);
        };
        input.click();
    };

    node.loadBackgroundImageFromBase64 = function (base64String) {
        if (Array.isArray(base64String)) {
            base64String = base64String.find(v => typeof v === "string" && v.trim() !== "") ?? base64String[0];
        }
        base64String = base64String ?? "";
        if (typeof base64String !== "string") {
            if (base64String && typeof base64String === "object") {
                if (typeof base64String.base64 === "string") base64String = base64String.base64;
                else if (typeof base64String.data === "string") base64String = base64String.data;
                else if (typeof base64String.image_base64 === "string") base64String = base64String.image_base64;
            }
        }
        if (typeof base64String !== "string") base64String = String(base64String);

        let dataUrl = base64String.trim();
        if (dataUrl === "") {
            this._backgroundImageObj = null;
            this.updateThisNodeGraph?.();
            return;
        }

        try {
            const img = new Image();
            img.onload = () => {
                this._backgroundImageObj = img;
                this.updateImageSize(img.width, img.height);
                this.updateThisNodeGraph?.();
            };
            img.onerror = () => {
                this._backgroundImageObj = null;
            };
            if (dataUrl.startsWith("data:")) {
                if (dataUrl.includes(",")) {
                    const parts = dataUrl.split(",", 2);
                    const prefix = parts[0];
                    const payload = (parts[1] ?? "").replace(/\s+/g, "");
                    dataUrl = `${prefix},${payload}`;
                }
                img.src = dataUrl;
            } else {
                const payload = dataUrl.replace(/\s+/g, "");
                let mime = "image/png";
                if (payload.startsWith("/9j/")) mime = "image/jpeg";
                else if (payload.startsWith("R0lGOD")) mime = "image/gif";
                else if (payload.startsWith("UklGR")) mime = "image/webp";
                img.src = `data:${mime};base64,${payload}`;
            }
        } catch (_error) {
            this._backgroundImageObj = null;
        }
    };

    node.onDragOver = function (e) {
        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        const canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        const canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        if (mouseX >= offsetX && mouseX <= offsetX + scaledWidth &&
            mouseY >= offsetY && mouseY <= offsetY + scaledHeight) {
            if (e.dataTransfer && e.dataTransfer.types) {
                const hasFiles = Array.from(e.dataTransfer.types).includes("Files");
                if (hasFiles) {
                    e.preventDefault();
                    e.stopPropagation();
                    return true;
                }
            }
        }
        return false;
    };

    node.onDragDrop = function (e) {
        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        const canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        const canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = shiftLeft + panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        if (mouseX < offsetX || mouseX > offsetX + scaledWidth || mouseY < offsetY || mouseY > offsetY + scaledHeight) {
            return false;
        }

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];

            if (!file.type.startsWith("image/")) {
                return false;
            }

            const reader = new FileReader();
            reader.onload = loadEvent => {
                try {
                    const dataURL = loadEvent.target.result;
                    let base64String = dataURL;
                    if (dataURL.includes(",")) {
                        base64String = dataURL.split(",")[1];
                    }

                    this._imageBase64Data = base64String;
                    getQuickMarkBase64Store()[String(this.id)] = base64String;

                    const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
                    if (imageBase64Widget) {
                        imageBase64Widget.value = `${QUICKMARK_BASE64_TOKEN_PREFIX}${this.id}`;
                    }

                    this.loadBackgroundImageFromBase64(dataURL);
                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];
                } catch (error) {
                    alert(`加载图片失败: ${error.message}`);
                }
            };
            reader.onerror = () => {
                alert("读取文件失败");
            };
            reader.readAsDataURL(file);

            e.preventDefault();
            e.stopPropagation();
            return true;
        }

        return false;
    };
}

app.registerExtension({
    name: "shaobkj.quick_mark",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Shaobkj_QuickMark") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, []);
            }
            this.shaobkjQuickMark = new ShaobkjQuickMarkController(this);
            if (this.initButtons) {
                this.initButtons();
            }
        };
    }
});
