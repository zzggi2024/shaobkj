import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FONT_STYLE_NODE_NAME = "Shaobkj_FontStyleSelector";
const STYLE_CACHE = {
    list: null,
    promise: null,
};

function ensureSelectorStyles() {
    if (document.getElementById("shaobkj-font-style-selector-style")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "shaobkj-font-style-selector-style";
    style.textContent = `
        .shaobkj-font-style-root {
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            padding: 6px;
            color: var(--fg-color, #ddd);
            font-size: 12px;
        }
        .shaobkj-font-style-replace-input {
            width: 100%;
            min-height: 44px;
            max-height: 180px;
            padding: 10px 12px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.25);
            color: inherit;
            outline: none;
            resize: none;
            overflow-y: hidden;
            line-height: 1.5;
            font: inherit;
            box-sizing: border-box;
        }
        .shaobkj-font-style-toolbar {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .shaobkj-font-style-search {
            flex: 1;
            min-width: 0;
            padding: 6px 8px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.25);
            color: inherit;
            outline: none;
        }
        .shaobkj-font-style-button {
            padding: 6px 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.08);
            color: inherit;
            cursor: pointer;
        }
        .shaobkj-font-style-button:hover {
            background: rgba(255, 255, 255, 0.14);
        }
        .shaobkj-font-style-status {
            opacity: 0.8;
            white-space: nowrap;
        }
        .shaobkj-font-style-body {
            display: flex;
            gap: 10px;
            min-height: 0;
            flex: 1;
        }
        .shaobkj-font-style-list {
            flex: 1.15;
            min-width: 0;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            align-content: flex-start;
            gap: 8px;
            padding: 4px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.15);
        }
        .shaobkj-font-style-card {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.06);
            cursor: pointer;
            user-select: none;
            transition: background 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
        }
        .shaobkj-font-style-card:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-1px);
        }
        .shaobkj-font-style-card.is-selected {
            border-color: rgba(93, 199, 112, 0.9);
            background: rgba(93, 199, 112, 0.18);
        }
        .shaobkj-font-style-check {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 1px solid rgba(255, 255, 255, 0.28);
            background: transparent;
            box-sizing: border-box;
            flex: 0 0 auto;
        }
        .shaobkj-font-style-card.is-selected .shaobkj-font-style-check {
            background: #5dc770;
            border-color: #5dc770;
        }
        .shaobkj-font-style-name {
            line-height: 1.2;
        }
        .shaobkj-font-style-preview {
            flex: 0.95;
            min-width: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.18);
        }
        .shaobkj-font-style-preview-image {
            width: 100%;
            aspect-ratio: 1.45 / 1;
            object-fit: contain;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .shaobkj-font-style-empty {
            opacity: 0.7;
            padding: 8px 10px;
        }
    `;
    document.head.appendChild(style);
}

function normalizeWidgetValue(value) {
    if (Array.isArray(value)) {
        return value.find(item => typeof item === "string") || "";
    }
    if (value == null) {
        return "";
    }
    return String(value);
}

async function getFontStyles() {
    if (STYLE_CACHE.list) {
        return STYLE_CACHE.list;
    }
    if (STYLE_CACHE.promise) {
        return STYLE_CACHE.promise;
    }

    STYLE_CACHE.promise = (async () => {
        const response = await api.fetchApi("/shaobkj/font_styles/list");
        if (response.status !== 200) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        STYLE_CACHE.list = Array.isArray(data?.styles) ? data.styles : [];
        return STYLE_CACHE.list;
    })();

    try {
        return await STYLE_CACHE.promise;
    } finally {
        STYLE_CACHE.promise = null;
    }
}

function createElement(tagName, className, textContent = "") {
    const element = document.createElement(tagName);
    if (className) {
        element.className = className;
    }
    if (textContent) {
        element.textContent = textContent;
    }
    return element;
}

function createSelectorUI(node, replaceWidget, selectionWidget) {
    ensureSelectorStyles();

    const root = createElement("div", "shaobkj-font-style-root");
    const replaceInput = createElement("textarea", "shaobkj-font-style-replace-input");
    replaceInput.rows = 1;
    replaceInput.placeholder = "替换文本输入";

    const toolbar = createElement("div", "shaobkj-font-style-toolbar");
    const searchInput = createElement("input", "shaobkj-font-style-search");
    searchInput.type = "text";
    searchInput.placeholder = "搜索风格...";

    const clearButton = createElement("button", "shaobkj-font-style-button", "清空");
    clearButton.type = "button";
    const status = createElement("div", "shaobkj-font-style-status", "加载中...");

    toolbar.append(searchInput, clearButton, status);

    const body = createElement("div", "shaobkj-font-style-body");
    const list = createElement("div", "shaobkj-font-style-list");
    const preview = createElement("div", "shaobkj-font-style-preview");
    const previewImage = createElement("img", "shaobkj-font-style-preview-image");
    previewImage.alt = "font style preview";
    preview.append(previewImage);
    body.append(list, preview);
    root.append(replaceInput, toolbar, body);

    const state = {
        styles: [],
        selected: new Set(),
        filterText: "",
        previewName: "",
    };

    function updateSelectionWidget() {
        selectionWidget.value = Array.from(state.selected).join(",");
        node.setDirtyCanvas?.(true, true);
    }

    function syncReplaceInputHeight() {
        replaceInput.style.height = "auto";
        const nextHeight = Math.min(Math.max(replaceInput.scrollHeight, 44), 180);
        replaceInput.style.height = `${nextHeight}px`;
    }

    function updatePreview(styleItem) {
        if (!styleItem) {
            previewImage.removeAttribute("src");
            state.previewName = "";
            return;
        }

        state.previewName = styleItem.name;
        previewImage.src = styleItem.image_url || "";
    }

    function getVisibleStyles() {
        const keyword = state.filterText.trim().toLowerCase();
        const filtered = state.styles.filter(styleItem => {
            if (!keyword) {
                return true;
            }
            const name = String(styleItem.name || "").toLowerCase();
            const nameCn = String(styleItem.name_cn || "").toLowerCase();
            const prompt = String(styleItem.prompt || "").toLowerCase();
            return name.includes(keyword) || nameCn.includes(keyword) || prompt.includes(keyword);
        });

        return filtered.sort((a, b) => {
            const selectedDiff = Number(state.selected.has(b.name)) - Number(state.selected.has(a.name));
            if (selectedDiff !== 0) {
                return selectedDiff;
            }
            return String(a.name_cn || a.name).localeCompare(String(b.name_cn || b.name), "zh-Hans-CN");
        });
    }

    function renderList() {
        list.innerHTML = "";
        const visibleStyles = getVisibleStyles();

        if (!visibleStyles.length) {
            list.append(createElement("div", "shaobkj-font-style-empty", "没有匹配的风格"));
            if (!state.previewName) {
                updatePreview(null);
            }
            status.textContent = `已选 ${state.selected.size} 个`;
            return;
        }

        for (const styleItem of visibleStyles) {
            const card = createElement("div", "shaobkj-font-style-card");
            if (state.selected.has(styleItem.name)) {
                card.classList.add("is-selected");
            }

            const check = createElement("div", "shaobkj-font-style-check");
            const name = createElement("div", "shaobkj-font-style-name", styleItem.name_cn || styleItem.name);
            card.append(check, name);

            card.addEventListener("mouseenter", () => updatePreview(styleItem));
            card.addEventListener("click", () => {
                if (state.selected.has(styleItem.name)) {
                    state.selected.delete(styleItem.name);
                } else {
                    state.selected.add(styleItem.name);
                }
                updateSelectionWidget();
                renderList();
                updatePreview(styleItem);
            });

            list.append(card);
        }

        if (!state.previewName) {
            const firstSelected = visibleStyles.find(styleItem => state.selected.has(styleItem.name));
            updatePreview(firstSelected || visibleStyles[0]);
        }

        status.textContent = `共 ${visibleStyles.length} 个，已选 ${state.selected.size} 个`;
    }

    function syncStateFromWidget() {
        state.selected.clear();
        const value = normalizeWidgetValue(selectionWidget.value);
        for (const item of value.split(",")) {
            const name = item.trim();
            if (name) {
                state.selected.add(name);
            }
        }
    }

    searchInput.addEventListener("input", event => {
        state.filterText = event.target.value || "";
        renderList();
    });

    replaceInput.addEventListener("input", event => {
        replaceWidget.value = event.target.value || "";
        syncReplaceInputHeight();
        node.setDirtyCanvas?.(true, true);
    });

    clearButton.addEventListener("click", () => {
        state.selected.clear();
        updateSelectionWidget();
        renderList();
    });

    syncStateFromWidget();
    replaceInput.value = normalizeWidgetValue(replaceWidget.value);
    syncReplaceInputHeight();

    getFontStyles()
        .then(styles => {
            state.styles = styles;
            renderList();
        })
        .catch(error => {
            status.textContent = "加载失败";
            list.innerHTML = "";
            list.append(createElement("div", "shaobkj-font-style-empty", `风格列表加载失败: ${error.message}`));
            updatePreview(null);
        });

    return root;
}

app.registerExtension({
    name: "shaobkj.font_style_selector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== FONT_STYLE_NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const apiWidgetIndex = Array.isArray(this.widgets) ? this.widgets.findIndex(widget => widget?.name === "API申请地址") : -1;
            if (apiWidgetIndex >= 0) {
                this.widgets.splice(apiWidgetIndex, 1);
            }

            const replaceWidget = this.widgets?.find(widget => widget.name === "替换文字");
            const selectionWidget = this.widgets?.find(widget => widget.name === "选择风格");
            if (!replaceWidget || !selectionWidget) {
                return;
            }

            replaceWidget.hidden = true;
            replaceWidget.computeSize = () => [0, -4];
            selectionWidget.hidden = true;
            selectionWidget.computeSize = () => [0, -4];

            if (!this.__shaobkjFontStyleWidget) {
                const root = createSelectorUI(this, replaceWidget, selectionWidget);
                this.__shaobkjFontStyleWidget = this.addDOMWidget("风格选择器", "shaobkj_font_style_selector", root);
            }

            if (!this.size || this.size[0] < 760 || this.size[1] < 560) {
                if (typeof this.setSize === "function") {
                    this.setSize([760, 560]);
                } else {
                    this.size = [760, 560];
                }
            }
        };
    },
});
