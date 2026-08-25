export const H3_PROMPT_EDITOR_HEIGHT = 270;

export function applyH3PromptEditorLayout(container, panel) {
    const height = `${H3_PROMPT_EDITOR_HEIGHT}px`;
    Object.assign(container.style, {
        width: "100%",
        height,
        minHeight: height,
        maxHeight: height,
        boxSizing: "border-box",
        overflow: "hidden",
        padding: "8px",
    });
    Object.assign(panel.style, {
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
    });
}

export function createH3PromptEditorWidgetOptions(options = {}) {
    return {
        ...options,
        getMinHeight: () => H3_PROMPT_EDITOR_HEIGHT,
        getHeight: () => H3_PROMPT_EDITOR_HEIGHT,
        getMaxHeight: () => H3_PROMPT_EDITOR_HEIGHT,
    };
}
