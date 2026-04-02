import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Shaobkj.ConcurrentImageEdit",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "Shaobkj_ConcurrentImageEdit_Sender" || nodeData.name === "Shaobkj_GroupedConcurrentImageEdit" || nodeData.name === "Shaobkj_APINode_Batch") {
			
            // Add a button widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Style the node (optional, if theme not covering it)
                // this.bgcolor = "#223322"; 

                // Add "Submit" Button
                // Check if button already exists to avoid duplicates (on reload)
                
                let buttonName = "🚀 发送任务 (Send)";
                let buttonAction = async () => {
                    console.log("[Shaobkj] Send button clicked - v2.2 (Anti-Crash)");
                    const graph = app.graph;
                    if (!graph) {
                        alert("⚠️ 错误：图表尚未就绪 (Graph not ready)。请稍后重试。");
                        return;
                    }
                    
                    // 🛡️ CRITICAL FIX: Sanitize widgets to prevent ComfyUI-Custom-Scripts (presetText.js) crash
                    // The other extension tries to read 'replace' on null values during serialization
                    if (this.widgets) {
                        for (const w of this.widgets) {
                            // Target string/text widgets that might have null values
                            if (w.value === null || w.value === undefined) {
                                console.warn(`[Shaobkj] Auto-fixing null value for widget: ${w.name}`);
                                w.value = "";
                            }
                        }
                    }

                    try {
                        console.log("[Shaobkj] Queuing prompt...");
                        await app.queuePrompt(0, 1);
                        console.log("[Shaobkj] Prompt queued successfully");
                        
                        app.ui.dialog.show("🚀 任务已发送至队列...");
                        setTimeout(() => { app.ui.dialog.close(); }, 1000);
                    } catch (error) {
                        console.error("[Shaobkj] Queue prompt failed:", error);
                        const stack = error.stack || "No stack trace";
                        alert("❌ 发送失败 (Send Failed):\n" + (error.message || error) + "\n\nStack:\n" + stack);
                    }
                };
                
                const hasButton = this.widgets && this.widgets.some(w => w.name === buttonName);
                
                if (!hasButton) {
                    // Use "Send" as value instead of null to prevent crashes in other extensions
                    this.addWidget("button", buttonName, "Send", buttonAction);
                }

                // Dynamic Input Slots are now managed by dynamic_inputs.js to avoid conflicts
                // We no longer perform checkSlots() here.
                
				return r;
			};
		}
	},
    
    // Optional: Listen for socket events to show success notification
    async setup() {

        
        api.addEventListener("shaobkj.concurrent.error", (event) => {
            const detail = event.detail;
            if (detail && detail.error) {
                alert("⚠️ 后台任务出错 (Background Task Error):\n" + detail.error);
            }
        });
    }
});
