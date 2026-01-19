import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Shaobkj.ConcurrentImageEdit",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "Shaobkj_ConcurrentImageEdit") {
			
            // Add a button widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Style the node
                this.color = "#006600"; // Green background for the node header
                this.bgcolor = "#003300"; // Darker green background

                // Add "Submit" Button
                this.addWidget("button", "🔴 立即提交 (Submit)", null, () => {
                    // Mapping widget names to API keys
                    const mapping = {
                        "image": "image_name",
                        "提示词": "prompt",
                        "API密钥": "api_key",
                        "API地址": "api_url",
                        "模型选择": "model",
                        "使用系统代理": "use_proxy",
                        "分辨率": "resolution",
                        "图片比例": "aspect_ratio",
                        "输入图像-长边设置": "long_side",
                        "等待时间": "wait_time",
                        "seed": "seed",
                        "保存路径": "save_path"
                    };

                    // Trigger Global Queue
                    app.queuePrompt(0, 1);
                    
                    // Optional: Show toast
                    app.ui.dialog.show("已添加到队列 (Added to Queue)...");
                    setTimeout(() => { app.ui.dialog.close(); }, 1000);
                });

                // Dynamic Image Inputs Logic (For Slot Expansion)
                // We need to manage input slots (connectors), NOT widgets, for image inputs.
                
                // Helper to check and add slots
                const checkSlots = () => {
                    if (!this.inputs) this.inputs = [];
                    
                    // Find highest connected image slot
                    let maxIndex = 0;
                    for (const slot of this.inputs) {
                        if (slot.name.startsWith("image_")) {
                            const num = parseInt(slot.name.replace("image_", ""));
                            if (!isNaN(num) && slot.link !== null) {
                                if (num > maxIndex) maxIndex = num;
                            }
                        }
                    }
                    
                    // Target: ensure we have at least maxIndex + 1 (next empty slot)
                    // But we start with image_1
                    const targetIndex = maxIndex + 1;
                    
                    // Ensure slots exist up to targetIndex
                    for (let i = 1; i <= targetIndex; i++) {
                        const name = `image_${i}`;
                        const existing = this.findInputSlot(name);
                        if (existing === -1) {
                            this.addInput(name, "IMAGE");
                        }
                    }
                };

                // Hook into connection changes to expand slots
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info, slot) {
                    const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                    // Only care about input connections (type 1)
                    if (type === 1) {
                         // Use timeout to let connection settle
                         setTimeout(checkSlots, 50);
                    }
                    return r;
                };
                
                // Also check on init
                setTimeout(checkSlots, 100);

				return r;
			};
		}
	},
    
    // Optional: Listen for socket events to show success notification
    async setup() {
        api.addEventListener("shaobkj.concurrent.success", (event) => {
            const detail = event.detail;
            if (detail && detail.filename) {
                // We can try to show a notification or toast
                // ComfyUI doesn't have a standard persistent toast, but we can log
                console.log("[Shaobkj] Concurrent task success:", detail);
            }
        });
        
        api.addEventListener("shaobkj.concurrent.error", (event) => {
             const detail = event.detail;
             if (detail && detail.error) {
                 alert("⚠️ 后台任务出错 (Background Task Error):\n" + detail.error);
             }
        });
    }
});
