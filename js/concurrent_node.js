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
                
                // Helper to check and add/remove slots
                const checkSlots = () => {
                    if (!this.inputs) this.inputs = [];
                    
                    // Find connected status of all image slots
                    const imageSlots = []; // Stores {index, slotObject}
                    
                    // 1. Identify existing image slots
                    for (let i = 0; i < this.inputs.length; i++) {
                        const slot = this.inputs[i];
                        if (slot.name.startsWith("image_")) {
                            const num = parseInt(slot.name.replace("image_", ""));
                            if (!isNaN(num)) {
                                imageSlots.push({ num: num, index: i, link: slot.link });
                            }
                        }
                    }
                    
                    // Sort by number
                    imageSlots.sort((a, b) => a.num - b.num);
                    
                    // 2. Determine target slot count
                    // We want to keep all connected slots, plus one empty slot at the end.
                    // But we also want to remove empty slots that are NOT the last one if we are "shrinking".
                    // Actually, simpler logic:
                    // Always ensure we have slots 1..N where N is (highest_connected_index + 1).
                    // If highest connected is 0 (none), we need image_1.
                    
                    let maxConnectedNum = 0;
                    for (const s of imageSlots) {
                        if (s.link !== null) {
                            if (s.num > maxConnectedNum) maxConnectedNum = s.num;
                        }
                    }
                    
                    const targetMaxNum = maxConnectedNum + 1;
                    
                    // 3. Add missing slots
                    for (let i = 1; i <= targetMaxNum; i++) {
                        const name = `image_${i}`;
                        const existing = this.findInputSlot(name);
                        if (existing === -1) {
                            this.addInput(name, "IMAGE");
                        }
                    }
                    
                    // 4. Remove extra slots (those > targetMaxNum)
                    // We iterate backwards to avoid index shifting issues when removing
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const slot = this.inputs[i];
                        if (slot.name.startsWith("image_")) {
                             const num = parseInt(slot.name.replace("image_", ""));
                             if (!isNaN(num) && num > targetMaxNum) {
                                 this.removeInput(i);
                             }
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
