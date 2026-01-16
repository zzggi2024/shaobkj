import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "Shaobkj.SoraVideo",
	async nodeCreated(node, app) {
		if (node.comfyClass === "Shaobkj_Sora_Video") {
            const durationWidget = node.widgets.find(w => w.name === "生成时长");
            const modelWidget = node.widgets.find(w => w.name === "模型");

            if (durationWidget && modelWidget) {
                // 保存原始回调
                const originalCallback = durationWidget.callback;
                
                // 重写回调
                durationWidget.callback = function(value) {
                    // value 可能是 "25" 字符串
                    if (String(value) === "25" && modelWidget.value === "sora-2") {
                        alert("时长25秒请选择sora-2-pro模型");
                        modelWidget.value = "sora-2-pro";
                        
                        // 强制刷新画布以更新 UI 显示
                        app.graph.setDirtyCanvas(true, true);
                    }
                    
                    // 执行原始回调（如果有）
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };
            }
        }
    }
});
