
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PLUGIN_NAME = "shaobkj";
const SAFE_PLUGIN_NAME = "shaobkj";
const STORAGE_KEY = `Shaobkj_${SAFE_PLUGIN_NAME}_access_key`;
const AUTH_API = `/${SAFE_PLUGIN_NAME}/auth`;
let authorized = false;
let showing = false;

function getStoredAccessKey() {
    return localStorage.getItem(STORAGE_KEY) || "";
}

function setStoredAccessKey(value) {
    const normalized = String(value || "").trim();
    if (normalized) localStorage.setItem(STORAGE_KEY, normalized);
    else localStorage.removeItem(STORAGE_KEY);
}

async function checkStatus() {
    const instanceId = getCloudInstanceIdFromRuntime();
    const statusUrl = instanceId ? `${AUTH_API}/status?instance_id=${encodeURIComponent(instanceId)}` : `${AUTH_API}/status`;
    const response = await api.fetchApi(statusUrl);
    const data = await response.json();
    authorized = !!data?.ok;
    if (!authorized && data?.message) {
        setStoredAccessKey("");
        showAuthDialog(data.message);
    }
    return authorized;
}

function normalizeCloudInstanceId(value) {
    return String(value || "").trim().toUpperCase().replace(/[^A-Z0-9_.:-]/g, "").slice(0, 128);
}

function getCloudInstanceIdFromUrl(url) {
    const rawUrl = String(url || "").trim();
    if (!rawUrl) return "";
    let parsedUrl;
    try {
        parsedUrl = new URL(rawUrl, window.location?.href || undefined);
    } catch (_error) {
        return "";
    }
    const dashboardMatch = parsedUrl.pathname.match(/\/dashboard\/sd\/([^/?#]+)/i);
    if (dashboardMatch?.[1]) return normalizeCloudInstanceId(dashboardMatch[1]);
    return "";
}

function getCloudInstanceIdFromRuntime() {
    return getCloudInstanceIdFromUrl(window.location?.href) || getCloudInstanceIdFromUrl(document.referrer);
}

async function verifyAccessKey(accessKey) {
    const instanceId = getCloudInstanceIdFromRuntime();
    const response = await api.fetchApi(`${AUTH_API}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: accessKey, instance_id: instanceId }),
    });
    const data = await response.json();
    if (!response.ok || !data?.ok) {
        throw new Error(data?.message || "授权码错误");
    }
    authorized = true;
    setStoredAccessKey(accessKey);
    return data;
}

function showAuthDialog(initialMessage = "") {
    if (authorized || showing) return;
    showing = true;
    const overlay = document.createElement("div");
    overlay.style.cssText = [
        "position:fixed", "inset:0", "background:rgba(0,0,0,0.45)",
        "display:flex", "align-items:center", "justify-content:center",
        "z-index:999999", "padding:20px", "pointer-events:auto",

    ].join(";");
    const panel = document.createElement("form");
    panel.style.cssText = [
        "width:min(420px,100%)", "background:#222", "border:1px solid #333",
        "border-radius:10px", "padding:18px", "color:#fff",
        "box-shadow:0 16px 50px rgba(0,0,0,0.45)", "display:grid", "gap:14px",
        "position:relative", "z-index:1000000", "pointer-events:auto",

    ].join(";");
    panel.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:22px;">🔑</span>
            <h3 style="margin:0;font-size:17px;">${PLUGIN_NAME} 授权码</h3>
        </div>
        <label style="display:grid;gap:6px;">
            <span style="font-size:13px;color:#aaa;">输入授权码</span>
            <input name="access_key" type="text" autocomplete="off" placeholder="请输入授权码..."
                style="width:100%;box-sizing:border-box;padding:9px 12px;border-radius:7px;border:1px solid #555;background:#111;color:#fff;outline:none;font-size:13px;" />
        </label>
        <div id="shaobkj-release-auth-status" style="font-size:12px;min-height:18px;"></div>
        <div style="display:flex;gap:8px;justify-content:flex-end;">
            <button type="button" data-action="cancel" style="padding:8px 14px;border-radius:8px;border:1px solid #555;background:#333;color:#fff;cursor:pointer;">取消</button>
            <button type="submit" data-action="confirm" style="padding:8px 16px;border-radius:8px;border:none;background:#4ec7b9;color:#082420;font-weight:700;cursor:pointer;">确认</button>
        </div>`;
    const inputEl = panel.querySelector('input[name="access_key"]');
    const statusEl = panel.querySelector("#shaobkj-release-auth-status");
    const confirmBtn = panel.querySelector('[data-action="confirm"]');
    inputEl.value = getStoredAccessKey();
    function cleanup() {
        showing = false;
        overlay.remove();
    }
    function setStatus(message, color = "#ffb4b4") {
        statusEl.textContent = message || "";
        statusEl.style.color = color;
    }
    panel.querySelector('[data-action="cancel"]').addEventListener("click", () => {
        setStatus("请先输入正确的授权码。");
        inputEl.focus();
    });
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            setStatus("请先输入正确的授权码。");
            inputEl.focus();
        }
    });
    panel.addEventListener("submit", async (event) => {
        event.preventDefault();
        const accessKey = inputEl.value.trim();
        if (!accessKey) {
            setStatus("授权码不能为空");
            inputEl.focus();
            return;
        }
        confirmBtn.disabled = true;
        confirmBtn.textContent = "验证中...";
        setStatus("正在验证授权码...", "#aaa");
        try {
            await verifyAccessKey(accessKey);
            setStatus("恭喜，验证完成", "#8ee8a8");
            window.alert("恭喜，验证完成");
            cleanup();
        } catch (_error) {
            setStatus("授权码错误，请重新输入");
            inputEl.focus();
        } finally {
            confirmBtn.disabled = false;
            confirmBtn.textContent = "确认";
        }
    });
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    if (initialMessage) {
        setStatus(initialMessage);
    }
    setTimeout(() => inputEl.focus(), 50);
}

app.registerExtension({
    name: `shaobkj.release_auth.${SAFE_PLUGIN_NAME}`,
    async setup() {
        const runCheck = async () => {
            try {
                const ok = await checkStatus();
                if (!ok) showAuthDialog();
            } catch (_error) {
                authorized = false;
                showAuthDialog("授权状态检查失败，请重新输入");
            }
        };
        setTimeout(runCheck, 500);
        setInterval(runCheck, 5000);
    },
});
