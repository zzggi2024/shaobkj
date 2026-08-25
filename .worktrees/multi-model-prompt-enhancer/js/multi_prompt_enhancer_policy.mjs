const COMMON_WIDGETS = [
  "功能类型",
  "运行渠道",
  "模型选择",
  "API密钥",
  "创作需求",
  "输出语言",
  "随机种",
];

const FEATURE_WIDGETS = {
  H3: [
    "生成模式",
    "目标时长",
    "画幅比例",
    "镜头数量",
    "改写幅度",
    "提示词模式",
    "官方场景预设",
    "原生音频",
  ],
  Seedance: [
    "任务意图",
    "目标时长",
    "镜头数量",
    "组织方式",
    "改写幅度",
    "提示词模式",
  ],
  "Music 3": [
    "歌词模式",
    "歌词语言",
    "目标时长",
    "创作幅度",
    "质量模式",
    "歌曲结构",
  ],
};

const ADVANCED_WIDGETS = {
  H3: ["参考模板", "素材补充", "硬性要求"],
  Seedance: ["参考模板", "素材用途", "素材补充", "硬性要求"],
  "Music 3": ["BPM", "调式", "拍号", "硬性要求与排除项"],
};

const MEDIA_SERIES = {
  H3: [
    { prefix: "参考图片", type: "IMAGE", max: 9 },
    { prefix: "参考视频", type: "VIDEO", max: 3 },
    { prefix: "参考音频", type: "AUDIO", max: 3 },
  ],
  Seedance: [
    { prefix: "参考图片", type: "IMAGE", max: 9 },
    { prefix: "参考视频", type: "VIDEO", max: 3 },
  ],
  "Music 3": [],
};

export const PERMANENT_EXPLANATION_PANEL = false;
export const MAX_NODE_HEIGHT = 760;
export const MODEL_LOCK_PROPERTY = "shaobkjModelLocked";

export function persistModelLock(properties, locked) {
  return {
    ...(properties && typeof properties === "object" ? properties : {}),
    [MODEL_LOCK_PROPERTY]: Boolean(locked),
  };
}

export function modelLockFromWorkflow(properties, currentValue) {
  if (String(currentValue || "智能选择") === "智能选择") return false;
  if (properties && Object.prototype.hasOwnProperty.call(properties, MODEL_LOCK_PROPERTY)) {
    return properties[MODEL_LOCK_PROPERTY] === true;
  }
  return true;
}

export function modelSelectionAfterRefresh(currentValue, models) {
  const options = new Set((models || []).filter(Boolean));
  const current = String(currentValue || "智能选择");
  return options.has(current) ? current : "智能选择";
}

export function createLatestRequestQueue(worker) {
  let running = false;
  let latestValue;
  let latestVersion = 0;
  let waiters = [];

  async function drain() {
    try {
      while (true) {
        const version = latestVersion;
        const value = latestValue;
        let result;
        let failure;
        try {
          result = await worker(value, () => version === latestVersion);
        } catch (error) {
          failure = error;
        }
        if (version !== latestVersion) continue;

        const settled = waiters;
        waiters = [];
        for (const waiter of settled) {
          if (failure) waiter.reject(failure);
          else waiter.resolve(result);
        }
        break;
      }
    } finally {
      running = false;
      if (waiters.length) {
        running = true;
        void drain();
      }
    }
  }

  return {
    run(value) {
      const samePendingValue = running && Object.is(value, latestValue);
      if (!samePendingValue) {
        latestValue = value;
        latestVersion += 1;
      }
      const promise = new Promise((resolve, reject) => {
        waiters.push({ resolve, reject });
      });
      if (!running) {
        running = true;
        void drain();
      }
      return promise;
    },
  };
}

function isLinked(input) {
  return Boolean(input && input.linked === true)
    || (input?.link !== null && input?.link !== undefined && input?.link !== -1);
}

function inputIndex(name, prefix) {
  const match = new RegExp(`^${prefix}(\\d+)$`).exec(String(name || ""));
  return match ? Number(match[1]) : 0;
}

function seriesInputs(inputs, series) {
  return (Array.isArray(inputs) ? inputs : [])
    .filter((input) => input?.name && inputIndex(input.name, series.prefix) > 0)
    .sort((left, right) => inputIndex(left.name, series.prefix) - inputIndex(right.name, series.prefix));
}

export function visibleWidgetNames(feature, values = {}) {
  const result = new Set(COMMON_WIDGETS);
  for (const name of FEATURE_WIDGETS[feature] || []) result.add(name);

  if (feature === "Music 3") {
    const lyricsMode = String(values["歌词模式"] || "自动");
    const songStructure = String(values["歌曲结构"] || "自动");
    if (["严格保留", "按要求润色"].includes(lyricsMode)) result.add("原歌词");
    if (lyricsMode === "按要求润色") result.add("润色要求");
    if (songStructure === "自定义") result.add("自定义结构标签");
  }

  if (values.__advancedExpanded === true || values["高级设置"] === true) {
    for (const name of ADVANCED_WIDGETS[feature] || []) result.add(name);
  }
  return result;
}

export function mediaSeriesForFeature(feature) {
  return (MEDIA_SERIES[feature] || []).map((series) => ({ ...series }));
}

export function reconcileSeries(inputs, series, _feature = "") {
  if (!series?.prefix || !series?.type) return [];
  const current = seriesInputs(inputs, series);
  const linkedIndexes = current
    .filter(isLinked)
    .map((input) => inputIndex(input.name, series.prefix));
  const highestLinked = linkedIndexes.length ? Math.max(...linkedIndexes) : 0;
  const targetCount = Math.min(series.max || Number.POSITIVE_INFINITY, Math.max(1, highestLinked + 1));
  const operations = [];
  const existingNames = new Set(current.map((input) => String(input.name)));

  for (let index = 1; index <= targetCount; index += 1) {
    const name = `${series.prefix}${index}`;
    if (!existingNames.has(name)) {
      operations.push({ action: "add", name, type: series.type });
    }
  }

  for (const input of [...current].reverse()) {
    const index = inputIndex(input.name, series.prefix);
    if (index > targetCount && !isLinked(input)) {
      operations.push({ action: "remove", name: input.name });
    }
  }
  return operations;
}

export function reconcileFeaturePorts(inputs, feature, values = {}) {
  const supported = mediaSeriesForFeature(feature);
  const supportedByPrefix = new Map(supported.map((series) => [series.prefix, series]));
  const knownPrefixes = ["参考图片", "参考视频", "参考音频"];
  const operations = [];

  for (const prefix of knownPrefixes) {
    const current = (Array.isArray(inputs) ? inputs : [])
      .filter((input) => input?.name && inputIndex(input.name, prefix) > 0);
    const series = supportedByPrefix.get(prefix);
    if (!series) {
      for (const input of current) {
        if (isLinked(input)) {
          operations.push({
            action: "retain-warning",
            name: input.name,
            label: `${input.name}（当前不参与）`,
          });
        } else {
          operations.push({ action: "remove", name: input.name });
        }
      }
    }
  }

  const mode = String(values["生成模式"] || "自动");
  const supportedFixed = feature === "Seedance"
    ? new Set(["首帧图", "尾帧图"])
    : feature === "H3"
      ? new Set(mode === "I2VA" ? ["首帧图"] : ["FL2VA", "自动"].includes(mode) ? ["首帧图", "尾帧图"] : mode === "L2VA" ? ["尾帧图"] : [])
      : new Set();
  for (const name of ["首帧图", "尾帧图"]) {
    const input = (Array.isArray(inputs) ? inputs : []).find((item) => item?.name === name);
    if (supportedFixed.has(name)) {
      continue;
    } else if (input && isLinked(input)) {
      operations.push({ action: "retain-warning", name, label: `${name}（当前不参与）` });
    } else if (input) {
      operations.push({ action: "remove", name });
    }
  }
  return operations;
}

export function computedNodeHeight(visibleWidgetCount, expanded = false) {
  const count = Math.max(0, Number(visibleWidgetCount) || 0);
  const base = expanded ? 280 : 220;
  const perWidget = expanded ? 20 : 18;
  return Math.min(MAX_NODE_HEIGHT, Math.max(base, Math.round(base + count * perWidget)));
}

export { COMMON_WIDGETS, FEATURE_WIDGETS, ADVANCED_WIDGETS };
