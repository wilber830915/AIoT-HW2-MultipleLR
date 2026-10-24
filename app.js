const CONFIGS = {
  flood: {
    key: 'flood',
    data: window.FLOOD_DATA,
    headerDataset: 'Kaggle S4E5 · Flood',
    heroTitle: '洪水機率預測：從 30 個候選特徵找出最精簡的線性模型',
    heroDesc: '以 <strong>Kaggle Playground S4E5</strong> 的洪水資料為例，完整走過 CRISP-DM 流程。比較 <strong>SelectKBest</strong>、<strong>RFE</strong>、<strong>Sequential</strong>、<strong>Lasso</strong>、<strong>Permutation Importance</strong> 五種特徵選擇方法，在 <code>k = 14 / 20 / 25 / 30</code> 下訓練 <strong>Multiple Linear Regression</strong>，並用 <strong>95% split conformal prediction interval</strong> 量化不確定性。',
    targetDisplay: 'FloodProbability',
    targetShort: 'FloodProbability',
    probability: true,
    currency: false,
    metricDigits: 6,
    targetDigits: 4,
    liveTitle: '即時洪水機率預測器',
    liveSubtitle: '調整 20 個原始風險因子，模型會自動計算互動特徵並即時預測。',
    liveLabel: 'Predicted FloodProbability',
    predSubtitle: '依預測機率 0.45 / 0.50 / 0.55 分級',
    scenarioLabels: ['低風險情境', '中位數情境', '高風險情境'],
    scenarioStats: { low: 'min', median: 'median', high: 'max' },
    targetChartSubtitle: 'FloodProbability 在 111 萬筆訓練資料中的分布。',
    dataNoun: '訓練資料',
    missingText: '缺失值 0',
    targetMeanText: '約 0.504',
    crispBusinessTitle: '預測洪水機率',
    crispBusiness: '目標為 <code>FloodProbability</code>，官方競賽指標為 R²，並輔以 RMSE、MAE。模型選用可解釋性高的多元線性迴歸。',
    crispPrep: '建立 10 組領域導向的互動特徵，形成 30 個候選特徵。資料切為 fit / validation / calibration / test，並以中位數補值、標準化。',
    crispDeployment: '輸出 <code>best_model.joblib</code>、<code>best_features.json</code>、<code>submission.csv</code> 與診斷圖。本頁的即時預測器直接使用匯出的原始尺度係數。'
  },
  automobile: {
    key: 'automobile',
    data: window.AUTOMOBILE_DATA,
    headerDataset: 'Kaggle · Automobile',
    heroTitle: '汽車價格預測：從 one-hot 特徵中找出精簡的線性模型',
    heroDesc: '以 <strong>Kaggle / UCI Automobile dataset</strong> 為例，目標為 <strong>Price（美元）</strong>。比較 <strong>SelectKBest</strong>、<strong>RFE</strong>、<strong>Sequential</strong>、<strong>Lasso</strong>、<strong>Permutation Importance</strong> 五種特徵選擇方法，在 <code>k = 8 / 16 / 24 / 32</code> 下訓練 <strong>Multiple Linear Regression</strong>，類別特徵先以 one-hot encoding 處理，並用 <strong>95% split conformal prediction interval</strong> 量化估價不確定性。',
    targetDisplay: 'Price (USD)',
    targetShort: 'Price',
    probability: false,
    currency: true,
    metricDigits: 0,
    targetDigits: 0,
    liveTitle: '即時汽車價格預測器',
    liveSubtitle: '調整 15 個數值特徵與 10 個類別特徵，模型會即時估算價格。',
    liveLabel: 'Predicted Price',
    predSubtitle: '以美元（USD）計價的線性估價',
    scenarioLabels: ['低價設定', '中位數設定', '高價設定'],
    scenarioStats: { low: 'q25', median: 'median', high: 'q75' },
    targetChartSubtitle: 'Price 在 201 筆有效資料中的分布。',
    dataNoun: '有效資料',
    missingText: '缺失值以中位數或眾數補值',
    targetMeanText: '約 13,207 美元',
    crispBusinessTitle: '預測汽車價格',
    crispBusiness: '目標為 <code>Price</code>（美元），以 R² 為主要評估指標，並輔以 RMSE、MAE。模型選用可解釋性高的多元線性迴歸。',
    crispPrep: '數值特徵以中位數補值，類別特徵以眾數補值並做 one-hot encoding（drop first），形成候選特徵。資料切為 fit / validation / calibration / test。',
    crispDeployment: '本頁使用 <code>automobile_model_data.js</code> 匯出的係數與預測區間；亦即時預測器直接套用原始尺度係數與 one-hot 對應。'
  }
};

const COLORS = {
  'SelectKBest': '#60a5fa',
  'RFE': '#fb923c',
  'Sequential': '#fbbf24',
  'Lasso': '#f87171',
  'Permutation': '#2dd4bf'
};
const ICONS = {
  'SelectKBest': 'fa-filter',
  'RFE': 'fa-scissors',
  'Sequential': 'fa-arrow-trend-up',
  'Lasso': 'fa-compress',
  'Permutation': 'fa-shuffle'
};
const TAGS = {
  'SelectKBest': 'Filter',
  'RFE': 'Wrapper',
  'Sequential': 'Greedy',
  'Lasso': 'L1',
  'Permutation': 'Model-based'
};

let D = CONFIGS.flood.data;
let CFG = CONFIGS.flood;
let METHODS = D.methods;
const charts = {};
const state = {
  dataset: 'flood',
  method: null,
  k: null,
  values: {}
};

const nf = (v, d = 6) => Number(v).toFixed(d);
const pf = (v, d = 2) => (Number(v) * 100).toFixed(d) + '%';
const clamp01 = v => Math.max(0, Math.min(1, v));
const pretty = name => name.replace(/_x_/g, ' × ');

function fmtR2(v) {
  return Number(v).toFixed(4);
}

function fmtMetric(v) {
  if (CFG.currency) {
    return '$' + Math.round(Number(v)).toLocaleString('en-US');
  }
  return Number(v).toFixed(CFG.metricDigits);
}

function fmtTarget(v) {
  if (CFG.currency) {
    return '$' + Math.round(Number(v)).toLocaleString('en-US');
  }
  return Number(v).toFixed(CFG.targetDigits);
}

function fmtAxisMetric(v) {
  if (CFG.currency) {
    return '$' + Math.round(Number(v)).toLocaleString('en-US');
  }
  return Number(v).toFixed(4);
}

function fmtSigned(v, digits) {
  const sign = v >= 0 ? '+' : '−';
  const abs = Math.abs(v);
  if (CFG.currency) {
    return sign + '$' + Math.round(abs).toLocaleString('en-US');
  }
  return sign + abs.toFixed(digits);
}

function modelRow(method, k) {
  return D.models[method].find(row => row.k === k);
}

function selectionBest(rows) {
  const minRow = rows.reduce(
    (best, row) => row.validation.rmse < best.validation.rmse ? row : best,
    rows[0]
  );
  const se = Number.isFinite(minRow.validation.se) ? minRow.validation.se : 0;
  const threshold = minRow.validation.rmse + se;
  const candidates = rows.filter(row => row.validation.rmse <= threshold);
  return candidates.reduce((best, row) => {
    if (row.n_features < best.n_features) return row;
    if (row.n_features === best.n_features && row.validation.rmse < best.validation.rmse) return row;
    return best;
  }, candidates[0]);
}

function bestKFor(method) {
  return selectionBest(D.models[method]).k;
}

function bestMethodForK(k) {
  return METHODS.reduce((best, method) => {
    const row = modelRow(method, k);
    const bestRow = modelRow(best, k);
    return row.validation.rmse < bestRow.validation.rmse ? method : best;
  }, METHODS[0]);
}

function context() {
  if (state.method === 'ALL') {
    const method = state.k === D.meta.best_k
      ? D.meta.best_method
      : bestMethodForK(state.k);
    return { method, item: modelRow(method, state.k) };
  }
  return { method: state.method, item: modelRow(state.method, state.k) };
}

function displayFeatures(item) {
  return item.coefficients
    .filter(coef => coef.feature !== 'Intercept')
    .map(coef => coef.feature);
}

function coefficientSource(coef) {
  if (coef.source) return coef.source;
  if (coef.feature === 'Intercept') return null;
  if (coef.feature.includes('_x_')) {
    const [left, right] = coef.feature.split('_x_');
    return { type: 'interaction', left, right };
  }
  return { type: 'numeric', name: coef.feature };
}

function getNumericSchema() {
  if (D.inputs && D.inputs.numeric) return D.inputs.numeric;
  return D.baseFeatures.map(name => {
    const stats = D.featureStats[name];
    return {
      name,
      min: stats.min,
      max: stats.max,
      median: stats.median,
      q25: stats.min,
      q75: stats.max,
      step: 1
    };
  });
}

function getCategoricalSchema() {
  return (D.inputs && D.inputs.categorical) ? D.inputs.categorical : [];
}

function roundToStep(value, step) {
  const s = Number(step) || 1;
  const decimals = s < 0.1 ? 2 : (s < 1 ? 1 : 0);
  const rounded = Math.round(Number(value) / s) * s;
  return Number(rounded.toFixed(decimals));
}

function formatInputValue(value, step) {
  const s = Number(step) || 1;
  if (s >= 1) return Math.round(Number(value)).toLocaleString('en-US');
  return Number(value).toFixed(s < 0.1 ? 2 : 1);
}

function populateControls() {
  const tabs = document.getElementById('algoTabs');
  tabs.innerHTML = '';
  METHODS.forEach(method => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn';
    btn.dataset.method = method;
    btn.innerHTML = `<i class="fa-solid ${ICONS[method]}"></i> ${method} <span class="badge-pill">${TAGS[method]}</span>`;
    tabs.appendChild(btn);
  });
  const allBtn = document.createElement('button');
  allBtn.className = 'tab-btn tab-all';
  allBtn.dataset.method = 'ALL';
  allBtn.innerHTML = '<i class="fa-solid fa-layer-group"></i> 比較全部 5 種方法';
  tabs.appendChild(allBtn);

  const kGroup = document.getElementById('kSelectorGroup');
  kGroup.innerHTML = '';
  D.meta.feature_counts.forEach(k => {
    const btn = document.createElement('button');
    btn.className = 'k-btn';
    btn.dataset.k = k;
    btn.innerHTML = `k = ${k} <span style="font-weight:400;font-size:11px;">(${k} features)</span>`;
    kGroup.appendChild(btn);
  });
}

function buildInputs() {
  const grid = document.getElementById('inputGrid');
  grid.innerHTML = '';

  if (!state.values[state.dataset]) state.values[state.dataset] = {};
  const values = state.values[state.dataset];
  const numeric = getNumericSchema();
  const categorical = getCategoricalSchema();

  numeric.forEach((schema, index) => {
    if (values[schema.name] == null) values[schema.name] = schema.median;
    const step = schema.step || 1;
    const value = values[schema.name];
    const group = document.createElement('div');
    group.className = 'input-group';
    group.id = 'numGroup-' + index;
    group.innerHTML = `
      <div class="input-label-row">
        <span>${schema.name}</span>
        <span class="input-val" id="numVal-${index}">${formatInputValue(value, step)}</span>
      </div>
      <input type="range" class="range-slider" data-kind="numeric" data-index="${index}"
             data-feature="${schema.name}" min="${schema.min}" max="${schema.max}"
             step="${step}" value="${value}">
    `;
    grid.appendChild(group);
  });

  categorical.forEach((schema, index) => {
    if (values[schema.name] == null) values[schema.name] = schema.default;
    const value = values[schema.name];
    const group = document.createElement('div');
    group.className = 'input-group';
    group.id = 'catGroup-' + index;
    group.innerHTML = `
      <div class="input-label-row">
        <span>${schema.name}</span>
        <span class="input-val" style="color:var(--cyan);background:rgba(34,211,238,.10);">category</span>
      </div>
      <select class="input-field" data-kind="categorical" data-index="${index}"
              data-feature="${schema.name}" style="width:100%;padding:9px 12px;">
        ${schema.options.map(option => `<option value="${option}"${option === value ? ' selected' : ''}>${option}</option>`).join('')}
      </select>
    `;
    grid.appendChild(group);
  });
}

function applyScenario(scenario) {
  const numeric = getNumericSchema();
  const categorical = getCategoricalSchema();
  const statKey = CFG.scenarioStats[scenario] || 'median';
  const values = state.values[state.dataset];

  numeric.forEach((schema, index) => {
    const raw = schema[statKey] != null ? schema[statKey] : schema.median;
    const value = Math.max(schema.min, Math.min(schema.max, roundToStep(raw, schema.step || 1)));
    values[schema.name] = value;
    const slider = document.querySelector(`input[data-kind="numeric"][data-index="${index}"]`);
    if (slider) slider.value = value;
    const label = document.getElementById('numVal-' + index);
    if (label) label.textContent = formatInputValue(value, schema.step || 1);
  });

  categorical.forEach((schema, index) => {
    values[schema.name] = schema.default;
    const select = document.querySelector(`select[data-kind="categorical"][data-index="${index}"]`);
    if (select) select.value = schema.default;
  });

  calculatePrediction();
}

function updateHero() {
  const best = D.meta.best_validation;
  document.getElementById('headerDataset').innerHTML = `<i class="fa-solid fa-database"></i> ${CFG.headerDataset}`;
  document.getElementById('headerMethods').innerHTML = `<i class="fa-solid fa-layer-group"></i> ${METHODS.length} 方法 × ${D.meta.feature_counts.length} 特徵數`;
  document.getElementById('heroTitle').textContent = CFG.heroTitle;
  document.getElementById('heroDesc').innerHTML = CFG.heroDesc;
  document.getElementById('heroRows').textContent = D.meta.train_rows.toLocaleString('en-US');
  const featureText = CFG.key === 'flood'
    ? `${D.meta.n_base_features} 原始 + ${D.meta.n_candidate_features - D.meta.n_base_features} 互動特徵`
    : `${D.meta.n_base_features} 原始特徵 · one-hot 後 ${D.meta.n_candidate_features} 候選`;
  document.getElementById('heroFeaturesPill').textContent = featureText;
  document.getElementById('heroBest').textContent = `${best.method} · k=${best.k}`;
  document.getElementById('heroR2').textContent = fmtR2(best.test_R2);
  document.getElementById('footerStats').textContent =
    `${D.meta.train_rows.toLocaleString('en-US')} rows · ${D.meta.n_base_features} predictors · ${D.meta.n_candidate_features} candidate features · ${D.experiments.length} experiments`;
  document.title = CFG.heroTitle;
}

function updateCrisp() {
  const meta = D.meta;
  const testText = meta.test_rows > 0
    ? `${meta.test_rows.toLocaleString('en-US')} 筆外部測試集`
    : `${meta.split.test} 筆 hold-out 測試集`;
  const catCount = getCategoricalSchema().length;
  const numCount = meta.n_base_features - catCount;

  document.getElementById('crispBusinessTitle').textContent = CFG.crispBusinessTitle;
  document.getElementById('crispBusiness').innerHTML = CFG.crispBusiness;
  document.getElementById('crispData').innerHTML =
    `<b>${meta.train_rows.toLocaleString('en-US')}</b> 筆${CFG.dataNoun}、${meta.n_base_features} 個原始特徵（${numCount} 數值 / ${catCount} 類別），${testText}。${CFG.missingText}，重複列 <b>${meta.duplicate_rows.toLocaleString('en-US')}</b>，目標平均 ${CFG.targetMeanText}。`;
  document.getElementById('crispPrep').innerHTML = CFG.crispPrep;
  document.getElementById('crispModelTitle').textContent = `${METHODS.length} 種方法 × ${meta.feature_counts.length} 種 k`;
  document.getElementById('crispModel').innerHTML =
    `${METHODS.join('、')}，共 ${D.experiments.length} 組實驗。每組皆以 <code>LinearRegression</code> 配適。`;
  document.getElementById('crispEvaluation').innerHTML = '以 1-SE 規則選擇最佳設定：在 Validation RMSE 最小值的 1 個標準誤範圍內，選擇特徵最少且誤差最低的模型。再於 hold-out test 計算 RMSE、MAE、R²。95% 預測區間採用 split conformal，校準集不參與模型訓練。';
  document.getElementById('crispDeployment').innerHTML = CFG.crispDeployment;

  document.getElementById('splitFit').textContent = meta.split.fit.toLocaleString('en-US');
  document.getElementById('splitVal').textContent = meta.split.validation.toLocaleString('en-US');
  document.getElementById('splitCal').textContent = meta.split.calibration.toLocaleString('en-US');
  document.getElementById('splitTest').textContent = meta.split.test.toLocaleString('en-US');
  document.getElementById('targetChartSubtitle').textContent = CFG.targetChartSubtitle;
  document.getElementById('liveTitle').innerHTML = `<i class="fa-solid fa-calculator" style="color:var(--emerald);"></i> ${CFG.liveTitle}`;
  document.getElementById('liveSubtitle').textContent = CFG.liveSubtitle;
  document.getElementById('predLabel').textContent = CFG.liveLabel;

  const scenarioIcons = ['fa-arrow-down', 'fa-equals', 'fa-arrow-up'];
  document.querySelectorAll('.scenario-btn').forEach((btn, index) => {
    btn.innerHTML = `<i class="fa-solid ${scenarioIcons[index]}"></i> ${CFG.scenarioLabels[index]}`;
  });
}

function updateControlState() {
  document.querySelectorAll('#algoTabs .tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.method === state.method);
  });
  const bestK = state.method === 'ALL' ? D.meta.best_k : bestKFor(state.method);
  document.querySelectorAll('#kSelectorGroup .k-btn').forEach(btn => {
    const k = Number(btn.dataset.k);
    btn.classList.toggle('active', k === state.k);
    btn.classList.toggle('sweet-spot', k === bestK);
  });
}

function updateMetrics(item) {
  const bestK = state.method === 'ALL' ? D.meta.best_k : bestKFor(state.method);
  document.getElementById('valRMSE').textContent = fmtMetric(item.validation.rmse);
  document.getElementById('valR2').textContent = fmtR2(item.validation.r2);
  document.getElementById('testRMSE').textContent = fmtMetric(item.test.rmse);
  document.getElementById('testR2').textContent = fmtR2(item.test.r2);
  document.getElementById('testMAE').textContent = fmtMetric(item.test.mae);
  document.getElementById('coverage').textContent = pf(item.interval.coverage, 2);

  const badge = document.getElementById('badgeValRMSE');
  if (state.k === bestK) {
    badge.className = 'badge-tag badge-green';
    badge.innerHTML = '<i class="fa-solid fa-award"></i> 最佳甜蜜點';
  } else if (state.k > bestK) {
    badge.className = 'badge-tag badge-amber';
    badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> 較高維度';
  } else {
    badge.className = 'badge-tag badge-blue';
    badge.textContent = `k = ${state.k}`;
  }

  document.getElementById('cardValRMSE').className = 'metric-card ' + (state.k === bestK ? 'sweet' : '');
  document.getElementById('cardCoverage').className = 'metric-card ' + (item.interval.coverage >= 0.94 ? 'sweet' : 'alert');
  const coverageBadge = document.getElementById('badgeCoverage');
  coverageBadge.className = 'badge-tag ' + (item.interval.coverage >= 0.94 ? 'badge-green' : 'badge-amber');
  coverageBadge.textContent = item.interval.coverage >= 0.94 ? '接近名目水準' : '低於名目水準';

  document.getElementById('rmseChartBadge').textContent = `1-SE 最佳 k=${bestK}`;
  document.getElementById('r2ChartBadge').textContent = `1-SE 最佳 k=${bestK}`;
  document.getElementById('coefCountBadge').textContent = `${item.n_features} predictors`;
  document.getElementById('parityBadge').textContent = `R² ${fmtR2(item.test.r2)}`;
  document.getElementById('residualBadge').textContent = `RMSE ${fmtMetric(item.test.rmse)}`;
}

function updateFeatures(item) {
  const features = displayFeatures(item);
  document.getElementById('featuresCount').textContent = features.length;
  const chips = document.getElementById('featuresListChips');
  chips.innerHTML = '';
  features.forEach(feature => {
    const chip = document.createElement('span');
    chip.className = 'feature-chip' + (feature.includes('_x_') || feature.includes(' = ') ? ' engineered' : '');
    chip.textContent = pretty(feature);
    chips.appendChild(chip);
  });

  const bestK = state.method === 'ALL' ? D.meta.best_k : bestKFor(state.method);
  const notice = document.getElementById('parsimonyNotice');
  if (state.k === bestK) {
    notice.className = 'badge-tag badge-green';
    notice.innerHTML = `<i class="fa-solid fa-star"></i> ${state.method === 'ALL' ? '1-SE 最佳設定' : '此方法 1-SE 最佳 k'}：${item.k} 個特徵，Validation RMSE ${fmtMetric(item.validation.rmse)}。`;
  } else {
    notice.className = 'badge-tag badge-blue';
    notice.innerHTML = `<i class="fa-solid fa-circle-info"></i> 目前以 ${item.k} 個特徵評估；此方法 1-SE 最佳為 k=${bestK}。`;
  }
}

function formatCoef(coef) {
  const abs = Math.abs(coef);
  if (CFG.currency) {
    return Number(coef).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  if (abs < 0.001) return coef.toExponential(3);
  return coef.toFixed(6);
}

function formatCI(value) {
  if (CFG.currency) {
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  }
  return Number(value).toFixed(4);
}

function updateEquation(item) {
  const box = document.getElementById('equationBox');
  const terms = item.coefficients.map(coef => {
    if (coef.feature === 'Intercept') {
      return `<span class="eq-intercept">${formatCoef(coef.coef)}</span>`;
    }
    const sign = coef.coef >= 0 ? '<span class="eq-pos">+</span>' : '<span class="eq-neg">−</span>';
    return `${sign} ${formatCoef(Math.abs(coef.coef))} × <span class="eq-var">${pretty(coef.feature)}</span>`;
  });
  box.innerHTML = `<strong>${CFG.targetShort}</strong> = ${terms.join(' ')}`;
}

function updateCoefficientTable(item) {
  const tbody = document.getElementById('coefTableBody');
  tbody.innerHTML = '';
  item.coefficients.forEach(coef => {
    const tr = document.createElement('tr');
    const isIntercept = coef.feature === 'Intercept';
    let pBadge = '—';
    if (coef.pvalue != null) {
      pBadge = coef.pvalue < 0.05
        ? `<span class="badge-tag badge-green">${coef.pvalue < 0.0001 ? '< 0.0001' : coef.pvalue.toFixed(4)} <i class="fa-solid fa-check"></i></span>`
        : `<span class="badge-tag badge-amber">${coef.pvalue.toFixed(4)} (n.s.)</span>`;
    }
    let vif = '—';
    if (!isIntercept && coef.vif != null) {
      const cls = coef.vif < 5 ? 'badge-green' : coef.vif < 10 ? 'badge-amber' : 'badge-rose';
      const label = coef.vif < 5 ? 'Safe' : coef.vif < 10 ? 'Moderate' : 'Collinear';
      vif = `<span class="badge-tag ${cls}">${coef.vif.toFixed(2)} (${label})</span>`;
    }
    tr.innerHTML = `
      <td><strong>${pretty(coef.feature)}</strong></td>
      <td style="font-family:'Fira Code',monospace;font-weight:600;">${formatCoef(coef.coef)}</td>
      <td style="font-family:'Fira Code',monospace;">${coef.stderr != null ? (CFG.currency ? formatCI(coef.stderr) : coef.stderr.toExponential(2)) : '—'}</td>
      <td style="font-family:'Fira Code',monospace;">${coef.tstat != null ? coef.tstat.toFixed(2) : '—'}</td>
      <td>${pBadge}</td>
      <td style="font-size:11.5px;color:var(--text-muted);font-family:'Fira Code',monospace;">[${coef.ci_lower != null ? formatCI(coef.ci_lower) : '—'}, ${coef.ci_upper != null ? formatCI(coef.ci_upper) : '—'}]</td>
      <td>${vif}</td>`;
    tbody.appendChild(tr);
  });
}

function updateOutliers(item) {
  const tbody = document.getElementById('outliersTableBody');
  tbody.innerHTML = '';
  const threshold = 2 * item.test.rmse;
  item.outliers.forEach(outlier => {
    const high = outlier.abs_error > threshold;
    const status = high
      ? '<span class="badge-tag badge-rose"><span class="status-dot dot-amber"></span> 高殘差</span>'
      : '<span class="badge-tag badge-green"><span class="status-dot dot-green"></span> 範圍內</span>';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:700;">#${outlier.rank}</td>
      <td>Sample ${Number(outlier.id).toLocaleString('en-US')}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtTarget(outlier.actual)}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtTarget(outlier.predicted)}</td>
      <td style="color:${outlier.residual >= 0 ? '#2dd4bf' : '#fb7185'};font-weight:600;font-family:'Fira Code',monospace;">${fmtSigned(outlier.residual, CFG.targetDigits)}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtTarget(outlier.abs_error)}</td>
      <td>${outlier.pct_error != null ? outlier.pct_error.toFixed(2) + '%' : '—'}</td>
      <td>${status}</td>`;
    tbody.appendChild(tr);
  });
}

function updateInputAvailability(item) {
  const numeric = getNumericSchema();
  const categorical = getCategoricalSchema();
  const usedNumeric = new Set();
  const usedCategorical = new Set();

  item.coefficients.forEach(coef => {
    const src = coefficientSource(coef);
    if (!src) return;
    if (src.type === 'numeric') usedNumeric.add(src.name);
    if (src.type === 'dummy') usedCategorical.add(src.name);
    if (src.type === 'interaction') {
      usedNumeric.add(src.left);
      usedNumeric.add(src.right);
    }
  });

  numeric.forEach((schema, index) => {
    document.getElementById('numGroup-' + index).classList.toggle('disabled', !usedNumeric.has(schema.name));
  });
  categorical.forEach((schema, index) => {
    document.getElementById('catGroup-' + index).classList.toggle('disabled', !usedCategorical.has(schema.name));
  });
}

function calculatePrediction() {
  const { item } = context();
  const values = state.values[state.dataset];

  D.interactions.forEach(([left, right]) => {
    values[`${left}_x_${right}`] = Number(values[left]) * Number(values[right]);
  });

  let prediction = 0;
  const contributions = [];
  item.coefficients.forEach(coef => {
    const src = coefficientSource(coef);
    if (!src) {
      prediction += coef.coef;
      return;
    }
    let value = 0;
    if (src.type === 'numeric') value = Number(values[src.name]);
    else if (src.type === 'dummy') value = values[src.name] === src.value ? 1 : 0;
    else if (src.type === 'interaction') value = Number(values[src.left]) * Number(values[src.right]);
    const contribution = coef.coef * value;
    prediction += contribution;
    contributions.push({ feature: coef.feature, contribution });
  });

  const display = CFG.probability ? clamp01(prediction) : Math.max(0, prediction);
  const q = item.interval.q_hat;
  const lower = Number.isFinite(q) ? Math.max(CFG.probability ? 0 : 0, prediction - q) : null;
  const upper = Number.isFinite(q) ? (CFG.probability ? Math.min(1, prediction + q) : prediction + q) : null;

  const predValue = document.getElementById('predValue');
  predValue.textContent = fmtTarget(display);

  let risk = '';
  let color = '#2dd4bf';
  if (CFG.probability) {
    if (display >= 0.55) { risk = '高風險'; color = '#fb7185'; }
    else if (display >= 0.50) { risk = '中高風險'; color = '#fbbf24'; }
    else if (display >= 0.45) { risk = '中低風險'; color = '#8ea7ff'; }
    else { risk = '低風險'; color = '#2dd4bf'; }
    predValue.style.color = color;
    predValue.style.textShadow = `0 0 30px ${color}66`;
    document.getElementById('predPercent').textContent = `≈ ${(display * 100).toFixed(2)}% 洪水機率`;
    document.getElementById('riskMeterFill').style.width = `${(display * 100).toFixed(1)}%`;
    document.getElementById('riskMeterFill').parentElement.style.display = '';
  } else {
    predValue.style.color = '#2dd4bf';
    predValue.style.textShadow = '0 0 30px rgba(45,212,191,.4)';
    document.getElementById('predPercent').textContent = CFG.predSubtitle;
    document.getElementById('riskMeterFill').parentElement.style.display = 'none';
  }

  document.getElementById('predInterval').textContent = lower != null
    ? `95% PI [${fmtTarget(lower)}, ${fmtTarget(upper)}] · q̂ = ${fmtTarget(q)}`
    : '95% PI：校準資料不足';

  document.getElementById('predFormulaNote').innerHTML = CFG.probability
    ? `<span style="color:${color};font-weight:700;">${risk}</span> · 使用 ${item.n_features} 個特徵 · ${CFG.predSubtitle}`
    : `${CFG.predSubtitle} · 使用 ${item.n_features} 個特徵 · ${context().method} · k=${item.k}`;

  const top = contributions
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 6);

  const chips = document.getElementById('contributionChips');
  chips.innerHTML = '';
  top.forEach(c => {
    const chip = document.createElement('span');
    chip.className = 'contribution-chip ' + (c.contribution >= 0 ? 'pos' : 'neg');
    chip.textContent = `${pretty(c.feature)} ${fmtSigned(c.contribution, 5)}`;
    chips.appendChild(chip);
  });
}

function makeLineOptions(kind) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false, labels: { color: '#c7d3e7', usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
      tooltip: {
        backgroundColor: 'rgba(10,16,28,.96)',
        borderColor: 'rgba(125,150,205,.25)',
        borderWidth: 1,
        titleColor: '#f2f6ff',
        bodyColor: '#c7d3e7',
        padding: 12,
        callbacks: {
          label: ctx => {
            const value = kind === 'rmse' ? fmtMetric(ctx.parsed.y) : fmtR2(ctx.parsed.y);
            return `${ctx.dataset.label}: ${value}`;
          },
          afterLabel: ctx => {
            const method = state.method === 'ALL' ? ctx.dataset.label : state.method;
            const k = D.meta.feature_counts[ctx.dataIndex];
            const row = modelRow(method, k);
            if (!row) return '';
            const features = displayFeatures(row);
            return 'Features: ' + features.slice(0, 5).map(pretty).join(', ') + (features.length > 5 ? '…' : '');
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(125,150,205,.10)' },
        ticks: { color: '#95a6c4', font: { size: 11 } },
        title: { display: true, text: 'Number of features (k)', color: '#65758f', font: { size: 11 } }
      },
      y: {
        grid: { color: 'rgba(125,150,205,.10)' },
        ticks: { color: '#95a6c4', font: { size: 11 }, callback: v => kind === 'rmse' ? fmtAxisMetric(v) : Number(v).toFixed(2) },
        title: { display: true, text: kind === 'rmse' ? 'Validation RMSE' : 'Validation R²', color: '#65758f', font: { size: 11 } }
      }
    },
    onClick: (event, elements) => {
      if (!elements.length) return;
      state.k = D.meta.feature_counts[elements[0].index];
      render();
    }
  };
}

function initCharts() {
  Chart.defaults.color = '#95a6c4';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.borderColor = 'rgba(125,150,205,.10)';

  charts.rmse = new Chart(document.getElementById('rmseChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: makeLineOptions('rmse')
  });

  charts.r2 = new Chart(document.getElementById('r2Chart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: makeLineOptions('r2')
  });

  charts.interval = new Chart(document.getElementById('intervalChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#c7d3e7', usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(10,16,28,.96)',
          borderColor: 'rgba(125,150,205,.25)',
          borderWidth: 1,
          titleColor: '#f2f6ff',
          bodyColor: '#c7d3e7',
          padding: 12,
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmtTarget(ctx.parsed.y)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(125,150,205,.08)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, maxTicksLimit: 10 },
          title: { display: true, text: 'Test samples sorted by predicted value', color: '#65758f', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(125,150,205,.10)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => fmtAxisTarget(v) },
          title: { display: true, text: CFG ? CFG.targetDisplay : 'Target', color: '#65758f', font: { size: 11 } }
        }
      }
    }
  });

  charts.target = new Chart(document.getElementById('targetChart'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Count',
        data: [],
        backgroundColor: 'rgba(91,124,250,.55)',
        borderColor: 'rgba(139,163,255,.8)',
        borderWidth: 1,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10,16,28,.96)',
          borderColor: 'rgba(125,150,205,.25)',
          borderWidth: 1,
          callbacks: {
            title: items => `Bin ${items[0].label}`,
            label: ctx => `Count: ${Number(ctx.parsed.y).toLocaleString('en-US')}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#95a6c4', font: { size: 9 }, maxTicksLimit: 8 },
          title: { display: true, text: 'Target value', color: '#65758f', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(125,150,205,.10)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => Number(v).toLocaleString('en-US') },
          title: { display: true, text: 'Count', color: '#65758f', font: { size: 11 } }
        }
      }
    }
  });

  charts.parity = new Chart(document.getElementById('parityChart'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#c7d3e7', usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(10,16,28,.96)',
          borderColor: 'rgba(125,150,205,.25)',
          borderWidth: 1,
          callbacks: {
            label: ctx => `Actual ${fmtTarget(ctx.parsed.x)}, Predicted ${fmtTarget(ctx.parsed.y)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(125,150,205,.08)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => fmtAxisTarget(v) },
          title: { display: true, text: 'Actual', color: '#65758f', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(125,150,205,.08)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => fmtAxisTarget(v) },
          title: { display: true, text: 'Predicted', color: '#65758f', font: { size: 11 } }
        }
      }
    }
  });

  charts.residual = new Chart(document.getElementById('residualChart'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#c7d3e7', usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(10,16,28,.96)',
          borderColor: 'rgba(125,150,205,.25)',
          borderWidth: 1,
          callbacks: {
            label: ctx => `Predicted ${fmtTarget(ctx.parsed.x)}, Residual ${fmtSigned(ctx.parsed.y, CFG.targetDigits)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(125,150,205,.08)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => fmtAxisTarget(v) },
          title: { display: true, text: 'Predicted', color: '#65758f', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(125,150,205,.08)' },
          ticks: { color: '#95a6c4', font: { size: 10 }, callback: v => fmtAxisTarget(v) },
          title: { display: true, text: 'Residual', color: '#65758f', font: { size: 11 } }
        }
      }
    }
  });
}

function fmtAxisTarget(v) {
  if (CFG.currency) return '$' + Math.round(Number(v)).toLocaleString('en-US');
  return Number(v).toFixed(3);
}

function updateComparisonCharts() {
  const shown = state.method === 'ALL' ? METHODS : [state.method];
  const labels = D.meta.feature_counts.map(String);

  const makeDatasets = valueKey => shown.map(method => {
    const rows = D.models[method];
    const bestK = bestKFor(method);
    return {
      label: method,
      data: rows.map(row => valueKey === 'rmse' ? row.validation.rmse : row.validation.r2),
      borderColor: COLORS[method],
      backgroundColor: COLORS[method] + '22',
      tension: .3,
      borderWidth: 2.5,
      pointRadius: rows.map(row => row.k === state.k ? 7 : (row.k === bestK ? 6 : 4)),
      pointHoverRadius: 8,
      pointBackgroundColor: rows.map(row => row.k === bestK ? '#2dd4bf' : COLORS[method]),
      pointBorderColor: '#0a111e',
      pointBorderWidth: 2
    };
  });

  charts.rmse.data.labels = labels;
  charts.rmse.data.datasets = makeDatasets('rmse');
  charts.rmse.options.plugins.legend.display = state.method === 'ALL';
  charts.rmse.update();

  charts.r2.data.labels = labels;
  charts.r2.data.datasets = makeDatasets('r2');
  charts.r2.options.plugins.legend.display = state.method === 'ALL';
  charts.r2.update();
}

function updateTargetChart() {
  const hist = D.targetHistogram;
  const labels = hist.centers.map(value => CFG.currency
    ? '$' + Math.round(value).toLocaleString('en-US')
    : Number(value).toFixed(3));
  charts.target.data.labels = labels;
  charts.target.data.datasets[0].data = hist.counts;
  charts.target.options.scales.x.title.text = CFG.targetDisplay;
  charts.target.update();
}

function updateIntervalChart(item) {
  const scatter = item.scatter;
  charts.interval.data.labels = scatter.predicted.map((_, index) => index + 1);
  charts.interval.data.datasets = [
    {
      label: '95% 上界',
      data: scatter.upper,
      borderColor: 'rgba(45,212,191,.45)',
      backgroundColor: 'rgba(45,212,191,.10)',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0,
      fill: false
    },
    {
      label: '95% 下界',
      data: scatter.lower,
      borderColor: 'rgba(45,212,191,.45)',
      backgroundColor: 'rgba(45,212,191,.10)',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0,
      fill: '-1'
    },
    {
      label: '預測值',
      data: scatter.predicted,
      borderColor: '#8ea7ff',
      backgroundColor: 'rgba(142,167,255,.15)',
      borderWidth: 2,
      pointRadius: 0,
      tension: .2
    },
    {
      label: '實際值',
      data: scatter.actual,
      borderColor: 'rgba(251,113,133,.85)',
      backgroundColor: 'rgba(251,113,133,.65)',
      showLine: false,
      pointRadius: 2.5,
      pointHoverRadius: 4
    }
  ];
  charts.interval.options.scales.y.title.text = CFG.targetDisplay;
  charts.interval.update();

  document.getElementById('intervalStats').innerHTML = `
    <span class="interval-stat">q̂ <b>${fmtTarget(item.interval.q_hat)}</b></span>
    <span class="interval-stat">Coverage <b>${pf(item.interval.coverage, 2)}</b></span>
    <span class="interval-stat">Width <b>${fmtTarget(item.interval.width)}</b></span>
  `;
}

function updateDiagnosticCharts(item) {
  const scatter = item.scatter;
  const actual = scatter.actual;
  const predicted = scatter.predicted;
  const residual = scatter.residual;
  const minValue = Math.min(...actual, ...predicted);
  const maxValue = Math.max(...actual, ...predicted);
  const span = Math.max(maxValue - minValue, CFG.currency ? 1 : 0.001);
  const lo = minValue - span * 0.04;
  const hi = maxValue + span * 0.04;
  const maxAbsResidual = Math.max(...residual.map(v => Math.abs(v))) * 1.15 || span;

  charts.parity.data.datasets = [
    {
      label: '測試樣本',
      data: actual.map((value, index) => ({ x: value, y: predicted[index] })),
      backgroundColor: 'rgba(91,124,250,.42)',
      borderColor: 'rgba(91,124,250,.7)',
      pointRadius: 2.2,
      pointHoverRadius: 4
    },
    {
      label: '完美預測 y = x',
      type: 'line',
      data: [{ x: lo, y: lo }, { x: hi, y: hi }],
      borderColor: '#2dd4bf',
      borderDash: [6, 6],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false
    }
  ];
  charts.parity.options.scales.x.min = lo;
  charts.parity.options.scales.x.max = hi;
  charts.parity.options.scales.y.min = lo;
  charts.parity.options.scales.y.max = hi;
  charts.parity.update();

  charts.residual.data.datasets = [
    {
      label: '殘差',
      data: predicted.map((value, index) => ({ x: value, y: residual[index] })),
      backgroundColor: 'rgba(251,113,133,.38)',
      borderColor: 'rgba(251,113,133,.7)',
      pointRadius: 2.2,
      pointHoverRadius: 4
    },
    {
      label: '零線',
      type: 'line',
      data: [{ x: lo, y: 0 }, { x: hi, y: 0 }],
      borderColor: 'rgba(199,211,231,.55)',
      borderDash: [6, 6],
      borderWidth: 1.2,
      pointRadius: 0,
      fill: false
    }
  ];
  charts.residual.options.scales.x.min = lo;
  charts.residual.options.scales.x.max = hi;
  charts.residual.options.scales.y.min = -maxAbsResidual;
  charts.residual.options.scales.y.max = maxAbsResidual;
  charts.residual.update();
}

function buildExperimentTable() {
  const tbody = document.getElementById('experimentTableBody');
  tbody.innerHTML = '';
  D.experiments.forEach((row, index) => {
    const isBest = row.method === D.meta.best_method && row.k === D.meta.best_k;
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.dataset.method = row.method;
    tr.dataset.k = row.k;
    tr.innerHTML = `
      <td style="font-weight:700;">#${index + 1}</td>
      <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${COLORS[row.method]};margin-right:7px;"></span>${row.method}${isBest ? ' <span class="badge-tag badge-green"><i class="fa-solid fa-star"></i> Best</span>' : ''}</td>
      <td style="font-family:'Fira Code',monospace;">${row.k}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtMetric(row.validation_RMSE)}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtR2(row.validation_R2)}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtMetric(row.test_RMSE)}</td>
      <td style="font-family:'Fira Code',monospace;">${fmtR2(row.test_R2)}</td>
      <td>${pf(row.interval_coverage, 2)}</td>
      <td style="color:var(--text-muted);">${row.runtime_seconds.toFixed(1)}s</td>`;
    tr.addEventListener('click', () => {
      state.method = row.method;
      state.k = row.k;
      render();
    });
    tbody.appendChild(tr);
  });
  document.getElementById('experimentCountBadge').textContent = D.experiments.length + ' experiments';
}

function updateExperimentHighlight() {
  const ctx = context();
  document.querySelectorAll('#experimentTableBody tr').forEach(tr => {
    tr.classList.toggle(
      'current-row',
      tr.dataset.method === ctx.method && Number(tr.dataset.k) === state.k
    );
  });
}

function render() {
  const { item } = context();
  updateControlState();
  updateMetrics(item);
  updateFeatures(item);
  updateEquation(item);
  updateCoefficientTable(item);
  updateOutliers(item);
  updateInputAvailability(item);
  updateComparisonCharts();
  updateIntervalChart(item);
  updateDiagnosticCharts(item);
  updateExperimentHighlight();
  calculatePrediction();
}

function setDataset(key) {
  if (!CONFIGS[key]) key = 'flood';
  state.dataset = key;
  CFG = CONFIGS[key];
  D = CFG.data;
  METHODS = D.methods;
  state.method = D.meta.best_method;
  state.k = D.meta.best_k;
  if (!state.values[key]) state.values[key] = {};

  const select = document.getElementById('datasetSelect');
  if (select) select.value = key;

  populateControls();
  buildInputs();
  updateHero();
  updateCrisp();
  updateTargetChart();
  buildExperimentTable();
  render();
}

function initialDataset() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get('dataset') || window.location.hash.replace('#', '');
  return CONFIGS[requested] ? requested : 'flood';
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('algoTabs').addEventListener('click', event => {
    const btn = event.target.closest('.tab-btn');
    if (!btn) return;
    state.method = btn.dataset.method;
    render();
  });

  document.getElementById('kSelectorGroup').addEventListener('click', event => {
    const btn = event.target.closest('.k-btn');
    if (!btn) return;
    state.k = Number(btn.dataset.k);
    render();
  });

  document.getElementById('inputGrid').addEventListener('input', event => {
    const input = event.target;
    if (!input.matches('input[data-kind="numeric"]')) return;
    const index = Number(input.dataset.index);
    const schema = getNumericSchema()[index];
    const value = Number(input.value);
    state.values[state.dataset][schema.name] = value;
    document.getElementById('numVal-' + index).textContent = formatInputValue(value, schema.step || 1);
    calculatePrediction();
  });

  document.getElementById('inputGrid').addEventListener('change', event => {
    const select = event.target;
    if (!select.matches('select[data-kind="categorical"]')) return;
    const schema = getCategoricalSchema()[Number(select.dataset.index)];
    state.values[state.dataset][schema.name] = select.value;
    calculatePrediction();
  });

  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', () => applyScenario(btn.dataset.scenario));
  });

  document.getElementById('datasetSelect').addEventListener('change', event => {
    setDataset(event.target.value);
  });

  initCharts();
  setDataset(initialDataset());
});
