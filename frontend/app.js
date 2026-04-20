/* ========================================================
   FraudShield AI — Frontend Logic
   ======================================================== */

const API = 'http://localhost:5000';

/* ── Particle background ── */
(function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  for (let i = 0; i < 90; i++) {
    particles.push({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.5 + 0.4,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.15,
      color: Math.random() > 0.5 ? '0,245,212' : '162,89,255',
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── Navbar scroll effect ── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  nav.style.boxShadow = window.scrollY > 30
    ? '0 4px 24px rgba(0,0,0,0.5)'
    : 'none';
});

/* ── Counter animation ── */
function animateCounter(el, target, duration = 1600, prefix = '', suffix = '') {
  const start = performance.now();
  const isFloat = String(target).includes('.');
  function update(ts) {
    const t = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val = isFloat
      ? (ease * target).toFixed(2)
      : Math.round(ease * target).toLocaleString();
    el.textContent = prefix + val + suffix;
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ── API Status Check ── */
async function checkAPI() {
  const badge = document.getElementById('api-status');
  try {
    const res = await fetch(`${API}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      badge.textContent = '● API Connected';
      badge.className = 'api-badge api-connected';
    } else throw new Error();
  } catch {
    badge.textContent = '● API Offline — Fallback Mode';
    badge.className = 'api-badge api-offline';
  }
}

/* ── Load & Render Dataset Stats ── */
async function loadStats() {
  let data;
  try {
    const res = await fetch(`${API}/api/stats`);
    data = await res.json();
  } catch {
    data = {
      total_transactions: 284807, fraud_count: 492, normal_count: 284315,
      fraud_percentage: 0.1727, normal_percentage: 99.8273,
      amount_mean: 88.35, fraud_amount_mean: 122.21,
    };
  }

  // Hero counters
  animateCounter(document.getElementById('stat-total'), data.total_transactions);
  animateCounter(document.getElementById('stat-fraud'), data.fraud_count);
  document.getElementById('stat-recall').textContent = '83%';
  document.getElementById('stat-auc').textContent = '0.99';

  // Overview cards
  document.getElementById('ov-total').textContent  = data.total_transactions.toLocaleString();
  document.getElementById('ov-fraud').textContent  = data.fraud_count.toLocaleString();
  document.getElementById('ov-normal').textContent = data.normal_count.toLocaleString();
  document.getElementById('ov-amount').textContent = `$${data.amount_mean.toFixed(2)}`;
  document.getElementById('ov-fraud-pct').textContent   = `${data.fraud_percentage}% of total`;
  document.getElementById('ov-normal-pct').textContent  = `${data.normal_percentage}% of total`;
  document.getElementById('ov-fraud-amount').textContent = `Fraud avg: $${data.fraud_amount_mean.toFixed(2)}`;
}

/* ── Chart defaults ── */
Chart.defaults.color = '#8899bb';
Chart.defaults.borderColor = 'rgba(0,245,212,0.08)';
Chart.defaults.font.family = 'Inter, sans-serif';

function glassPlugin() { return {}; }

/* ── EDA Charts ── */
async function loadEDA() {
  let data;
  try {
    const res = await fetch(`${API}/api/eda`);
    data = await res.json();
  } catch {
    data = {
      class_distribution: { labels: ['Normal (0)', 'Fraud (1)'], values: [284315, 492] },
      sampling_comparison: {
        before_smote:    { normal: 226602, fraud: 378 },
        after_smote:     { normal: 226602, fraud: 226602 },
        after_undersample: { normal: 378, fraud: 378 },
      },
      amount_by_class: {
        fraud:  { bins: ['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K+'], counts: [196, 74, 87, 73, 42, 20] },
        normal: { bins: ['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K+'], counts: [120000, 52000, 45000, 38000, 22000, 7315] },
      },
      pipeline_steps: [],
    };
  }

  // Class Distribution — Doughnut
  new Chart(document.getElementById('classDistChart'), {
    type: 'doughnut',
    data: {
      labels: data.class_distribution.labels,
      datasets: [{
        data: data.class_distribution.values,
        backgroundColor: ['rgba(0,245,212,0.7)', 'rgba(255,77,109,0.7)'],
        borderColor: ['#00f5d4', '#ff4d6d'],
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, font: { size: 13 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const val = ctx.raw;
              const total = ctx.chart.data.datasets[0].data.reduce((a,b) => a+b, 0);
              return ` ${val.toLocaleString()} (${(val/total*100).toFixed(2)}%)`;
            }
          }
        }
      },
      cutout: '70%',
    }
  });

  // SMOTE Comparison — Bar
  const { before_smote, after_smote, after_undersample } = data.sampling_comparison;
  new Chart(document.getElementById('smoteChart'), {
    type: 'bar',
    data: {
      labels: ['Before SMOTE', 'After SMOTE', 'Undersampled'],
      datasets: [
        {
          label: 'Normal',
          data: [before_smote.normal, after_smote.normal, after_undersample.normal],
          backgroundColor: 'rgba(0,245,212,0.6)', borderColor: '#00f5d4', borderWidth: 1,
        },
        {
          label: 'Fraud',
          data: [before_smote.fraud, after_smote.fraud, after_undersample.fraud],
          backgroundColor: 'rgba(255,77,109,0.6)', borderColor: '#ff4d6d', borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v >= 1000 ? (v/1000).toFixed(0)+'K' : v } },
      },
      plugins: { legend: { position: 'top' } },
    }
  });

  // Amount by class — Bar
  new Chart(document.getElementById('amountChart'), {
    type: 'bar',
    data: {
      labels: data.amount_by_class.fraud.bins,
      datasets: [
        {
          label: 'Fraud',
          data: data.amount_by_class.fraud.counts,
          backgroundColor: 'rgba(255,77,109,0.65)', borderColor: '#ff4d6d', borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Amount Range' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Count' } },
      },
      plugins: { legend: { position: 'top' } },
    }
  });

  // Amount stats — Grouped bar
  new Chart(document.getElementById('amountStatsChart'), {
    type: 'bar',
    data: {
      labels: ['Mean Amount', 'Median Amount'],
      datasets: [
        {
          label: 'Normal',
          data: [88.29, 22.0],
          backgroundColor: 'rgba(0,245,212,0.6)', borderColor: '#00f5d4', borderWidth: 1,
        },
        {
          label: 'Fraud',
          data: [122.21, 9.25],
          backgroundColor: 'rgba(255,77,109,0.6)', borderColor: '#ff4d6d', borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => '$' + v } },
      },
      plugins: { legend: { position: 'top' } },
    }
  });

  // Build pipeline
  buildPipeline(data.pipeline_steps);
}

/* ── Pipeline Steps ── */
function buildPipeline(steps) {
  const fallback = [
    { step: 1,  name: 'Data Loading',          desc: 'Load creditcard.csv (284K rows)' },
    { step: 2,  name: 'Data Cleaning',          desc: 'Remove 1,081 duplicate rows' },
    { step: 3,  name: 'EDA',                    desc: 'Analyze class imbalance & distributions' },
    { step: 4,  name: 'Feature Scaling',         desc: 'StandardScaler on Amount & Time' },
    { step: 5,  name: 'SMOTE Oversampling',      desc: 'Balance dataset 378 → 226K fraud rows' },
    { step: 6,  name: 'Model Training',          desc: 'LR, Decision Tree, Random Forest' },
    { step: 7,  name: 'Hyperparameter Tuning',   desc: 'GridSearchCV — max_depth=10, n=100' },
    { step: 8,  name: 'Threshold Tuning',        desc: 'Threshold 0.4 boosts recall to 83%' },
    { step: 9,  name: 'Anomaly Detection',       desc: 'Isolation Forest & LOF applied' },
    { step: 10, name: 'Evaluation',              desc: 'Precision, Recall, F1, ROC-AUC compared' },
  ];
  const list = (steps && steps.length) ? steps : fallback;
  const container = document.getElementById('pipeline-steps');
  container.innerHTML = '';

  list.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'pipeline-step';
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML = `
      <div class="pipeline-step-num">STEP ${String(step.step).padStart(2, '0')}</div>
      <div class="pipeline-step-name">${step.name}</div>
      <div class="pipeline-step-desc">${step.desc}</div>
    `;
    container.appendChild(el);

    if (i < list.length - 1) {
      const arrow = document.createElement('div');
      arrow.className = 'pipeline-arrow';
      arrow.innerHTML = '→';
      container.appendChild(arrow);
    }
  });
}

/* ── Model Metrics ── */
async function loadMetrics() {
  let data;
  try {
    const res = await fetch(`${API}/api/model-metrics`);
    data = await res.json();
  } catch {
    data = [
      { name: 'Logistic Regression',  precision: 0.08, recall: 0.91, f1_score: 0.15, accuracy: 0.97,  roc_auc: 0.97,  type: 'supervised' },
      { name: 'Decision Tree',        precision: 0.12, recall: 0.82, f1_score: 0.21, accuracy: 0.98,  roc_auc: 0.90,  type: 'supervised' },
      { name: 'Random Forest (Tuned)',precision: 0.35, recall: 0.83, f1_score: 0.49, accuracy: 1.00,  roc_auc: 0.99,  type: 'supervised' },
      { name: 'Isolation Forest',     precision: 0.04, recall: 0.27, f1_score: 0.07, accuracy: 0.95,  roc_auc: null,  type: 'anomaly' },
    ];
  }

  // Find best by recall
  const bestIdx = data.reduce((best, m, i, arr) =>
    m.recall > arr[best].recall ? i : best, 0);

  // Recall bar chart
  const colors = data.map((m, i) =>
    i === bestIdx ? '#00f5d4' : 'rgba(0,245,212,0.35)');
  new Chart(document.getElementById('recallChart'), {
    type: 'bar',
    data: {
      labels: data.map(m => m.name),
      datasets: [{
        label: 'Recall (Fraud)',
        data: data.map(m => m.recall),
        backgroundColor: colors,
        borderColor: colors,
        borderWidth: 1, borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.04)' },
             ticks: { callback: v => (v*100).toFixed(0)+'%' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' } },
      },
      plugins: { legend: { display: false } },
    }
  });

  // Metrics Table
  const tbody = document.getElementById('metrics-tbody');
  const colClass = val => {
    if (val >= 0.7) return 'metric-val-good';
    if (val >= 0.4) return 'metric-val-mid';
    return 'metric-val-low';
  };
  tbody.innerHTML = data.map((m, i) => `
    <tr class="${i === bestIdx ? 'best-row' : ''}">
      <td>${i === bestIdx ? '★ ' : ''}${m.name}</td>
      <td class="${colClass(m.precision)}">${(m.precision*100).toFixed(1)}%</td>
      <td class="${colClass(m.recall)}">${(m.recall*100).toFixed(1)}%</td>
      <td class="${colClass(m.f1_score)}">${(m.f1_score*100).toFixed(1)}%</td>
      <td class="${colClass(m.accuracy)}">${(m.accuracy*100).toFixed(1)}%</td>
      <td class="${colClass(m.roc_auc || 0)}">${m.roc_auc !== null ? m.roc_auc.toFixed(2) : 'N/A'}</td>
      <td><span class="badge-type badge-${m.type || 'supervised'}">${m.type === 'anomaly' ? 'Anomaly' : 'Supervised'}</span></td>
    </tr>
  `).join('');
}

/* ── Build V input grids ── */
function buildVInputs() {
  const g1 = document.getElementById('v-grid-1');
  const g2 = document.getElementById('v-grid-2');
  for (let i = 1; i <= 14; i++) {
    g1.appendChild(makeVInput(i));
  }
  for (let i = 15; i <= 28; i++) {
    g2.appendChild(makeVInput(i));
  }
}

function makeVInput(i) {
  const wrap = document.createElement('div');
  wrap.className = 'v-input-wrap';
  wrap.innerHTML = `
    <label for="inp-v${i}">V${i}</label>
    <input type="number" id="inp-v${i}" data-v="${i}" placeholder="0" step="0.0001" value="0" />
  `;
  return wrap;
}

/* ── Preset values ── */
const PRESETS = {
  normal: {
    time: 406, amount: 149.62,
    V: { 1: -1.36, 2: -0.07, 3: 2.54, 4: 1.38, 5: -0.34, 6: 0.46, 7: 0.24,
         8: 0.10,  9: 0.36, 10: 0.09, 11:-0.55, 12:-0.62, 13:-0.99, 14:-0.31,
        15: 1.47, 16:-0.47, 17: 0.21, 18: 0.03, 19: 0.40, 20: 0.25,
        21:-0.02, 22: 0.28, 23:-0.11, 24: 0.07, 25: 0.13, 26:-0.19, 27: 0.13, 28:-0.02 },
  },
  fraud: {
    time: 406, amount: 1.0,
    V: { 1:-3.04, 2: 1.96, 3:-3.17, 4: 0.85, 5:-1.15, 6: 0.66, 7:-1.56,
         8: 0.43,  9:-3.14, 10:-0.76, 11: 1.79, 12:-0.50, 13:-0.45, 14:-0.45,
        15:-0.57, 16: 1.73, 17:-0.46, 18:-0.16, 19: 0.17, 20:-0.03,
        21:-0.47, 22: 0.21, 23: 0.22, 24: 0.05, 25: 0.08, 26:-0.24, 27: 0.11, 28: 0.04 },
  },
};

window.fillPreset = function(type) {
  if (type === 'clear') {
    document.getElementById('inp-time').value = 0;
    document.getElementById('inp-amount').value = '';
    for (let i = 1; i <= 28; i++) {
      const el = document.getElementById(`inp-v${i}`);
      if (el) el.value = 0;
    }
    return;
  }
  const p = PRESETS[type];
  document.getElementById('inp-time').value = p.time;
  document.getElementById('inp-amount').value = p.amount;
  for (let i = 1; i <= 28; i++) {
    const el = document.getElementById(`inp-v${i}`);
    if (el) el.value = p.V[i] !== undefined ? p.V[i] : 0;
  }
};

/* ── Gauge Chart ── */
let gaugeChart = null;
function drawGauge(prob) {
  const canvas = document.getElementById('gaugeChart');
  if (gaugeChart) { gaugeChart.destroy(); gaugeChart = null; }

  const color = prob < 0.4 ? '#06d6a0' : prob < 0.7 ? '#f4a261' : '#ff4d6d';
  gaugeChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [prob, 1 - prob],
        backgroundColor: [color, 'rgba(255,255,255,0.05)'],
        borderColor: [color, 'transparent'],
        borderWidth: 2,
      }],
    },
    options: {
      circumference: 180,
      rotation: -90,
      cutout: '72%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 800, easing: 'easeOutCubic' },
    },
  });
}

/* ── Predict Form ── */
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn     = document.getElementById('predict-btn');
  const btnText = document.getElementById('predict-btn-text');
  const spinner = document.getElementById('predict-spinner');

  btn.disabled = true;
  btnText.textContent = 'Analyzing...';
  spinner.classList.remove('hidden');

  const placeholder = document.getElementById('result-placeholder');
  const resultContent = document.getElementById('result-content');
  const resultError   = document.getElementById('result-error');

  placeholder.classList.add('hidden');
  resultContent.classList.add('hidden');
  resultError.classList.add('hidden');

  const payload = {
    Time:   parseFloat(document.getElementById('inp-time').value) || 0,
    Amount: parseFloat(document.getElementById('inp-amount').value) || 0,
  };
  for (let i = 1; i <= 28; i++) {
    const el = document.getElementById(`inp-v${i}`);
    payload[`V${i}`] = parseFloat(el ? el.value : 0) || 0;
  }

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(8000),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || 'API error');
    }

    const prob = data.fraud_probability;
    const isFraud = data.is_fraud;
    const risk = data.risk_level || (prob >= 0.7 ? 'HIGH' : prob >= 0.4 ? 'MEDIUM' : 'LOW');
    const riskColor = { HIGH: '#ff4d6d', MEDIUM: '#f4a261', LOW: '#06d6a0' }[risk];

    // Update gauge
    drawGauge(prob);
    document.getElementById('result-prob').textContent = (prob * 100).toFixed(1) + '%';
    document.getElementById('result-risk').textContent  = risk + ' RISK';
    document.getElementById('result-risk').style.color   = riskColor;

    // Verdict
    const verdict     = document.getElementById('result-verdict');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictText = document.getElementById('verdict-text');
    verdict.className = 'result-verdict ' + (isFraud ? 'fraud' : 'safe');
    verdictIcon.textContent = isFraud ? '🚨' : '✅';
    verdictText.textContent = isFraud ? 'FRAUD DETECTED — Flag this transaction' : 'Transaction appears legitimate';

    // Details
    document.getElementById('detail-prob').textContent = (prob * 100).toFixed(2) + '%';
    document.getElementById('detail-prob').style.color  = riskColor;
    document.getElementById('detail-risk').textContent  = risk;
    document.getElementById('detail-risk').style.color  = riskColor;

    resultContent.classList.remove('hidden');

  } catch (err) {
    document.getElementById('result-error-msg').textContent =
      err.message.includes('Failed to fetch') || err.name === 'TimeoutError'
        ? '⚠️ Cannot reach backend. Start Flask: python backend/app.py'
        : err.message;
    resultError.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btnText.textContent = '🔍 Analyze Transaction';
    spinner.classList.add('hidden');
  }
});

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  buildVInputs();
  checkAPI();
  loadStats();
  loadEDA();
  loadMetrics();
});
