// Global Chart instances
let historicalChart = null;
let decompChart = null;
let lossChart = null;
let benchmarkChart = null;
let evalChart = null;
let forecastChart = null;
let activeEventSource = null;

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchStatusAndData();
});

function formatNumber(num) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  return new Intl.NumberFormat("id-ID").format(Math.round(num));
}

function formatDecimal(num, decimals = 2) {
  if (num === null || num === undefined || isNaN(num)) return "-";
  return new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(num);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.querySelectorAll(".nav-tab").forEach(el => {
    el.classList.remove("active-tab");
    el.classList.add("text-slate-600");
  });

  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetTab) targetTab.classList.remove("hidden");

  const btn = document.getElementById(`tabBtn-${tabId}`);
  if (btn) {
    btn.classList.add("active-tab");
    btn.classList.remove("text-slate-600");
  }

  setTimeout(() => {
    if (tabId === 'dashboard' && historicalChart) historicalChart.resize();
    if (tabId === 'eda' && decompChart) decompChart.resize();
    if (tabId === 'config' && lossChart) lossChart.resize();
    if (tabId === 'benchmark' && benchmarkChart) benchmarkChart.resize();
    if (tabId === 'evaluation' && evalChart) evalChart.resize();
    if (tabId === 'forecast' && forecastChart) forecastChart.resize();
  }, 100);
}

function initCharts() {
  // 1. Historical Chart
  const ctxHist = document.getElementById("historicalChart").getContext("2d");
  historicalChart = new Chart(ctxHist, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Konsumsi Listrik (kWh)",
        data: [],
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.08)",
        borderWidth: 2,
        fill: true,
        tension: 0.25,
        pointRadius: 2,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${formatNumber(ctx.raw)} kWh`
          }
        }
      },
      scales: {
        y: {
          ticks: { callback: (v) => formatNumber(v) },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 2. Decomposition Chart (Trend, Seasonal, Residual)
  const ctxDecomp = document.getElementById("decompChart").getContext("2d");
  decompChart = new Chart(ctxDecomp, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Data Asli",
          data: [],
          borderColor: "#94a3b8",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1
        },
        {
          label: "Tren (12-MA)",
          data: [],
          borderColor: "#2563eb",
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: "Komponen Musiman (kWh)",
          data: [],
          borderColor: "#10b981",
          borderDash: [3, 3],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${formatNumber(ctx.raw)} kWh`
          }
        }
      },
      scales: {
        y: {
          ticks: { callback: (v) => formatNumber(v) },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 3. Real-Time Loss Chart
  const ctxLoss = document.getElementById("lossChart").getContext("2d");
  lossChart = new Chart(ctxLoss, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "MSE Training Loss",
        data: [],
        borderColor: "#dc2626",
        backgroundColor: "rgba(220, 38, 38, 0.05)",
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: true, position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => `MSE: ${ctx.raw.toExponential(4)}`
          }
        }
      },
      scales: {
        y: { type: "linear", grid: { color: "#f1f5f9" } },
        x: { title: { display: true, text: "Epoch" }, grid: { display: false } }
      }
    }
  });

  // 4. Benchmark Chart
  const ctxBench = document.getElementById("benchmarkChart").getContext("2d");
  benchmarkChart = new Chart(ctxBench, {
    type: "line",
    data: {
      labels: [],
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toExponential(3)}`
          }
        }
      },
      scales: {
        y: { type: "linear", title: { display: true, text: "MSE Loss" }, grid: { color: "#f1f5f9" } },
        x: { title: { display: true, text: "Epoch" }, grid: { display: false } }
      }
    }
  });

  // 5. Evaluation Chart
  const ctxEval = document.getElementById("evalChart").getContext("2d");
  evalChart = new Chart(ctxEval, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Aktual",
          data: [],
          borderColor: "#2563eb",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        },
        {
          label: "Prediksi JST CG-BPR",
          data: [],
          borderColor: "#10b981",
          borderDash: [4, 4],
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${formatNumber(ctx.raw)} kWh`
          }
        }
      },
      scales: {
        y: { ticks: { callback: (v) => formatNumber(v) }, grid: { color: "#f1f5f9" } },
        x: { grid: { display: false } }
      }
    }
  });

  // 6. Forecast Chart with 95% Confidence Interval Band
  const ctxFore = document.getElementById("forecastChart").getContext("2d");
  forecastChart = new Chart(ctxFore, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Data Historis Aktual",
          data: [],
          borderColor: "#64748b",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2
        },
        {
          label: "Prediksi Konsumsi Listrik",
          data: [],
          borderColor: "#2563eb",
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: "#2563eb",
          tension: 0.2
        },
        {
          label: "Batas Atas (95% CI)",
          data: [],
          borderColor: "rgba(37, 99, 235, 0.2)",
          borderWidth: 1,
          pointRadius: 0,
          fill: "+1",
          backgroundColor: "rgba(37, 99, 235, 0.12)",
          tension: 0.2
        },
        {
          label: "Batas Bawah (95% CI)",
          data: [],
          borderColor: "rgba(37, 99, 235, 0.2)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${formatNumber(ctx.raw)} kWh`
          }
        }
      },
      scales: {
        y: { ticks: { callback: (v) => formatNumber(v) }, grid: { color: "#f1f5f9" } },
        x: { grid: { display: false } }
      }
    }
  });
}

async function fetchStatusAndData() {
  try {
    const res = await fetch("/api/status");
    const status = await res.json();
    if (status.has_data) {
      await loadHistoricalData();
      await loadEdaData();
    }
    if (status.has_evaluation) {
      await loadEvaluationData();
    }
    if (status.has_benchmark) {
      await loadBenchmarkData();
    }
  } catch (err) {
    console.error("Gagal memeriksa status:", err);
  }
}

async function loadHistoricalData() {
  try {
    const res = await fetch("/api/data");
    if (!res.ok) return;
    const json = await res.json();

    document.getElementById("statTotalKwh").innerText = `${formatNumber(json.stats.total)} kWh`;
    document.getElementById("statMeanKwh").innerText = `${formatNumber(json.stats.mean)} kWh`;
    document.getElementById("statMaxKwh").innerText = `${formatNumber(json.stats.max)} kWh`;
    document.getElementById("statCountData").innerText = `${json.stats.count} Bulan`;

    historicalChart.data.labels = json.periods;
    historicalChart.data.datasets[0].data = json.values;
    historicalChart.update();
  } catch (err) {
    console.error("Error loading historical data:", err);
  }
}

async function loadEdaData() {
  try {
    // 1. Stationarity ADF Test
    const resAdf = await fetch("/api/eda/stationarity");
    if (resAdf.ok) {
      const adf = await resAdf.json();
      document.getElementById("adfStatValue").innerText = adf.adf_statistic;
      document.getElementById("adfPValue").innerText = adf.p_value;
      document.getElementById("adfCrit5").innerText = adf.critical_values["5%"];
      document.getElementById("adfUsedLag").innerText = `${adf.used_lag} bulan`;

      const badge = document.getElementById("adfStatusBadge");
      const text = document.getElementById("adfConclusionText");
      if (adf.is_stationary) {
        badge.innerText = "Stasioner (Tolak H0)";
        badge.className = "px-3 py-1 text-xs font-bold rounded-full bg-emerald-100 text-emerald-800";
        text.innerHTML = `<b>Uji ADF:</b> Nilai ADF Stat (${adf.adf_statistic}) < Nilai Kritis 5% (${adf.critical_values["5%"]}). <b>Data stasioner</b>, pola variansi stabil.`;
      } else {
        badge.innerText = "Memiliki Tren / Non-Stasioner";
        badge.className = "px-3 py-1 text-xs font-bold rounded-full bg-amber-100 text-amber-800";
        text.innerHTML = `<b>Uji ADF:</b> Nilai ADF Stat (${adf.adf_statistic}) > Nilai Kritis 5% (${adf.critical_values["5%"]}). <b>Data memiliki tren jangka panjang</b>; jaringan saraf tiruan (JST) dengan normalisasi Min-Max mampu menangani pola non-linier ini secara adaptif.`;
      }
    }

    // 2. Seasonal Decomposition
    const resDecomp = await fetch("/api/eda/decomposition");
    if (resDecomp.ok) {
      const decomp = await resDecomp.json();
      decompChart.data.labels = decomp.periods;
      decompChart.data.datasets[0].data = decomp.actual;
      decompChart.data.datasets[1].data = decomp.decomposition.trend;
      decompChart.data.datasets[2].data = decomp.decomposition.seasonal;
      decompChart.update();
    }
  } catch (err) {
    console.error("Error loading EDA data:", err);
  }
}

async function loadSampleData() {
  try {
    const formData = new FormData();
    formData.append("use_sample", "true");
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    alert(data.message || "Data sampel berhasil dimuat!");
    await loadHistoricalData();
    await loadEdaData();
  } catch (err) {
    alert("Gagal memuat data sampel: " + err);
  }
}

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    document.getElementById("fileNameDisplay").innerText = `File Terpilih: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }
}

async function submitUpload() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Silakan pilih file CSV/Excel terlebih dahulu!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const btn = document.getElementById("btnUploadSubmit");
  btn.disabled = true;
  btn.innerText = "Mengunggah...";

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Gagal mengunggah file");
    alert(data.message || "File berhasil diunggah!");
    await loadHistoricalData();
    await loadEdaData();
  } catch (err) {
    alert("Terjadi kesalahan: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="check" class="w-4 h-4"></i> Unggah & Proses Data`;
    lucide.createIcons();
  }
}

async function startTraining() {
  const lag = parseInt(document.getElementById("paramLag").value);
  const hidden_layers = document.getElementById("paramHidden").value;
  const activation = document.getElementById("paramActivation").value;
  const optimizer = document.getElementById("paramOptimizer").value;
  const max_epochs = parseInt(document.getElementById("paramMaxEpochs").value);
  const target_error = parseFloat(document.getElementById("paramTargetError").value);
  const restart_interval = parseInt(document.getElementById("paramRestartInterval").value);
  const powell_threshold = parseFloat(document.getElementById("paramPowellThreshold").value);
  const test_size = parseFloat(document.getElementById("paramTestSize").value);

  const payload = {
    lag,
    hidden_layers,
    activation,
    optimizer,
    max_epochs,
    target_error,
    restart_interval,
    powell_threshold,
    test_size
  };

  const btn = document.getElementById("btnStartTrain");
  btn.disabled = true;
  btn.innerText = "Melatih Model...";

  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.update();

  let restartCount = 0;
  document.getElementById("liveRestartCount").innerText = "0 kali";

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || "Gagal memulai pelatihan");

    listenTrainingStream(max_epochs);
  } catch (err) {
    alert("Error: " + err.message);
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Mulai Pelatihan Model JST`;
    lucide.createIcons();
  }
}

function listenTrainingStream(maxEpochs) {
  if (activeEventSource) {
    activeEventSource.close();
  }

  const statusPill = document.getElementById("trainingStatusPill");
  const progressBar = document.getElementById("trainProgressBar");
  const progressLabel = document.getElementById("trainProgressLabel");
  const epochCount = document.getElementById("trainEpochCount");
  const liveLoss = document.getElementById("liveLossValue");
  const liveGrad = document.getElementById("liveGradNorm");
  const liveRestart = document.getElementById("liveRestartCount");

  statusPill.innerText = "Status: Sedang Melatih...";
  statusPill.className = "text-xs px-2.5 py-1 rounded-full font-medium bg-amber-100 text-amber-800 animate-pulse";

  let restarts = 0;
  activeEventSource = new EventSource("/api/train/stream");

  activeEventSource.onmessage = (event) => {
    try {
      if (event.data.startsWith(":")) return; // ignore comments / keep-alive pings

      const data = JSON.parse(event.data);

      if (data.done) {
        activeEventSource.close();
        activeEventSource = null;
        statusPill.innerText = "Status: Selesai!";
        statusPill.className = "text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-100 text-emerald-800";

        const btn = document.getElementById("btnStartTrain");
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Mulai Pelatihan Model JST`;
        lucide.createIcons();

        loadEvaluationData();
        return;
      }

      const epoch = data.epoch;
      const loss = data.loss;
      const gradNorm = data.gradient_norm;
      const progress = Math.min(100, (epoch / maxEpochs) * 100);

      progressBar.style.width = `${progress}%`;
      progressLabel.innerText = `Progress Pelatihan: ${progress.toFixed(1)}%`;
      epochCount.innerText = `Epoch: ${epoch} / ${maxEpochs}`;
      liveLoss.innerText = loss.toExponential(4);
      liveGrad.innerText = gradNorm.toFixed(5);

      if (data.is_restart && epoch > 0) {
        restarts += 1;
        liveRestart.innerText = `${restarts} kali (${data.restart_reason || 'Powell'})`;
      }

      lossChart.data.labels.push(epoch);
      lossChart.data.datasets[0].data.push(loss);
      lossChart.update();
    } catch (e) {
      console.error("SSE parse error:", e);
    }
  };

  activeEventSource.onerror = (err) => {
    console.warn("SSE stream closed/reconnecting:", err);
  };
}

async function runBenchmarkTest() {
  const btn = document.getElementById("btnRunBenchmark");
  btn.disabled = true;
  btn.innerText = "Menjalankan 3-Model Benchmark...";

  try {
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_epochs: 150 })
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || "Gagal menjalankan benchmark");

    renderBenchmarkResults(json.benchmark);
  } catch (err) {
    alert("Error Benchmark: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play-circle" class="w-4 h-4"></i> Jalankan Benchmark 3-Model`;
    lucide.createIcons();
  }
}

async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmark");
    if (res.ok) {
      const json = await res.json();
      renderBenchmarkResults(json);
    }
  } catch (e) {
    console.error(e);
  }
}

function renderBenchmarkResults(benchmark) {
  // Update Benchmark Chart
  benchmarkChart.data.labels = benchmark.epochs;
  benchmarkChart.data.datasets = Object.values(benchmark.loss_histories).map(item => ({
    label: item.name,
    data: item.losses,
    borderColor: item.color,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1
  }));
  benchmarkChart.update();

  // Update Table
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  benchmark.models.forEach(m => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50 transition border-b border-slate-100";
    tr.innerHTML = `
      <td class="px-4 py-2.5 font-bold text-slate-800 flex items-center gap-2">
        <span class="w-2.5 h-2.5 rounded-full inline-block" style="background-color: ${m.color}"></span> ${m.name}
      </td>
      <td class="px-4 py-2.5 text-right font-mono">${formatNumber(m.mse)}</td>
      <td class="px-4 py-2.5 text-right font-mono font-semibold">${formatNumber(m.rmse)}</td>
      <td class="px-4 py-2.5 text-right font-bold text-blue-600">${m.mape}%</td>
      <td class="px-4 py-2.5 text-right font-mono">${m.elapsed_time_sec}s</td>
      <td class="px-4 py-2.5 text-center font-semibold text-slate-700">${m.total_restarts}</td>
    `;
    tbody.appendChild(tr);
  });

  const conclBox = document.getElementById("academicConclusionBox");
  conclBox.classList.remove("hidden");
  document.getElementById("academicConclusionText").innerText = benchmark.academic_conclusion;
  lucide.createIcons();
}

async function loadEvaluationData() {
  try {
    const res = await fetch("/api/evaluation");
    if (!res.ok) return;
    const json = await res.json();

    const testMetrics = json.test_metrics;
    document.getElementById("evalMseValue").innerText = formatNumber(testMetrics.mse);
    document.getElementById("evalRmseValue").innerText = `${formatNumber(testMetrics.rmse)} kWh`;
    document.getElementById("evalMapeValue").innerText = `${formatDecimal(testMetrics.mape, 2)}%`;

    const badge = document.getElementById("evalMapeBadge");
    if (testMetrics.mape < 10) {
      badge.innerText = "Sangat Akurat (MAPE < 10%)";
      badge.className = "inline-block text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded mt-0.5";
    } else if (testMetrics.mape < 20) {
      badge.innerText = "Baik (10% - 20%)";
      badge.className = "inline-block text-[10px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded mt-0.5";
    } else {
      badge.innerText = "Cukup";
      badge.className = "inline-block text-[10px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded mt-0.5";
    }

    if (json.diagnostics && json.diagnostics.durbin_watson) {
      const dw = json.diagnostics.durbin_watson;
      document.getElementById("evalDwValue").innerText = dw.durbin_watson;
      const dwBadge = document.getElementById("evalDwBadge");
      dwBadge.innerText = dw.interpretation;
      dwBadge.className = dw.status === "Ideal"
        ? "inline-block text-[10px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded mt-0.5"
        : "inline-block text-[10px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded mt-0.5";
    }

    const allPeriods = [...json.train_periods, ...json.test_periods];
    const allActual = [...json.train_actual, ...json.test_actual];
    const allPredicted = [...json.train_predicted, ...json.test_predicted];

    evalChart.data.labels = allPeriods;
    evalChart.data.datasets[0].data = allActual;
    evalChart.data.datasets[1].data = allPredicted;
    evalChart.update();
  } catch (err) {
    console.error("Gagal memuat evaluasi:", err);
  }
}

async function executeForecast() {
  const steps = parseInt(document.getElementById("forecastSteps").value) || 12;
  const btn = document.getElementById("btnRunForecast");
  btn.disabled = true;
  btn.innerText = "Memproyeksikan...";

  try {
    const res = await fetch("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps })
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || "Gagal melakukan peramalan");

    renderForecastResults(json.forecast);
  } catch (err) {
    alert("Gagal melakukan peramalan: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="sparkles" class="w-4 h-4"></i> Ramalkan Sekarang`;
    lucide.createIcons();
  }
}

function renderForecastResults(predictions) {
  const histPeriods = historicalChart.data.labels;
  const histValues = historicalChart.data.datasets[0].data;

  const futurePeriods = predictions.map(p => p.Periode);
  const futureValues = predictions.map(p => p.Konsumsi_Prediksi_kWh);
  const lowerBounds = predictions.map(p => p.Batas_Bawah_95_kWh);
  const upperBounds = predictions.map(p => p.Batas_Atas_95_kWh);

  const combinedLabels = [...histPeriods, ...futurePeriods];

  const histSeries = [...histValues, ...Array(futureValues.length).fill(null)];

  const foreSeries = Array(histValues.length - 1).fill(null);
  foreSeries.push(histValues[histValues.length - 1]);
  futureValues.forEach(val => foreSeries.push(val));

  const upperSeries = Array(histValues.length - 1).fill(null);
  upperSeries.push(histValues[histValues.length - 1]);
  upperBounds.forEach(val => upperSeries.push(val));

  const lowerSeries = Array(histValues.length - 1).fill(null);
  lowerSeries.push(histValues[histValues.length - 1]);
  lowerBounds.forEach(val => lowerSeries.push(val));

  forecastChart.data.labels = combinedLabels;
  forecastChart.data.datasets[0].data = histSeries;
  forecastChart.data.datasets[1].data = foreSeries;
  forecastChart.data.datasets[2].data = upperSeries;
  forecastChart.data.datasets[3].data = lowerSeries;
  forecastChart.update();

  const tbody = document.getElementById("forecastTableBody");
  tbody.innerHTML = "";

  predictions.forEach((row, idx) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50 transition border-b border-slate-100";
    tr.innerHTML = `
      <td class="px-4 py-2.5 text-center font-medium text-slate-500">${idx + 1}</td>
      <td class="px-4 py-2.5 font-bold text-slate-800">${row.Periode}</td>
      <td class="px-4 py-2.5">${row.Tahun}</td>
      <td class="px-4 py-2.5">${row.Bulan}</td>
      <td class="px-4 py-2.5 text-right font-mono text-slate-500">${formatNumber(row.Batas_Bawah_95_kWh)} kWh</td>
      <td class="px-4 py-2.5 text-right font-bold text-blue-600">${formatNumber(row.Konsumsi_Prediksi_kWh)} kWh</td>
      <td class="px-4 py-2.5 text-right font-mono text-slate-500">${formatNumber(row.Batas_Atas_95_kWh)} kWh</td>
    `;
    tbody.appendChild(tr);
  });
}
