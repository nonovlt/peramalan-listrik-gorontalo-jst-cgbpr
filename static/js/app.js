// Global Chart instances
let historicalChart = null;
let lossChart = null;
let evalChart = null;
let forecastChart = null;
let activeEventSource = null;

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchStatusAndData();
});

// Format Angka ke format standar Indonesia
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

// Tab Switching
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

  // Trigger resize on charts when switching tabs
  setTimeout(() => {
    if (tabId === 'dashboard' && historicalChart) historicalChart.resize();
    if (tabId === 'config' && lossChart) lossChart.resize();
    if (tabId === 'evaluation' && evalChart) evalChart.resize();
    if (tabId === 'forecast' && forecastChart) forecastChart.resize();
  }, 100);
}

// Inisialisasi awal objek Chart.js
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
          ticks: {
            callback: (v) => formatNumber(v)
          },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 2. Real-Time Loss Chart
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
      animation: false, // Matikan animasi untuk performa streaming tinggi
      plugins: {
        legend: { display: true, position: 'top' },
        tooltip: {
          callbacks: {
            label: (ctx) => `MSE: ${ctx.raw.toExponential(4)}`
          }
        }
      },
      scales: {
        y: {
          type: "linear",
          grid: { color: "#f1f5f9" }
        },
        x: {
          title: { display: true, text: "Epoch" },
          grid: { display: false }
        }
      }
    }
  });

  // 3. Evaluation Chart (Actual vs Fitted)
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
          label: "Prediksi Model JST CG-BPR",
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
        y: {
          ticks: { callback: (v) => formatNumber(v) },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 4. Forecast Chart
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
          label: "Proyeksi Peramalan Masa Depan",
          data: [],
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.1)",
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: "#2563eb",
          fill: true,
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
        y: {
          ticks: { callback: (v) => formatNumber(v) },
          grid: { color: "#f1f5f9" }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

// Fetch Status dan Data saat halaman dibuka
async function fetchStatusAndData() {
  try {
    const res = await fetch("/api/status");
    const status = await res.json();
    if (status.has_data) {
      await loadHistoricalData();
    }
    if (status.has_evaluation) {
      await loadEvaluationData();
    }
  } catch (err) {
    console.error("Gagal memeriksa status sistem:", err);
  }
}

// Memuat data historis dan memperbarui chart + ringkasan statistik
async function loadHistoricalData() {
  try {
    const res = await fetch("/api/data");
    if (!res.ok) return;
    const json = await res.json();

    // Update Stats
    document.getElementById("statTotalKwh").innerText = `${formatNumber(json.stats.total)} kWh`;
    document.getElementById("statMeanKwh").innerText = `${formatNumber(json.stats.mean)} kWh`;
    document.getElementById("statMaxKwh").innerText = `${formatNumber(json.stats.max)} kWh`;
    document.getElementById("statCountData").innerText = `${json.stats.count} Bulan`;

    // Update Historical Chart
    historicalChart.data.labels = json.periods;
    historicalChart.data.datasets[0].data = json.values;
    historicalChart.update();
  } catch (err) {
    console.error("Error loading historical data:", err);
  }
}

// Muat sampel data bawaan Kota Gorontalo
async function loadSampleData() {
  try {
    const formData = new FormData();
    formData.append("use_sample", "true");
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    alert(data.message || "Data sampel berhasil dimuat!");
    await loadHistoricalData();
  } catch (err) {
    alert("Gagal memuat data sampel: " + err);
  }
}

// Event handler saat file dipilih di formulir upload
function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    document.getElementById("fileNameDisplay").innerText = `File Terpilih: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }
}

// Submit upload file
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
  } catch (err) {
    alert("Terjadi kesalahan: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="check" class="w-4 h-4"></i> Unggah & Proses Data`;
    lucide.createIcons();
  }
}

// Memulai Pelatihan Model JST dengan optimasi CG-BPR
async function startTraining() {
  const lag = parseInt(document.getElementById("paramLag").value);
  const hidden_layers = document.getElementById("paramHidden").value;
  const max_epochs = parseInt(document.getElementById("paramMaxEpochs").value);
  const target_error = parseFloat(document.getElementById("paramTargetError").value);
  const restart_interval = parseInt(document.getElementById("paramRestartInterval").value);
  const powell_threshold = parseFloat(document.getElementById("paramPowellThreshold").value);
  const test_size = parseFloat(document.getElementById("paramTestSize").value);

  const payload = {
    lag,
    hidden_layers,
    max_epochs,
    target_error,
    restart_interval,
    powell_threshold,
    test_size
  };

  const btn = document.getElementById("btnStartTrain");
  btn.disabled = true;
  btn.innerText = "Melatih Model...";

  // Reset visualizer
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

    // Buka Server-Sent Events (SSE) Stream
    listenTrainingStream(max_epochs);
  } catch (err) {
    alert("Error: " + err.message);
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Mulai Pelatihan Model JST`;
    lucide.createIcons();
  }
}

// SSE Listener untuk Real-time Training Loss Curve
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

        // Otomatis muat evaluasi
        loadEvaluationData();
        return;
      }

      // Update UI metrics
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

      // Append ke loss chart
      lossChart.data.labels.push(epoch);
      lossChart.data.datasets[0].data.push(loss);
      lossChart.update();
    } catch (e) {
      console.error("Error parsing SSE stream message:", e);
    }
  };

  activeEventSource.onerror = (err) => {
    console.warn("SSE stream closed or error:", err);
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    const btn = document.getElementById("btnStartTrain");
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Mulai Pelatihan Model JST`;
    lucide.createIcons();
    loadEvaluationData();
  };
}

// Memuat data hasil evaluasi model (Testing Set metrics & Actual vs Fitted)
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
      badge.innerText = "Akurasi Sangat Tinggi (MAPE < 10%)";
      badge.className = "inline-block text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded mt-0.5";
    } else if (testMetrics.mape < 20) {
      badge.innerText = "Akurasi Baik (10% - 20%)";
      badge.className = "inline-block text-[11px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded mt-0.5";
    } else {
      badge.innerText = "Cukup / Perlu Optimasi Parameter";
      badge.className = "inline-block text-[11px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded mt-0.5";
    }

    // Gabungkan urutan periode kronologis: Train dilanjutkan Test
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

// Eksekusi Mesin Peramalan n-bulan ke depan
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

// Render grafik peramalan dan tabel
function renderForecastResults(predictions) {
  // Ambil data historis terakhir untuk kesinambungan grafik
  const histPeriods = historicalChart.data.labels;
  const histValues = historicalChart.data.datasets[0].data;

  // Siapkan label gabungan (historis + masa depan)
  const futurePeriods = predictions.map(p => p.Periode);
  const futureValues = predictions.map(p => p.Konsumsi_Prediksi_kWh);

  const combinedLabels = [...histPeriods, ...futurePeriods];

  // Dataset Historis: isi nilai historis, sisanya null
  const histSeries = [...histValues, ...Array(futureValues.length).fill(null)];

  // Dataset Peramalan: titik awal menyambung dari nilai historis terakhir
  const foreSeries = Array(histValues.length - 1).fill(null);
  foreSeries.push(histValues[histValues.length - 1]); // Titik sambung
  futureValues.forEach(val => foreSeries.push(val));

  forecastChart.data.labels = combinedLabels;
  forecastChart.data.datasets[0].data = histSeries;
  forecastChart.data.datasets[1].data = foreSeries;
  forecastChart.update();

  // Update Tabel Hasil Peramalan
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
      <td class="px-4 py-2.5 text-right font-bold text-blue-600">${formatNumber(row.Konsumsi_Prediksi_kWh)} kWh</td>
    `;
    tbody.appendChild(tr);
  });
}
