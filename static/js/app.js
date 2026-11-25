(() => {
  "use strict";

  const DEBOUNCE_MS = 300;

  const state = {
    datasetId: null,
    filename: "",
    columns: [],
    numericColumns: [],
    hasGroundTruth: false,
    rowCount: 0,
    selectedFeatures: [],
    xAxis: "",
    yAxis: "",
    contamination: 0.02,
    contaminationAuto: false,
    nEstimators: 100,
    missingStrategy: "impute",
    defaultContamination: 0.05,
    detectionData: null,
    sortKey: "anomaly_score",
    sortDir: "desc",
    examples: [],
    debounceTimer: null,
    activeController: null,
  };

  const dom = {
    exampleSelect: document.getElementById("exampleSelect"),
    exampleDescription: document.getElementById("exampleDescription"),
    csvFileInput: document.getElementById("csvFileInput"),
    chooseFileBtn: document.getElementById("chooseFileBtn"),
    dropZone: document.getElementById("dropZone"),
    datasetMeta: document.getElementById("datasetMeta"),
    errorBanner: document.getElementById("errorBanner"),
    warningBanner: document.getElementById("warningBanner"),
    featureCheckboxes: document.getElementById("featureCheckboxes"),
    selectAllFeaturesBtn: document.getElementById("selectAllFeaturesBtn"),
    xAxisSelect: document.getElementById("xAxisSelect"),
    yAxisSelect: document.getElementById("yAxisSelect"),
    contaminationSlider: document.getElementById("contaminationSlider"),
    contaminationValue: document.getElementById("contaminationValue"),
    contaminationAuto: document.getElementById("contaminationAuto"),
    estimatorsSlider: document.getElementById("estimatorsSlider"),
    estimatorsValue: document.getElementById("estimatorsValue"),
    missingStrategySelect: document.getElementById("missingStrategySelect"),
    missingStatusText: document.getElementById("missingStatusText"),
    resetBtn: document.getElementById("resetBtn"),
    downloadCsvTopBtn: document.getElementById("downloadCsvTopBtn"),
    downloadCsvTableBtn: document.getElementById("downloadCsvTableBtn"),
    loadingIndicator: document.getElementById("loadingIndicator"),
    projectionNote: document.getElementById("projectionNote"),
    statTotalRows: document.getElementById("statTotalRows"),
    statMissingNote: document.getElementById("statMissingNote"),
    statAnomalyCount: document.getElementById("statAnomalyCount"),
    statAnomalyPct: document.getElementById("statAnomalyPct"),
    statThreshold: document.getElementById("statThreshold"),
    statContaminationMode: document.getElementById("statContaminationMode"),
    statFourthLabel: document.getElementById("statFourthLabel"),
    statFourthValue: document.getElementById("statFourthValue"),
    statFourthSub: document.getElementById("statFourthSub"),
    scatterPlot: document.getElementById("scatterPlot"),
    scoreHistogram: document.getElementById("scoreHistogram"),
    anomaliesTableHead: document.getElementById("anomaliesTableHead"),
    anomaliesTableBody: document.getElementById("anomaliesTableBody"),
    tableSubtitle: document.getElementById("tableSubtitle"),
  };

  function showError(message) {
    if (!message) {
      dom.errorBanner.classList.add("hidden");
      dom.errorBanner.textContent = "";
      return;
    }
    dom.errorBanner.textContent = message;
    dom.errorBanner.classList.remove("hidden");
  }

  function showWarning(message) {
    if (!message) {
      dom.warningBanner.classList.add("hidden");
      dom.warningBanner.textContent = "";
      return;
    }
    dom.warningBanner.textContent = message;
    dom.warningBanner.classList.remove("hidden");
  }

  function setLoading(isLoading) {
    if (isLoading) {
      dom.loadingIndicator.classList.remove("hidden");
    } else {
      dom.loadingIndicator.classList.add("hidden");
    }
  }

  function formatNumber(val) {
    if (val === null || val === undefined || Number.isNaN(val)) {
      return "—";
    }
    if (typeof val === "number") {
      if (Number.isInteger(val)) {
        return val.toLocaleString();
      }
      return Math.abs(val) >= 100
        ? val.toLocaleString(undefined, { maximumFractionDigits: 2 })
        : val.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return String(val);
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function updateParameterLabels() {
    if (state.contaminationAuto) {
      dom.contaminationValue.textContent = "Auto";
      dom.contaminationSlider.disabled = true;
    } else {
      dom.contaminationSlider.disabled = false;
      const pct = (state.contamination * 100).toFixed(1);
      dom.contaminationValue.textContent = `${state.contamination.toFixed(3)} (${pct}%)`;
    }
    dom.estimatorsValue.textContent = `${state.nEstimators}`;
  }

  function renderFeatureCheckboxes() {
    dom.featureCheckboxes.textContent = "";
    state.numericColumns.forEach((col) => {
      const label = document.createElement("label");
      label.className = "checkbox-item";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = col;
      checkbox.checked = state.selectedFeatures.includes(col);

      checkbox.addEventListener("change", () => {
        const checked = Array.from(
          dom.featureCheckboxes.querySelectorAll("input[type='checkbox']:checked")
        ).map((el) => el.value);

        if (checked.length < 2) {
          checkbox.checked = true;
          showWarning("Select at least 2 numeric columns for Isolation Forest.");
          return;
        }
        showWarning("");
        state.selectedFeatures = checked;
        scheduleDetect(0);
      });

      const span = document.createElement("span");
      span.textContent = col;

      label.appendChild(checkbox);
      label.appendChild(span);
      dom.featureCheckboxes.appendChild(label);
    });
  }

  function renderAxisSelectors() {
    [dom.xAxisSelect, dom.yAxisSelect].forEach((selectEl) => {
      selectEl.textContent = "";
      state.numericColumns.forEach((col) => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        selectEl.appendChild(opt);
      });
    });

    if (state.numericColumns.length >= 2) {
      if (!state.numericColumns.includes(state.xAxis)) {
        state.xAxis = state.numericColumns[0];
      }
      if (!state.numericColumns.includes(state.yAxis) || state.yAxis === state.xAxis) {
        state.yAxis = state.numericColumns.find((c) => c !== state.xAxis) || state.numericColumns[1];
      }
    } else if (state.numericColumns.length === 1) {
      state.xAxis = state.numericColumns[0];
      state.yAxis = state.numericColumns[0];
    }

    dom.xAxisSelect.value = state.xAxis;
    dom.yAxisSelect.value = state.yAxis;
  }

  function applyLoadedDataset(payload, isExample) {
    showError("");
    state.datasetId = payload.dataset_id;
    state.filename = payload.filename;
    state.columns = payload.columns || [];
    state.numericColumns = payload.numeric_columns || [];
    state.hasGroundTruth = Boolean(payload.has_ground_truth);
    state.rowCount = payload.row_count || 0;

    state.selectedFeatures = [...state.numericColumns];
    state.xAxis = state.numericColumns[0] || "";
    state.yAxis = state.numericColumns[1] || state.numericColumns[0] || "";

    const defaultCont = payload.default_contamination
      ? parseFloat(payload.default_contamination)
      : 0.05;
    state.defaultContamination = Number.isFinite(defaultCont) ? defaultCont : 0.05;
    state.contamination = state.defaultContamination;
    state.contaminationAuto = false;
    state.nEstimators = 100;
    state.missingStrategy = "impute";

    dom.contaminationSlider.value = String(state.contamination);
    dom.contaminationAuto.checked = false;
    dom.estimatorsSlider.value = String(state.nEstimators);
    dom.missingStrategySelect.value = state.missingStrategy;
    updateParameterLabels();

    const totalMissing = Object.values(payload.missing_counts || {}).reduce(
      (sum, count) => sum + Number(count || 0),
      0
    );
    const missingNote = totalMissing > 0 ? ` · ${totalMissing} missing cells` : "";
    dom.datasetMeta.textContent = `${state.filename} · ${state.rowCount.toLocaleString()} rows · ${state.numericColumns.length} numeric features${missingNote}`;

    if (isExample && payload.description) {
      dom.exampleDescription.textContent = payload.description;
    } else {
      dom.exampleDescription.textContent =
        "Uploaded dataset active in memory. Switch back to any example anytime.";
      dom.exampleSelect.value = "";
    }

    if (Array.isArray(payload.warnings) && payload.warnings.length > 0) {
      showWarning(payload.warnings.join(" "));
    } else {
      showWarning("");
    }

    renderFeatureCheckboxes();
    renderAxisSelectors();

    if (state.numericColumns.length >= 2) {
      scheduleDetect(0);
    } else {
      dom.downloadCsvTopBtn.disabled = true;
      dom.downloadCsvTableBtn.disabled = true;
      showError("At least 2 numeric columns are required to run Isolation Forest.");
    }
  }

  function scheduleDetect(delayMs = DEBOUNCE_MS) {
    if (state.debounceTimer) {
      clearTimeout(state.debounceTimer);
    }
    state.debounceTimer = setTimeout(() => {
      runDetection();
    }, delayMs);
  }

  async function runDetection() {
    if (!state.datasetId || state.selectedFeatures.length < 2) {
      return;
    }

    if (state.activeController) {
      state.activeController.abort();
    }
    const controller = new AbortController();
    state.activeController = controller;

    setLoading(true);
    showError("");

    const body = {
      dataset_id: state.datasetId,
      features: state.selectedFeatures,
      contamination: state.contaminationAuto ? "auto" : state.contamination,
      n_estimators: state.nEstimators,
      missing_strategy: state.missingStrategy,
      random_state: 42,
    };

    try {
      const response = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Detection failed.");
      }

      state.detectionData = data;
      dom.downloadCsvTopBtn.disabled = false;
      dom.downloadCsvTableBtn.disabled = false;

      renderSummary(data.summary);
      renderScatterPlot();
      renderHistogram();
      renderTopAnomaliesTable();
    } catch (err) {
      if (err.name === "AbortError") {
        return;
      }
      showError(err.message || "Unable to complete anomaly detection.");
    } finally {
      if (state.activeController === controller) {
        setLoading(false);
      }
    }
  }

  function renderSummary(summary) {
    dom.statTotalRows.textContent = summary.total_rows.toLocaleString();
    const missing = summary.missing_handling || {};
    dom.statMissingNote.textContent = missing.message || "No missing values";
    dom.missingStatusText.textContent = missing.message || "Empty numeric cells handled cleanly.";

    dom.statAnomalyCount.textContent = summary.anomaly_count.toLocaleString();
    dom.statAnomalyPct.textContent = `${summary.anomaly_percentage.toFixed(2)}% of analyzed rows`;

    dom.statThreshold.textContent = summary.score_threshold.toFixed(4);
    dom.statContaminationMode.textContent =
      summary.contamination === "auto"
        ? "Auto threshold mode"
        : `Target contamination: ${(Number(summary.contamination) * 100).toFixed(1)}%`;

    const gt = summary.ground_truth_metrics;
    if (gt) {
      const precPct = (gt.precision * 100).toFixed(1);
      const recPct = (gt.recall * 100).toFixed(1);
      dom.statFourthLabel.textContent = "Ground-Truth Accuracy";
      dom.statFourthValue.textContent = `${precPct}% / ${recPct}%`;
      dom.statFourthSub.textContent = `Precision / Recall (${gt.true_positives} of ${gt.actual_anomalies} caught)`;
    } else {
      dom.statFourthLabel.textContent = "Model Configuration";
      dom.statFourthValue.textContent = `${summary.features_used.length} cols`;
      dom.statFourthSub.textContent = `${summary.n_estimators} trees · Standardized`;
    }
  }

  function buildHoverText(row, featuresUsed) {
    const lines = [
      `<b>Row #${row.index + 1}</b> · <b>${row.label === "anomaly" ? "Anomaly" : "Normal"}</b>`,
      `Anomaly Score: <b>${row.anomaly_score.toFixed(4)}</b>`,
    ];
    const shownCols = new Set(featuresUsed);
    if (state.xAxis) shownCols.add(state.xAxis);
    if (state.yAxis) shownCols.add(state.yAxis);

    shownCols.forEach((col) => {
      const rawVal = row.values[col];
      const valText = rawVal === null || rawVal === undefined ? "missing (imputed)" : formatNumber(rawVal);
      lines.push(`${escapeHtml(col)}: ${escapeHtml(valText)}`);
    });
    return lines.join("<br>");
  }

  function getCoord(row, axisName) {
    if (row.values[axisName] !== null && row.values[axisName] !== undefined) {
      return row.values[axisName];
    }
    if (row.model_values && row.model_values[axisName] !== undefined) {
      return row.model_values[axisName];
    }
    return null;
  }

  function renderScatterPlot() {
    if (!state.detectionData || typeof Plotly === "undefined") {
      return;
    }

    const { results, summary } = state.detectionData;
    const xCol = state.xAxis;
    const yCol = state.yAxis;
    const featuresUsed = summary.features_used || [];

    if (featuresUsed.length > 2) {
      dom.projectionNote.textContent =
        `Showing 2D projection (${xCol} vs ${yCol}). The Isolation Forest model evaluated all ${featuresUsed.length} selected features (${featuresUsed.join(", ")}).`;
    } else {
      dom.projectionNote.textContent =
        `Plotting ${xCol} vs ${yCol}. Normal points are blue; anomalies are larger red markers.`;
    }

    const normalX = [];
    const normalY = [];
    const normalText = [];
    const anomalyX = [];
    const anomalyY = [];
    const anomalyText = [];

    results.forEach((row) => {
      const xVal = getCoord(row, xCol);
      const yVal = getCoord(row, yCol);
      if (xVal === null || yVal === null) {
        return;
      }
      const hover = buildHoverText(row, featuresUsed);
      if (row.label === "anomaly") {
        anomalyX.push(xVal);
        anomalyY.push(yVal);
        anomalyText.push(hover);
      } else {
        normalX.push(xVal);
        normalY.push(yVal);
        normalText.push(hover);
      }
    });

    const traces = [
      {
        name: `Normal (${normalX.length})`,
        x: normalX,
        y: normalY,
        text: normalText,
        hoverinfo: "text",
        type: "scattergl",
        mode: "markers",
        marker: {
          color: "#2563eb",
          size: 7,
          opacity: 0.58,
        },
      },
      {
        name: `Anomaly (${anomalyX.length})`,
        x: anomalyX,
        y: anomalyY,
        text: anomalyText,
        hoverinfo: "text",
        type: "scatter",
        mode: "markers",
        marker: {
          color: "#dc2626",
          size: 13,
          opacity: 0.95,
          line: {
            color: "#ffffff",
            width: 1.5,
          },
        },
      },
    ];

    const layout = {
      margin: { l: 58, r: 20, t: 16, b: 48 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {
        family: "DM Sans, sans-serif",
        size: 12,
        color: "#0f172a",
      },
      xaxis: {
        title: { text: xCol, standoff: 10 },
        gridcolor: "#f1f5f9",
        zerolinecolor: "#e2e8f0",
      },
      yaxis: {
        title: { text: yCol, standoff: 10 },
        gridcolor: "#f1f5f9",
        zerolinecolor: "#e2e8f0",
      },
      legend: {
        orientation: "h",
        x: 0,
        y: 1.08,
        itemclick: "toggle",
        itemdoubleclick: "toggleothers",
      },
      hovermode: "closest",
      dragmode: "zoom",
    };

    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    };

    Plotly.react(dom.scatterPlot, traces, layout, config);
  }

  function renderHistogram() {
    if (!state.detectionData || typeof Plotly === "undefined") {
      return;
    }

    const { results, summary } = state.detectionData;
    const threshold = summary.score_threshold;

    const normalScores = [];
    const anomalyScores = [];
    results.forEach((row) => {
      if (row.label === "anomaly") {
        anomalyScores.push(row.anomaly_score);
      } else {
        normalScores.push(row.anomaly_score);
      }
    });

    const traces = [
      {
        name: "Normal",
        x: normalScores,
        type: "histogram",
        xbins: { start: 0, end: 1.001, size: 0.025 },
        marker: { color: "rgba(37, 99, 235, 0.72)" },
        hovertemplate: "Score bin: %{x}<br>Normal rows: %{y}<extra></extra>",
      },
      {
        name: "Anomaly",
        x: anomalyScores,
        type: "histogram",
        xbins: { start: 0, end: 1.001, size: 0.025 },
        marker: { color: "rgba(220, 38, 38, 0.88)" },
        hovertemplate: "Score bin: %{x}<br>Anomalies: %{y}<extra></extra>",
      },
    ];

    const layout = {
      barmode: "stack",
      margin: { l: 58, r: 20, t: 28, b: 46 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {
        family: "DM Sans, sans-serif",
        size: 12,
        color: "#0f172a",
      },
      xaxis: {
        title: { text: "Normalized Anomaly Score (0 = typical, 1 = most isolated)", standoff: 10 },
        range: [0, 1.02],
        gridcolor: "#f1f5f9",
      },
      yaxis: {
        title: { text: "Row Count", standoff: 10 },
        gridcolor: "#f1f5f9",
      },
      legend: {
        orientation: "h",
        x: 0,
        y: 1.12,
      },
      shapes: [
        {
          type: "line",
          x0: threshold,
          x1: threshold,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: {
            color: "#dc2626",
            width: 2,
            dash: "dash",
          },
        },
      ],
      annotations: [
        {
          x: threshold,
          y: 1,
          yref: "paper",
          text: `Threshold: ${threshold.toFixed(4)}`,
          showarrow: false,
          xanchor: threshold > 0.78 ? "right" : "left",
          yanchor: "bottom",
          xshift: threshold > 0.78 ? -6 : 6,
          font: {
            family: "JetBrains Mono, monospace",
            size: 11,
            color: "#dc2626",
          },
          bgcolor: "rgba(255,255,255,0.88)",
        },
      ],
    };

    const config = {
      responsive: true,
      displaylogo: false,
    };

    Plotly.react(dom.scoreHistogram, traces, layout, config);
  }

  function getRowSortValue(row, key) {
    if (key === "index") return row.index;
    if (key === "anomaly_score") return row.anomaly_score;
    if (key === "label") return row.label;
    return row.values[key];
  }

  function renderTopAnomaliesTable() {
    if (!state.detectionData) {
      return;
    }

    const { results } = state.detectionData;
    const top20 = [...results]
      .sort((a, b) => b.anomaly_score - a.anomaly_score)
      .slice(0, 20);

    const displayCols = state.numericColumns.slice(0, 6);
    const columns = [
      { key: "index", label: "Row #", numeric: true },
      ...displayCols.map((col) => ({ key: col, label: col, numeric: true })),
      { key: "anomaly_score", label: "Anomaly Score", numeric: true },
      { key: "label", label: "Status", numeric: false },
    ];

    top20.sort((a, b) => {
      const va = getRowSortValue(a, state.sortKey);
      const vb = getRowSortValue(b, state.sortKey);
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      let cmp = 0;
      if (typeof va === "number" && typeof vb === "number") {
        cmp = va - vb;
      } else {
        cmp = String(va).localeCompare(String(vb));
      }
      return state.sortDir === "asc" ? cmp : -cmp;
    });

    dom.anomaliesTableHead.textContent = "";
    const headRow = document.createElement("tr");

    columns.forEach((col) => {
      const th = document.createElement("th");
      const isActive = state.sortKey === col.key;
      th.setAttribute("aria-sort", isActive ? (state.sortDir === "asc" ? "ascending" : "descending") : "none");

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = col.numeric ? "sort-btn align-right" : "sort-btn";

      const labelSpan = document.createElement("span");
      labelSpan.textContent = col.label;
      btn.appendChild(labelSpan);

      const arrowSpan = document.createElement("span");
      arrowSpan.className = "sort-arrow";
      arrowSpan.textContent = isActive ? (state.sortDir === "asc" ? "▲" : "▼") : "";
      btn.appendChild(arrowSpan);

      btn.addEventListener("click", () => {
        if (state.sortKey === col.key) {
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = col.key;
          state.sortDir = col.numeric ? "desc" : "asc";
        }
        renderTopAnomaliesTable();
      });

      th.appendChild(btn);
      headRow.appendChild(th);
    });
    dom.anomaliesTableHead.appendChild(headRow);

    dom.anomaliesTableBody.textContent = "";
    top20.forEach((row) => {
      const tr = document.createElement("tr");

      const idxTd = document.createElement("td");
      idxTd.className = "num-cell";
      idxTd.textContent = String(row.index + 1);
      tr.appendChild(idxTd);

      displayCols.forEach((col) => {
        const td = document.createElement("td");
        td.className = "num-cell";
        td.textContent = formatNumber(row.values[col]);
        tr.appendChild(td);
      });

      const scoreTd = document.createElement("td");
      scoreTd.className = "num-cell";
      scoreTd.textContent = row.anomaly_score.toFixed(4);
      tr.appendChild(scoreTd);

      const labelTd = document.createElement("td");
      if (row.label === "anomaly") {
        labelTd.className = "status-anomaly";
        labelTd.textContent = "▲ Anomaly";
      } else {
        labelTd.className = "status-normal";
        labelTd.textContent = "Normal";
      }
      tr.appendChild(labelTd);

      dom.anomaliesTableBody.appendChild(tr);
    });

    dom.tableSubtitle.textContent =
      `Showing the ${top20.length} highest-scoring rows out of ${results.length.toLocaleString()}. Click column headers to sort.`;
  }

  function csvEscape(value) {
    if (value === null || value === undefined) {
      return "";
    }
    const str = String(value);
    if (/[",\n\r]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  }

  function downloadResultsCsv() {
    if (!state.detectionData || !state.detectionData.results.length) {
      return;
    }

    const headers = [...state.columns, "anomaly_score", "label"];
    const lines = [headers.map(csvEscape).join(",")];

    state.detectionData.results.forEach((row) => {
      const cells = state.columns.map((col) => csvEscape(row.values[col]));
      cells.push(csvEscape(row.anomaly_score.toFixed(4)));
      cells.push(csvEscape(row.label));
      lines.push(cells.join(","));
    });

    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = (state.filename || "dataset").replace(/\.csv$/i, "");
    link.href = url;
    link.download = `${baseName}_anomaly_results.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async function loadExample(name) {
    if (!name) return;
    setLoading(true);
    showError("");
    try {
      const response = await fetch(`/api/examples/${encodeURIComponent(name)}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to load example dataset.");
      }
      applyLoadedDataset(data, true);
    } catch (err) {
      setLoading(false);
      showError(err.message || "Could not load example dataset.");
    }
  }

  async function uploadCsvFile(file) {
    if (!file) return;
    setLoading(true);
    showError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "File upload failed.");
      }
      applyLoadedDataset(data, false);
    } catch (err) {
      setLoading(false);
      showError(err.message || "Unable to upload CSV file.");
    }
  }

  async function initExamples() {
    try {
      const response = await fetch("/api/examples");
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to list examples.");
      }
      state.examples = data.examples || [];
      dom.exampleSelect.textContent = "";

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select an example dataset...";
      dom.exampleSelect.appendChild(placeholder);

      state.examples.forEach((ex) => {
        const opt = document.createElement("option");
        opt.value = ex.name;
        opt.textContent = `${ex.title} (${ex.row_count} rows)`;
        dom.exampleSelect.appendChild(opt);
      });

      if (state.examples.length > 0) {
        const first = state.examples[0].name;
        dom.exampleSelect.value = first;
        await loadExample(first);
      }
    } catch (err) {
      showError(err.message || "Failed to initialize example datasets.");
    }
  }

  function bindEvents() {
    dom.exampleSelect.addEventListener("change", (e) => {
      if (e.target.value) {
        loadExample(e.target.value);
      }
    });

    dom.chooseFileBtn.addEventListener("click", () => {
      dom.csvFileInput.click();
    });

    dom.csvFileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) {
        uploadCsvFile(file);
        dom.csvFileInput.value = "";
      }
    });

    dom.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dom.dropZone.classList.add("drag-over");
    });

    dom.dropZone.addEventListener("dragleave", () => {
      dom.dropZone.classList.remove("drag-over");
    });

    dom.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dom.dropZone.classList.remove("drag-over");
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) {
        uploadCsvFile(file);
      }
    });

    dom.selectAllFeaturesBtn.addEventListener("click", () => {
      state.selectedFeatures = [...state.numericColumns];
      showWarning("");
      renderFeatureCheckboxes();
      scheduleDetect(0);
    });

    dom.xAxisSelect.addEventListener("change", (e) => {
      state.xAxis = e.target.value;
      renderScatterPlot();
    });

    dom.yAxisSelect.addEventListener("change", (e) => {
      state.yAxis = e.target.value;
      renderScatterPlot();
    });

    dom.contaminationSlider.addEventListener("input", (e) => {
      state.contamination = parseFloat(e.target.value);
      updateParameterLabels();
      scheduleDetect(DEBOUNCE_MS);
    });

    dom.contaminationAuto.addEventListener("change", (e) => {
      state.contaminationAuto = e.target.checked;
      updateParameterLabels();
      scheduleDetect(0);
    });

    dom.estimatorsSlider.addEventListener("input", (e) => {
      state.nEstimators = parseInt(e.target.value, 10);
      updateParameterLabels();
      scheduleDetect(DEBOUNCE_MS);
    });

    dom.missingStrategySelect.addEventListener("change", (e) => {
      state.missingStrategy = e.target.value;
      scheduleDetect(0);
    });

    dom.resetBtn.addEventListener("click", () => {
      if (!state.datasetId) return;
      state.selectedFeatures = [...state.numericColumns];
      state.xAxis = state.numericColumns[0] || "";
      state.yAxis = state.numericColumns[1] || state.numericColumns[0] || "";
      state.contamination = state.defaultContamination;
      state.contaminationAuto = false;
      state.nEstimators = 100;
      state.missingStrategy = "impute";

      dom.contaminationSlider.value = String(state.contamination);
      dom.contaminationAuto.checked = false;
      dom.estimatorsSlider.value = String(state.nEstimators);
      dom.missingStrategySelect.value = state.missingStrategy;
      showWarning("");
      updateParameterLabels();
      renderFeatureCheckboxes();
      renderAxisSelectors();
      scheduleDetect(0);
    });

    dom.downloadCsvTopBtn.addEventListener("click", downloadResultsCsv);
    dom.downloadCsvTableBtn.addEventListener("click", downloadResultsCsv);
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    updateParameterLabels();
    initExamples();
  });
})();
