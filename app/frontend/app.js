(() => {
  "use strict";

  const CLASS_INFO = [
    { code: "AKIEC", name: "Actinic keratoses", detail: "Actinic keratoses and intraepithelial carcinoma" },
    { code: "BCC", name: "Basal cell carcinoma", detail: "Basal cell carcinoma" },
    { code: "BKL", name: "Benign keratosis-like lesions", detail: "Benign keratosis-like lesions" },
    { code: "DF", name: "Dermatofibroma", detail: "Dermatofibroma" },
    { code: "MEL", name: "Melanoma", detail: "Melanoma" },
    { code: "NV", name: "Melanocytic nevus", detail: "Melanocytic nevi" },
    { code: "VASC", name: "Vascular lesions", detail: "Vascular lesions" },
  ];

  const state = { selectedFile: null, latestResult: null, currentView: "overview", isAnalyzing: false };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  function sendFrameHeight() {}

  function renderClassLists() {
    const html = CLASS_INFO.map((item) => `
      <div class="class-entry"><span class="class-code">${item.code}</span><span>${item.name}</span></div>
    `).join("");
    const overview = $("#class-list-overview");
    const model = $("#class-list-model");
    if (overview) overview.innerHTML = html;
    if (model) model.innerHTML = html;
  }

  function setView(viewName) {
    state.currentView = viewName;
    $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${viewName}`));
    $$(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.view === viewName));
    const names = { overview: "Overview", analyze: "Image analysis", model: "Model information" };
    $("#breadcrumb-current").textContent = names[viewName] || "Overview";
    window.scrollTo({ top: 0, behavior: "smooth" });

  }

  function showUpload() {
    state.latestResult = null;
    state.isAnalyzing = false;
    $("#upload-layout").classList.remove("hidden");
    $("#loading-panel").classList.add("hidden");
    $("#results-view").classList.add("hidden");
    setView("analyze");

  }

  function setUploadError(message) {
    const element = $("#upload-error");
    element.textContent = message;
    element.classList.remove("hidden");
  }

  function clearUploadError() {
    $("#upload-error").textContent = "";
    $("#upload-error").classList.add("hidden");
  }

  function resetSelectedFile() {
    state.selectedFile = null;
    state.previewDataUrl = null;
    $("#image-input").value = "";
    $("#preview-panel").classList.add("hidden");
    $("#drop-zone").classList.remove("hidden");
    $("#analyze-button").disabled = true;
    clearUploadError();
  }

  function formatBytes(bytes) {
    return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function loadImageFile(file) {
    clearUploadError();
    if (!file) return;
    const validType = ["image/jpeg", "image/png"].includes(file.type) || /\.(jpe?g|png)$/i.test(file.name);
    if (!validType) {
      resetSelectedFile();
      setUploadError("Unsupported file type. Please choose a JPG, JPEG or PNG image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      resetSelectedFile();
      setUploadError("This file is larger than 10 MB. Please choose a smaller image.");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      state.selectedFile = file;
      $("#image-preview").src = objectUrl;
      $("#selected-file-name").textContent = file.name;
      $("#image-dimensions").textContent = `${image.naturalWidth} × ${image.naturalHeight} · ${formatBytes(file.size)}`;
      $("#drop-zone").classList.add("hidden");
      $("#preview-panel").classList.remove("hidden");
      $("#analyze-button").disabled = false;
  
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setUploadError("The selected image could not be opened. Please try another file.");
    };
    image.src = objectUrl;
  }

  async function beginAnalysis() {
    if (!state.selectedFile || state.isAnalyzing) return;
    clearUploadError();
    state.isAnalyzing = true;
    $("#analyze-button").disabled = true;
    $("#upload-layout").classList.add("hidden");
    $("#loading-panel").classList.remove("hidden");
    $("#results-view").classList.add("hidden");

    try {
      const form = new FormData();
      form.append("image", state.selectedFile);
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The image could not be analyzed.");
      renderResult(payload);
    } catch (error) {
      state.isAnalyzing = false;
      $("#loading-panel").classList.add("hidden");
      $("#upload-layout").classList.remove("hidden");
      $("#analyze-button").disabled = false;
      setUploadError(error.message || "Something went wrong. Please try again.");
    }
  }

  function renderProbabilities(probabilities) {
    const entries = Object.entries(probabilities || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
    $("#probability-list").innerHTML = entries.map(([name, probability]) => {
      const value = Math.max(0, Math.min(100, Number(probability) || 0));
      return `<div class="probability-item"><span class="probability-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span><div class="probability-track"><span style="width:${value}%"></span></div><span class="probability-value">${value.toFixed(2)}%</span></div>`;
    }).join("");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    })[character]);
  }

  function renderResult(result) {
    if (!result) return;
    state.latestResult = result;
    state.isAnalyzing = false;
    $("#upload-layout").classList.add("hidden");
    $("#loading-panel").classList.add("hidden");
    $("#results-view").classList.remove("hidden");
    $("#predicted-class").textContent = result.predictedClass || "Prediction unavailable";
    const confidence = Math.max(0, Math.min(100, Number(result.confidence) || 0));
    $("#confidence-value").textContent = `${confidence.toFixed(2)}%`;
    $("#confidence-bar").style.width = `${confidence}%`;
    renderProbabilities(result.probabilities);
    $("#result-original").src = result.originalImage || "";
    $("#result-gradcam").src = result.gradcamImage || "";
    setView("analyze");

  }

  function renderError(message) {
    if (!message) return;
    state.isAnalyzing = false;
    $("#loading-panel").classList.add("hidden");
    $("#results-view").classList.add("hidden");
    $("#upload-layout").classList.remove("hidden");
    setView("analyze");
    setUploadError(message);
    $("#analyze-button").disabled = !state.selectedFile;

  }

  function downloadSummary() {
    if (!state.latestResult) return;
    const result = state.latestResult;
    const lines = [
      "DermVision - Analysis Summary",
      "================================",
      `Predicted class: ${result.predictedClass || "N/A"}`,
      `Model confidence: ${Number(result.confidence || 0).toFixed(2)}%`,
      "",
      "Class probabilities:",
      ...Object.entries(result.probabilities || {}).map(([name, value]) => `${name}: ${Number(value).toFixed(2)}%`),
      "",
      "Grad-CAM: Generated by the existing DermVision Grad-CAM implementation.",
      "",
      "DISCLAIMER: Educational and research prototype only. Not for diagnosis, treatment, or clinical decision-making.",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "dermvision-analysis-summary.txt";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function wireEvents() {
    $$(".nav-link").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    ["#hero-start", "#workflow-start"].forEach((selector) => $(selector).addEventListener("click", showUpload));
    $("#drop-zone").addEventListener("click", (event) => {
      if (event.target.id !== "browse-button") $("#image-input").click();
    });
    $("#drop-zone").addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        $("#image-input").click();
      }
    });
    $("#browse-button").addEventListener("click", (event) => {
      event.stopPropagation();
      $("#image-input").click();
    });
    $("#image-input").addEventListener("change", (event) => loadImageFile(event.target.files[0]));
    $("#remove-file").addEventListener("click", resetSelectedFile);
    $("#analyze-button").addEventListener("click", beginAnalysis);
    $("#drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("#drop-zone").classList.add("dragover"); });
    $("#drop-zone").addEventListener("dragleave", () => $("#drop-zone").classList.remove("dragover"));
    $("#drop-zone").addEventListener("drop", (event) => {
      event.preventDefault();
      $("#drop-zone").classList.remove("dragover");
      const file = event.dataTransfer.files[0];
      if (file) loadImageFile(file);
    });
    ["#new-analysis", "#result-new-analysis"].forEach((selector) => $(selector).addEventListener("click", () => {
      resetSelectedFile();
      showUpload();

    }));
    $("#download-summary").addEventListener("click", downloadSummary);
  }

  function setupSidebar() {
    const toggle = $("#sidebar-toggle");
    const backdrop = $("#sidebar-backdrop");
    const mobile = () => window.matchMedia("(max-width: 740px)").matches;
    const storeKey = "dermvision-sidebar-collapsed";
    let saved = false;
    try { saved = localStorage.getItem(storeKey) === "true"; } catch (_) {}
    document.body.classList.toggle("sidebar-collapsed", saved);
    const sync = () => {
      const expanded = mobile() ? document.body.classList.contains("sidebar-mobile-open") : !document.body.classList.contains("sidebar-collapsed");
      toggle.setAttribute("aria-expanded", String(expanded));
    };
    const change = () => {
      if (mobile()) document.body.classList.toggle("sidebar-mobile-open");
      else {
        const isCollapsed = document.body.classList.toggle("sidebar-collapsed");
        try { localStorage.setItem(storeKey, String(isCollapsed)); } catch (_) {}
      }
      sync();
    };
    toggle.addEventListener("click", change);
    backdrop.addEventListener("click", () => { document.body.classList.remove("sidebar-mobile-open"); sync(); });
    $$(".nav-link").forEach((button) => button.addEventListener("click", () => {
      if (mobile()) document.body.classList.remove("sidebar-mobile-open");
      sync();
    }));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("sidebar-mobile-open")) {
        document.body.classList.remove("sidebar-mobile-open");
        sync();
      }
    });
    window.addEventListener("resize", () => {
      if (!mobile()) document.body.classList.remove("sidebar-mobile-open");
      sync();
    });
    sync();
  }

  function init() {
    renderClassLists();
    wireEvents();
    setupSidebar();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
