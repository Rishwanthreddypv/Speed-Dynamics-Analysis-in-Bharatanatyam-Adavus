const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let videoInput;
let dropZone;
let fileSummary;
let processingCard;
let resultsSection;
let toast;
let selectedFile = null;
let activeJobId = null;
let pollTimer = null;
let elapsedTimer = null;
let startedAt = null;
let toastTimer = null;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatElapsed(milliseconds) {
  const total = Math.floor(milliseconds / 1000);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function selectVideo(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {
    showToast("Please choose a supported video file.");
    return;
  }
  selectedFile = file;
  $("#fileName").textContent = file.name;
  $("#fileMeta").textContent = `${formatBytes(file.size)} · Ready to analyse`;
  dropZone.hidden = true;
  fileSummary.hidden = false;
  $("#analyseButton").disabled = false;
  resultsSection.hidden = true;
}

function stopMonitoring() {
  clearTimeout(pollTimer);
  clearInterval(elapsedTimer);
}

function resetWorkspace() {
  stopMonitoring();
  activeJobId = null;
  selectedFile = null;
  videoInput.value = "";
  dropZone.hidden = false;
  fileSummary.hidden = true;
  processingCard.hidden = true;
  resultsSection.hidden = true;
  $("#analyseButton").disabled = true;
  $("#progressBar").style.width = "0%";
  $("#progressPercent").textContent = "0%";
}

function setStage(stage, message, status) {
  const safeStage = Math.min(Math.max(Number(stage) || 0, 0), 4);
  const percentages = [8, 34, 62, 86, 100];
  $("#processingTitle").textContent = message || "Analysing movement";
  $("#progressPercent").textContent = status === "queued" ? "Queued" : `${percentages[safeStage]}%`;
  $("#progressBar").style.width = `${percentages[safeStage]}%`;
  $$(".pipeline-step").forEach((step, index) => {
    step.classList.toggle("active", safeStage < 4 && index === safeStage);
    step.classList.toggle("complete", safeStage >= 4 || index < safeStage);
  });
}

function artifactUrl(jobId, path) {
  return `/api/jobs/${jobId}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function renderResults(job) {
  const metrics = job.metrics || {};
  const artifactCards = (job.artifacts || []).map((item) => `
    <a class="artifact-card" href="${artifactUrl(job.id, item.path)}">
      <span class="artifact-kind">${item.kind || "file"}</span>
      <strong>${item.name}</strong>
      <small>${formatBytes(item.size)}</small>
    </a>`).join("");
  resultsSection.innerHTML = `
    <div class="results-header">
      <div><div class="eyebrow">ANALYSIS COMPLETE</div><h2>Movement overview</h2></div>
      <button class="ghost-button" id="newResultAnalysis" type="button">Analyse another video</button>
    </div>
    <div class="summary-grid">
      <article><span>Frames analysed</span><strong>${metrics.frames ?? "—"}</strong><small>${metrics.fps ?? 0} FPS</small></article>
      <article><span>Duration</span><strong>${metrics.duration ?? 0}s</strong><small>Full video</small></article>
      <article><span>Keypoints</span><strong>${metrics.keypoints ?? 4}</strong><small>Toe + heel</small></article>
      <article><span>Observations</span><strong>${metrics.observations ?? 0}</strong><small>Tracked rows</small></article>
    </div>
    <div class="artifacts-heading"><div><div class="eyebrow">GENERATED FILES</div><h3>Download results</h3></div><span>${job.artifacts.length} files</span></div>
    <div class="artifact-grid">${artifactCards || "<p>No downloadable artifacts were found.</p>"}</div>`;
  resultsSection.hidden = false;
  $("#newResultAnalysis").addEventListener("click", resetWorkspace);
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Could not read analysis status.");
    const job = await response.json();
    setStage(job.stage, job.message, job.status);
    if (job.status === "complete") {
      stopMonitoring();
      processingCard.hidden = true;
      $("#analyseButton").disabled = false;
      renderResults(job);
      return;
    }
    if (job.status === "failed") {
      stopMonitoring();
      $("#analyseButton").disabled = false;
      showToast(job.error || "The analysis failed.");
      $("#processingTitle").textContent = "Analysis failed";
      return;
    }
    pollTimer = setTimeout(pollJob, 1500);
  } catch (error) {
    showToast(error.message);
    pollTimer = setTimeout(pollJob, 3000);
  }
}

async function startAnalysis() {
  if (!selectedFile) return;
  $("#analyseButton").disabled = true;
  processingCard.hidden = false;
  resultsSection.hidden = true;
  setStage(0, "Uploading video", "running");
  processingCard.scrollIntoView({ behavior: "smooth", block: "center" });
  startedAt = Date.now();
  elapsedTimer = setInterval(() => {
    $("#fileMeta").textContent = `${formatBytes(selectedFile.size)} · Elapsed ${formatElapsed(Date.now() - startedAt)}`;
  }, 1000);

  const form = new FormData();
  form.append("video", selectedFile);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Upload failed.");
    activeJobId = payload.job_id;
    pollJob();
  } catch (error) {
    stopMonitoring();
    $("#analyseButton").disabled = false;
    processingCard.hidden = true;
    showToast(error.message);
  }
}

document.body.innerHTML = document.body.innerHTML
  .replaceAll("Nattadavu", "Thattadavu")
  .replaceAll("First 20 seconds", "Full video")
  .replaceAll("first 20 sec selected", "full video")
  .replaceAll("across the first 20 seconds.", "across the full performance.")
  .replaceAll("20 sec · 600 frames", "Full-video analysis")
  .replaceAll("analysed locally in this prototype", "processed by the local analysis pipeline");

videoInput = $("#videoInput");
dropZone = $("#dropZone");
fileSummary = $("#fileSummary");
processingCard = $("#processingCard");
resultsSection = $("#resultsSection");
toast = $("#toast");

$("#chooseButton").addEventListener("click", (event) => { event.stopPropagation(); videoInput.click(); });
dropZone.addEventListener("click", () => videoInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); videoInput.click(); }
});
videoInput.addEventListener("change", () => selectVideo(videoInput.files[0]));
$("#removeFile").addEventListener("click", resetWorkspace);
$("#analyseButton").addEventListener("click", startAnalysis);
$("#newAnalysis").addEventListener("click", resetWorkspace);
$("#demoButton").hidden = true;
$("#recentDemo").hidden = true;

["dragenter", "dragover"].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => selectVideo(event.dataTransfer.files[0]));
$("#menuButton").addEventListener("click", () => $("#sidebar").classList.add("open"));
$("#sidebarClose").addEventListener("click", () => $("#sidebar").classList.remove("open"));
