const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const videoInput = $("#videoInput");
const dropZone = $("#dropZone");
const fileSummary = $("#fileSummary");
const analyseButton = $("#analyseButton");
const processingCard = $("#processingCard");
const resultsSection = $("#resultsSection");
const toast = $("#toast");
let selectedFile = null;
let toastTimer;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function formatBytes(bytes) {
  if (!bytes) return "Review sample";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function selectVideo(file, isDemo = false) {
  if (!isDemo && file && !file.type.startsWith("video/")) {
    showToast("Please choose a video file.");
    return;
  }
  selectedFile = file || { name: "Nattadavu_01.mp4", size: 90060991 };
  $("#fileName").textContent = selectedFile.name;
  $("#fileMeta").textContent = `${formatBytes(selectedFile.size)} · first 20 seconds selected`;
  dropZone.hidden = true;
  fileSummary.hidden = false;
  analyseButton.disabled = false;
  processingCard.hidden = true;
  resultsSection.hidden = true;
}

function resetWorkspace() {
  selectedFile = null;
  videoInput.value = "";
  dropZone.hidden = false;
  fileSummary.hidden = true;
  analyseButton.disabled = true;
  processingCard.hidden = true;
  resultsSection.hidden = true;
  $("#progressBar").style.width = "0%";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function drawChart(canvas) {
  const scale = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, rect.width * scale);
  canvas.height = Math.max(1, rect.height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const width = rect.width;
  const height = rect.height;
  const seed = Number(canvas.dataset.seed || 1);

  ctx.strokeStyle = "rgba(255,255,255,.055)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (height / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }

  const series = [
    { color: "#6da8ff", offset: 0.15, amp: .21 },
    { color: "#ff6b5a", offset: 0.52, amp: .17 },
    { color: "#76d49b", offset: 0.78, amp: .10 },
  ];
  series.forEach((line, index) => {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = index === 2 ? 1.25 : 1.5;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = (i / 100) * width;
      const step = Math.sin((i + seed * 8) / (6.5 + index)) * line.amp;
      const beat = Math.sin((i + seed * 13) / 2.9) * (index === 2 ? .045 : .025);
      const y = height * (line.offset + step + beat);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  });
}

function drawAllCharts() {
  $$('canvas[data-seed]').forEach(drawChart);
}

function runMockAnalysis() {
  if (!selectedFile) return;
  analyseButton.disabled = true;
  processingCard.hidden = false;
  resultsSection.hidden = true;
  processingCard.scrollIntoView({ behavior: "smooth", block: "center" });

  const stages = [
    { at: 0, title: "Extracting 600 frames", detail: "0 / 600" },
    { at: 28, title: "Running RTMW3D pose detection", detail: "RTMW3D · 133 points" },
    { at: 62, title: "Tracking toe and heel movement", detail: "Toe + heel · X/Y/Z" },
    { at: 86, title: "Preparing graphs and exports", detail: "Graphs + CSV + video" },
  ];
  let progress = 0;
  const timer = setInterval(() => {
    progress += Math.floor(Math.random() * 5) + 2;
    progress = Math.min(progress, 100);
    const stageIndex = progress >= 86 ? 3 : progress >= 62 ? 2 : progress >= 28 ? 1 : 0;
    const stage = stages[stageIndex];
    $("#processingTitle").textContent = stage.title;
    $("#progressPercent").textContent = `${progress}%`;
    $("#progressBar").style.width = `${progress}%`;
    $$(".pipeline-step").forEach((step, index) => {
      step.classList.toggle("active", index === stageIndex);
      step.classList.toggle("complete", index < stageIndex || progress === 100);
    });
    if (stageIndex === 0) {
      $(".pipeline-step small").textContent = `${Math.round(progress / 28 * 600)} / 600`;
    }

    if (progress === 100) {
      clearInterval(timer);
      setTimeout(() => {
        processingCard.hidden = true;
        resultsSection.hidden = false;
        analyseButton.disabled = false;
        drawAllCharts();
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 350);
    }
  }, 110);
}

$("#chooseButton").addEventListener("click", (event) => { event.stopPropagation(); videoInput.click(); });
dropZone.addEventListener("click", () => videoInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); videoInput.click(); }
});
videoInput.addEventListener("change", () => selectVideo(videoInput.files[0]));
$("#removeFile").addEventListener("click", resetWorkspace);
$("#demoButton").addEventListener("click", () => selectVideo(null, true));
$("#recentDemo").addEventListener("click", () => { selectVideo(null, true); runMockAnalysis(); });
$("#newAnalysis").addEventListener("click", resetWorkspace);
analyseButton.addEventListener("click", runMockAnalysis);

["dragenter", "dragover"].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => selectVideo(event.dataTransfer.files[0]));

$$('[data-toast]').forEach((button) => button.addEventListener("click", () => showToast(button.dataset.toast)));
$("#menuButton").addEventListener("click", () => $("#sidebar").classList.add("open"));
$("#sidebarClose").addEventListener("click", () => $("#sidebar").classList.remove("open"));
window.addEventListener("resize", () => { if (!resultsSection.hidden) drawAllCharts(); });
window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "n" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) resetWorkspace();
});
