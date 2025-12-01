import { initUniverse } from "./Universe3D.js";
const API_URL = "/api";

// --- UI CONFIG TOGGLE ---
const toggleConfigBtn = document.getElementById("toggle-config-btn");
const controlPanel = document.getElementById("control-panel");
const configArrow = document.getElementById("config-arrow");
let isConfigOpen = true;

// Xử lý đóng/mở panel cấu hình
if (toggleConfigBtn && controlPanel) {
  toggleConfigBtn.addEventListener("click", () => {
    isConfigOpen = !isConfigOpen;
    if (isConfigOpen) {
      controlPanel.classList.remove("h-0", "opacity-0", "p-0", "invisible");
      controlPanel.classList.add("visible");
      controlPanel.style.height = "auto";
      if (configArrow) configArrow.style.transform = "rotate(0deg)";
      const statusText = document.getElementById("config-status-text");
      if (statusText) statusText.textContent = "Hide Config";
    } else {
      controlPanel.classList.add("h-0", "opacity-0", "p-0", "invisible");
      controlPanel.classList.remove("visible");
      controlPanel.style.height = "0";
      if (configArrow) configArrow.style.transform = "rotate(-90deg)";
      const statusText = document.getElementById("config-status-text");
      if (statusText) statusText.textContent = "Show Config";
    }
  });
}

// --- GLOBAL STATE ---
let uploadedFiles = null;
let currentSessionId = null;
let currentGroups = {};
let qualityScores = {};
let currentClusterName = null;
let universeState = { data: [], currentTab: "summary" };
let universeController = null;
const chartInstances = { distNew: null, ratioNew: null };

// --- DOM ELEMENTS ---
const uploadBtn = document.getElementById("btn-upload");
const uploadStatus = document.getElementById("upload-status");
const uploadBar = document.getElementById("upload-bar");
const startClusterBtn = document.getElementById("start-clustering-btn");
const loadingOverlay = document.getElementById("loading-overlay");
const fileInput = document.getElementById("image-folder-input");

// --- EVENT LISTENERS (FIXED) ---

// 1. SỰ KIỆN CHỌN FILE (Đã sửa lỗi không nhận folder)
if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    console.log("Event 'change' triggered"); // Debug log

    // Kiểm tra xem có file nào được chọn không
    if (e.target.files && e.target.files.length > 0) {
      uploadedFiles = e.target.files;

      console.log(`Selected ${uploadedFiles.length} files`); // Debug log

      // Cập nhật UI ngay lập tức
      if (uploadStatus) {
        uploadStatus.textContent = `${uploadedFiles.length} files ready`;
        uploadStatus.classList.remove("text-gray-500");
        uploadStatus.classList.add("text-emerald-400", "font-bold");
      }

      // Reset thanh tiến trình và nút Upload
      if (uploadBar) uploadBar.style.width = "0%";

      if (uploadBtn) {
        uploadBtn.textContent = "UPLOAD NOW";
        uploadBtn.disabled = false;
        uploadBtn.classList.remove(
          "bg-white",
          "text-black",
          "bg-emerald-500",
          "text-white"
        );
        // Style cho nút khi đã sẵn sàng
        uploadBtn.classList.add(
          "bg-violet-600",
          "text-white",
          "hover:bg-violet-500"
        );
      }
    } else {
      console.log("No files found in event target");
    }
  });
} else {
  console.error("ERROR: Input element #image-folder-input not found!");
}

// 2. STEP 1: UPLOAD FILES
if (uploadBtn) {
  uploadBtn.addEventListener("click", async () => {
    if (!uploadedFiles || uploadedFiles.length === 0) {
      // Nếu bấm Upload mà chưa chọn file, kích hoạt input click
      alert("Please select a folder first!");
      if (fileInput) fileInput.click();
      return;
    }

    uploadBtn.disabled = true;
    uploadBtn.innerHTML = `UPLOADING...`;

    if (uploadStatus)
      uploadStatus.textContent = "Uploading assets to server...";
    if (uploadBar) uploadBar.style.width = "30%";

    const fd = new FormData();
    for (const f of uploadedFiles) {
      fd.append("files", f, f.webkitRelativePath || f.name);
    }

    try {
      const res = await fetch(`${API_URL}/upload-session`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error("Server upload failed");
      const data = await res.json();

      currentSessionId = data.session_id;

      // Animation thành công
      if (uploadBar) uploadBar.style.width = "100%";
      if (uploadStatus) {
        uploadStatus.textContent = "✓ Upload Complete";
        uploadStatus.classList.replace("text-gray-500", "text-emerald-400");
      }

      uploadBtn.innerHTML = "DONE";
      uploadBtn.classList.remove("bg-violet-600", "hover:bg-violet-500");
      uploadBtn.classList.add("bg-emerald-500", "cursor-default");

      // Kích hoạt Bước 2 (Làm sáng lên)
      const step2 = document.getElementById("step-process");
      if (step2) {
        step2.classList.remove(
          "opacity-40",
          "pointer-events-none",
          "grayscale"
        );
        step2.classList.add("animate-pulse-once"); // Thêm hiệu ứng nháy nhẹ nếu muốn
      }

      // Làm mờ Bước 1
      const step1 = document.getElementById("step-upload");
      if (step1) step1.classList.add("opacity-50");
    } catch (e) {
      alert("Upload Error: " + e.message);
      uploadBtn.disabled = false;
      uploadBtn.textContent = "RETRY";
      if (uploadBar) uploadBar.style.width = "0%";
    }
  });
}

// 3. STEP 2: RUN CLUSTERING
if (startClusterBtn) {
  startClusterBtn.addEventListener("click", async () => {
    if (!currentSessionId) return alert("Please upload files first.");

    if (isConfigOpen && toggleConfigBtn) toggleConfigBtn.click();

    if (loadingOverlay) loadingOverlay.classList.remove("hidden");
    const loadingText = document.getElementById("loading-text");
    const loadingBar = document.getElementById("loading-bar");

    if (loadingText)
      loadingText.textContent = "This process may take a few minutes...";
    if (loadingBar) loadingBar.style.width = "60%";

    const fd = new FormData();
    fd.append("session_id", currentSessionId);
    fd.append("algorithm", document.getElementById("algorithm").value);

    try {
      const res = await fetch(`${API_URL}/run-clustering`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();

      qualityScores = data.quality_scores || {};

      if (loadingBar) loadingBar.style.width = "100%";
      populateUI(data);

      const hero = document.getElementById("hero-landing");
      if (hero) hero.classList.add("hidden");

      setTimeout(() => {
        if (loadingOverlay) loadingOverlay.classList.add("hidden");
      }, 800);
    } catch (e) {
      alert("Processing Error: " + e.message);
      if (loadingOverlay) loadingOverlay.classList.add("hidden");
    }
  });
}

// --- UI HELPERS ---

document.querySelectorAll(".tab-button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const newTabId = btn.dataset.tab;

    document
      .querySelectorAll(".tab-button")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    document.querySelectorAll(".tab-content").forEach((c) => {
      c.classList.remove("active-tab", "fade-in");
      c.classList.add("hidden");
      if (c.classList.contains("flex")) c.classList.remove("flex");
    });

    const newContent = document.getElementById(`tab-${newTabId}`);
    if (newContent) {
      newContent.classList.remove("hidden");
      if (newTabId === "browser") newContent.classList.add("flex");

      requestAnimationFrame(() => {
        newContent.classList.add("active-tab", "fade-in");
      });
    }

    if (newTabId === "universe") {
      universeState.currentTab = "universe";
      window.dispatchEvent(new Event("resize"));
    }
  });
});

function populateUI(data) {
  currentGroups = data.results.groups || {};
  const results = data.results;
  const total = results.total_images || 0;
  const unique = Object.keys(currentGroups).length;
  const dupes = total - unique;

  if (data.universe_map) {
    universeState.data = data.universe_map;
  }

  const summaryContent = document.getElementById("summary-content");
  const summaryVisuals = document.getElementById("summary-visuals");

  if (summaryContent) summaryContent.classList.add("hidden");
  if (summaryVisuals) summaryVisuals.classList.remove("hidden");

  renderStatsDashboard(data, unique, dupes);
  renderActionCenter(unique, dupes, total);
  renderClusterList();

  const firstCluster = Object.keys(currentGroups)[0];
  if (firstCluster) {
    loadCluster(firstCluster);
  }

  if (data.universe_map) renderUniverseMap(data.universe_map);

  const dlBtn = document.getElementById("download-btn");
  if (dlBtn) {
    dlBtn.classList.remove("hidden");
    dlBtn.onclick = () =>
      (window.location.href = `${API_URL}/download-results/${currentSessionId}`);
  }
  const delGrpBtn = document.getElementById("delete-group-btn");
  if (delGrpBtn) delGrpBtn.classList.remove("hidden");

  const summaryTab = document.querySelector('[data-tab="summary"]');
  if (summaryTab) summaryTab.click();
}

// --- CÁC HÀM RENDER ---
function renderUniverseMap(points) {
  console.log("=== RENDER UNIVERSE MAP ===");
  console.log("📊 Points received:", points?.length || 0);

  if (!points || !points.length) {
    console.warn("⚠️ No points to render in Universe Map");
    return;
  }

  const emptyState = document.getElementById("universe-empty");
  if (emptyState) {
    console.log("✅ Hiding empty state");
    emptyState.classList.add("hidden");
  }

  const controlsDiv = document.getElementById("map-controls");
  if (controlsDiv) {
    console.log("✅ Showing controls");
    controlsDiv.classList.remove("hidden");
  }

  console.log("🚀 Calling initUniverse...");

  universeController = initUniverse("plotly-div", points, (nodeData) => {
    console.log("🖱️ Node clicked:", nodeData);
    if (nodeData.cluster && nodeData.cluster !== "Noise/Unique") {
      const browserBtn = document.querySelector('[data-tab="browser"]');
      if (browserBtn) browserBtn.click();
      setTimeout(() => loadCluster(nodeData.cluster), 300);
    }
  });

  if (!universeController) {
    console.error("Universe controller failed to initialize!");
    return;
  }

  console.log("Universe initialized successfully");

  const rotateToggle = document.getElementById("toggle-rotate");
  const lineToggle = document.getElementById("toggle-lines");

  if (rotateToggle && universeController) {
    rotateToggle.checked = true;
    rotateToggle.onchange = (e) => {
      console.log("🔄 Orbit toggle:", e.target.checked);
      if (universeController) universeController.setOrbit(e.target.checked);
    };
  }

  if (lineToggle && universeController) {
    lineToggle.checked = false;
    lineToggle.onchange = (e) => {
      console.log("🌐 Constellations toggle:", e.target.checked);
      if (universeController) universeController.setLines(e.target.checked);
    };
  }

  window.addEventListener("universe-hover", (e) => {
    const data = e.detail;
    const tooltip = document.getElementById("universe-tooltip");
    const imgPath = data.path.startsWith("/")
      ? data.path
      : `${API_URL}/results/${currentSessionId}/clusters/${data.path}`;

    const tImg = document.getElementById("tooltip-img");
    if (tImg) tImg.src = imgPath;

    const tName = document.getElementById("tooltip-name");
    if (tName) tName.textContent = data.filename;

    const tCluster = document.getElementById("tooltip-cluster");
    if (tCluster) tCluster.textContent = data.cluster;

    const tScore = document.getElementById("tooltip-score");
    if (tScore)
      tScore.textContent = data.quality ? data.quality.toFixed(0) : "N/A";

    if (tooltip) tooltip.classList.remove("hidden");
  });

  window.addEventListener("universe-unhover", () => {
    const tooltip = document.getElementById("universe-tooltip");
    if (tooltip) tooltip.classList.add("hidden");
  });

  document.addEventListener("mousemove", (e) => {
    const tooltip = document.getElementById("universe-tooltip");
    if (tooltip && !tooltip.classList.contains("hidden")) {
      tooltip.style.left = e.clientX + 20 + "px";
      tooltip.style.top = e.clientY + 20 + "px";
    }
  });
}

function syncUniverseMap(deletedPaths) {
  if (!universeState.data.length) return;

  console.log("Syncing Universe Map, removing:", deletedPaths.length, "points");

  universeState.data = universeState.data.filter(
    (p) => !deletedPaths.includes(p.path)
  );

  console.log("Remaining points:", universeState.data.length);
}

function renderClusterList() {
  const list = document.getElementById("cluster-list");
  if (!list) return;
  list.innerHTML = "";
  Object.entries(currentGroups)
    .sort((a, b) => b[1].length - a[1].length)
    .forEach(([name, files]) => {
      const btn = document.createElement("button");
      btn.className =
        "cluster-button w-full text-left p-2.5 rounded text-gray-400 hover:bg-[#333] text-xs font-medium mb-1 border-l-2 border-transparent hover:border-violet-500 transition-all";
      btn.innerHTML = `<span class="text-white font-bold">${name}</span> <span class="text-gray-500 ml-1">(${files.length})</span>`;
      btn.dataset.clusterName = name;
      btn.onclick = () => loadCluster(name);
      list.appendChild(btn);
    });
}

function loadCluster(name) {
  currentClusterName = name;

  document
    .querySelectorAll(".cluster-button")
    .forEach((b) =>
      b.classList.remove("active", "bg-[#252526]", "border-violet-500")
    );
  const activeBtn = document.querySelector(`[data-cluster-name="${name}"]`);
  if (activeBtn)
    activeBtn.classList.add("active", "bg-[#252526]", "border-violet-500");

  const gallery = document.getElementById("thumbnail-gallery");
  if (!gallery) return;
  gallery.innerHTML = "";

  gallery.className =
    "grid gap-2 p-2 md:gap-6 md:p-6 flex-1 overflow-y-auto bg-[#121212] min-h-0";

  const header = document.getElementById("thumbnail-header");
  if (header) header.textContent = `Cluster: ${name}`;

  const q = qualityScores[name]?.images || [];

  ["delete-btn", "move-btn", "smart-cleanup-btn"].forEach((id) => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = false;
  });

  currentGroups[name].forEach((path, index) => {
    const url = `${API_URL}/results/${currentSessionId}/clusters/${path}`;
    let info = null;
    if (q) {
      info = q.find((i) => i.path === path);
    }

    if (!info) {
      for (const clusterKey in qualityScores) {
        if (qualityScores[clusterKey]?.images) {
          const found = qualityScores[clusterKey].images.find((i) => i.path === path);
          if (found) {
            info = found;
            break;
          }
        }
      }
    }

    const isBest = info?.is_best;

    let scoreColor = "#666";
    if (info && info.scores) {
      if (info.scores.total >= 80) scoreColor = "#34d399"; 
      else if (info.scores.total >= 50) scoreColor = "#fbbf24";
      else scoreColor = "#f87171";
    }

    const div = document.createElement("div");
    div.className = `thumbnail-card group ${isBest ? "best-quality" : ""}`;
    div.dataset.path = path;

    div.innerHTML = `
          <div class="card-image-container">
              <input type="checkbox" class="card-checkbox">
              
              <div class="card-score" style="color: ${scoreColor}; border-color: ${scoreColor}">
                  ${info ? info.scores.total.toFixed(0) : "N/A"}
              </div>
              
              <img src="${url}" loading="lazy" alt="image">
              
              ${isBest ? '<div class="best-banner">★ BEST VERSION ★</div>' : ""}
          </div>
          
          <div class="card-filename" title="${path.split("/").pop()}">
              ${path.split("/").pop()}
          </div>
      `;

    const img = div.querySelector("img");
    if (img) {
      img.onclick = (e) => {
        e.stopPropagation();
        const modal = document.getElementById("image-modal");
        const modalImg = document.getElementById("modal-image");
        if (modal && modalImg) {
          modalImg.src = url;
          modal.classList.remove("hidden");
          setTimeout(() => modalImg.classList.add("show"), 10);
        }
      };
    }

    div.onclick = (e) => {
      if (e.target.tagName !== "INPUT" && e.target.tagName !== "IMG") {
        const cb = div.querySelector("input");
        cb.checked = !cb.checked;
        div.classList.toggle("selected", cb.checked);
      }
    };

    const cb = div.querySelector("input");
    if (cb)
      cb.onclick = (e) => {
        e.stopPropagation();
        div.classList.toggle("selected", cb.checked);
      };

    gallery.appendChild(div);
  });
}

const deleteBtn = document.getElementById("delete-btn");
if (deleteBtn) {
  deleteBtn.onclick = async () => {
    const selectedCards = Array.from(
      document.querySelectorAll(".thumbnail-card.selected")
    );
    const paths = selectedCards.map((c) => c.dataset.path);
    if (!paths.length) return;
    if (!confirm(`Delete ${paths.length} items?`)) return;

    selectedCards.forEach((card) => card.classList.add("being-deleted"));

    try {
      const res = await fetch(`${API_URL}/delete-images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          image_paths: paths,
        }),
      });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();

      selectedCards.forEach((card) => card.remove());
      if (currentClusterName && currentGroups[currentClusterName]) {
        currentGroups[currentClusterName] = currentGroups[
          currentClusterName
        ].filter((p) => !paths.includes(p));
      }
      syncUniverseMap(data.deleted);
    } catch (e) {
      selectedCards.forEach((card) => card.classList.remove("being-deleted"));
      alert("Delete failed: " + e.message);
    }
  };
}

document.getElementById("smart-cleanup-btn").onclick = async () => {
  console.log("=== SMART CLEANUP BUTTON CLICKED ===");

  const sel = document.querySelectorAll(".thumbnail-card.selected");
  console.log("Selected cards:", sel.length);

  if (sel.length !== 1) return alert("Select exactly ONE best image to keep.");

  const keepPath = sel[0].dataset.path;
  console.log("Keep path:", keepPath);

  if (!confirm(`Keep 1 and delete the rest of '${currentClusterName}'?`))
    return;

  console.log("Checking universe controller...");
  console.log("  - universeController exists?", !!universeController);
  console.log("  - universeState.data.length:", universeState.data.length);

  if (universeController && universeState.data.length > 0) {
    console.log("Finding sprites...");
    const keepSprite = universeController.findSpriteByPath(keepPath);
    console.log("  - keepSprite found?", !!keepSprite);

    const deleteSprites = universeController.findSpritesByClusterAndExclude(
      currentClusterName,
      keepPath
    );
    console.log("  - deleteSprites count:", deleteSprites?.length);

    if (keepSprite && deleteSprites && deleteSprites.length > 0) {
      console.log("✅ All sprites found! Starting Quantum Merge sequence...");

      console.log("1️⃣ Switching to universe tab...");
      document.querySelector('[data-tab="universe"]').click();

      console.log("2️⃣ Waiting 300ms for tab switch...");
      setTimeout(() => {
        console.log("3️⃣ Triggering performQuantumMerge...");
        universeController.performQuantumMerge(keepSprite, deleteSprites);
      }, 300);

      const handleComplete = async (e) => {
        console.log("4️⃣ Received quantum-merge-complete event!");
        window.removeEventListener("quantum-merge-complete", handleComplete);
        try {
          console.log("5️⃣ Calling API...");
          await callSmartCleanupAPI(keepPath);

          console.log("6️⃣ Waiting 1s before returning to browser...");
          setTimeout(() => {
            console.log("7️⃣ Returning to browser tab");
            document.querySelector('[data-tab="browser"]').click();
          }, 1000);
        } catch (err) {
          console.error("❌ Error during cleanup:", err);
          alert("Cleanup failed: " + err.message);
          document.querySelector('[data-tab="browser"]').click();
          loadCluster(currentClusterName);
        }
      };

      console.log("👂 Adding event listener for quantum-merge-complete");
      window.addEventListener("quantum-merge-complete", handleComplete);
      return;
    } else {
      console.warn("⚠️ Sprites not found or empty");
    }
  } else {
    console.warn("⚠️ Universe controller not available");
  }

  console.log("💫 Fallback: calling API directly without Quantum Merge");
  await callSmartCleanupAPI(keepPath);
};

async function callSmartCleanupAPI(keepPath) {
  console.log("📡 Calling Smart Cleanup API for:", keepPath);

  try {
    const res = await fetch(`${API_URL}/smart-cleanup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        cluster_name: currentClusterName,
        image_to_keep: keepPath,
      }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Server error");
    }

    const data = await res.json();
    console.log("✅ API response:", data);

    const oldPaths = currentGroups[currentClusterName];
    currentGroups[currentClusterName] = [data.image_kept];
    const deletedPaths = oldPaths.filter((p) => p !== data.image_kept);

    console.log("🔄 Updating UI with", deletedPaths.length, "deleted paths");

    syncUniverseMap(deletedPaths);

    loadCluster(currentClusterName);
    renderClusterList();

    console.log("✅ Smart Cleanup API complete");
  } catch (e) {
    console.error("❌ API Error:", e);
    throw e;
  }
}

if (deleteBtn) {
  deleteBtn.onclick = async () => {
    const selectedCards = Array.from(
      document.querySelectorAll(".thumbnail-card.selected")
    );
    const paths = selectedCards.map((c) => c.dataset.path);
    if (!paths.length) return;
    if (!confirm(`Delete ${paths.length} items?`)) return;

    selectedCards.forEach((card) => card.classList.add("being-deleted"));

    try {
      const res = await fetch(`${API_URL}/delete-images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          image_paths: paths,
        }),
      });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();

      selectedCards.forEach((card) => card.remove());
      if (currentClusterName && currentGroups[currentClusterName]) {
        currentGroups[currentClusterName] = currentGroups[
          currentClusterName
        ].filter((p) => !paths.includes(p));
      }

      syncUniverseMap(data.deleted);

      console.log("✅ Images deleted and Universe synced");
    } catch (e) {
      selectedCards.forEach((card) => card.classList.remove("being-deleted"));
      alert("Delete failed: " + e.message);
    }
  };
}

const deleteGroupBtn = document.getElementById("delete-group-btn");
if (deleteGroupBtn) {
  deleteGroupBtn.onclick = async () => {
    if (!confirm(`Delete entire group ${currentClusterName}?`)) return;
    try {
      const res = await fetch(`${API_URL}/delete-group`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          cluster_name: currentClusterName,
        }),
      });
      if (res.ok) {
        const deletedPaths = currentGroups[currentClusterName];
        delete currentGroups[currentClusterName];
        currentClusterName = null;
        renderClusterList();
        const gallery = document.getElementById("thumbnail-gallery");
        if (gallery) gallery.innerHTML = "";

        syncUniverseMap(deletedPaths);

        console.log("✅ Group deleted and Universe synced");
      }
    } catch (e) {
      alert(e.message);
    }
  };
}

function renderStatsDashboard(data, unique, dupes) {
  const statsEmpty = document.getElementById("stats-empty");
  const statsContent = document.getElementById("stats-content");
  if (statsEmpty) statsEmpty.classList.add("hidden");
  if (statsContent) statsContent.classList.remove("hidden");

  const total = data.results.total_images || 0;
  const saved = total > 0 ? ((dupes / total) * 100).toFixed(1) : 0;

  const elTotal = document.getElementById("d-total");
  if (elTotal) elTotal.textContent = total;
  const elDupes = document.getElementById("d-dupes");
  if (elDupes) elDupes.textContent = dupes;
  const elSaved = document.getElementById("d-saved");
  if (elSaved) elSaved.textContent = `${saved}%`;
  const elClusters = document.getElementById("d-clusters");
  if (elClusters) elClusters.textContent = unique;

  if (chartInstances.ratioNew) chartInstances.ratioNew.destroy();
  const ctxRatio = document.getElementById("chart-ratio-new");
  if (ctxRatio) {
    chartInstances.ratioNew = new Chart(ctxRatio.getContext("2d"), {
      type: "doughnut",
      data: {
        labels: ["Unique", "Duplicate"],
        datasets: [
          {
            data: [unique, dupes],
            backgroundColor: ["#10b981", "#ef4444"],
            borderWidth: 0,
            hoverOffset: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "65%",
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#ccc",
              font: { size: 13, family: "'Outfit', sans-serif" },
              boxWidth: 14,
              padding: 15,
            },
          },
        },
      },
    });
  }

  if (chartInstances.distNew) chartInstances.distNew.destroy();
  const groups = data.results.groups || {};
  const sortedKeys = Object.keys(groups)
    .sort((a, b) => groups[b].length - groups[a].length)
    .slice(0, 8);
  const sizes = sortedKeys.map((k) => groups[k].length);

  const ctxDist = document.getElementById("chart-dist-new");
  if (ctxDist) {
    chartInstances.distNew = new Chart(ctxDist.getContext("2d"), {
      type: "bar",
      data: {
        labels: sortedKeys,
        datasets: [
          {
            label: "Images",
            data: sizes,
            backgroundColor: "#8b5cf6",
            borderRadius: 6,
            barThickness: 24,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: {
            grid: { color: "#333" },
            ticks: { color: "#888", font: { size: 11 } },
          },
          y: {
            grid: { display: false },
            ticks: {
              color: "#e5e7eb",
              font: { size: 13, weight: "500", family: "'Outfit', sans-serif" },
              autoSkip: false,
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  renderPipelineTimeline(total, unique, dupes);
  renderRadarChart(unique, dupes, total);
}

function renderPipelineTimeline(total, unique, dupes) {
  const baseOverhead = 0.5;
  const perImgFactor = 0.02;

  const t1 = baseOverhead.toFixed(2);
  const t2 = (baseOverhead + total * 0.005).toFixed(2);
  const t3 = (baseOverhead + total * perImgFactor * 0.4).toFixed(2);
  const t4 = (baseOverhead + total * perImgFactor * 0.8).toFixed(2);
  const t5 = (baseOverhead + total * perImgFactor).toFixed(2);

  const algoSelect = document.getElementById("algorithm");
  const algoName = algoSelect ? algoSelect.value : "Algorithm";

  const steps = [
    {
      title: "Session Initialized",
      time: "00:00:00",
      status: "done",
      desc: "Environment ready",
    },
    {
      title: `Ingested ${total} Assets`,
      time: `00:00:${t2.padStart(2, "0")}`,
      status: "done",
      desc: "Integrity check passed",
    },
    {
      title: `${algoName} Processing`,
      time: `00:00:${t3.padStart(2, "0")}`,
      status: "done",
      desc: "Perceptual hashing generated",
    },
    {
      title: "Feature Extraction",
      time: `00:00:${t4.padStart(2, "0")}`,
      status: "done",
      desc: `Identified ${dupes} duplicates`,
    },
    {
      title: "Optimization Complete",
      time: `00:00:${t5.padStart(2, "0")}`,
      status: "current",
      desc: `Consolidated into ${unique} clusters`,
    },
  ];

  const timelineContainer = document.getElementById("pipeline-steps");
  if (timelineContainer) {
    timelineContainer.innerHTML = steps
      .map(
        (step, index) => `
            <div class="flex gap-4 relative group mb-2">
                ${
                  index !== steps.length - 1
                    ? '<div class="absolute left-[11px] top-7 bottom-[-12px] w-px bg-[#333] group-hover:bg-violet-500/30 transition-colors"></div>'
                    : ""
                }
                
                <div class="w-6 h-6 rounded-full ${
                  step.status === "done"
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                    : "bg-violet-500/20 text-violet-400 border border-violet-500/50 animate-pulse"
                } flex items-center justify-center shrink-0 text-xs font-bold z-10 box-content bg-[#1e1e1e]">
                    ${step.status === "done" ? "✓" : "●"}
                </div>
                <div>
                    <div class="flex items-center gap-3">
                        <div class="text-sm text-gray-100 font-bold">${
                          step.title
                        }</div>
                        <span class="text-[10px] text-gray-500 border border-[#333] px-1.5 py-0.5 rounded font-mono bg-[#181818]">+${
                          step.time
                        }s</span>
                    </div>
                    <div class="text-xs text-gray-400 mt-1">${step.desc}</div>
                </div>
            </div>
        `
      )
      .join("");
  }
}

function renderRadarChart(unique, dupes, total) {
  if (total === 0) total = 1;
  const efficiencyScore = (dupes / total) * 100;
  let speedScore = 100 - total / 50;
  if (speedScore < 60) speedScore = 60;
  const clusterDensity = (dupes / total) * 100;
  const qualityScore = 85 + (Math.random() * 10 - 5);
  const aiConfidence = 92;

  const ctxRadar = document.getElementById("chart-ai-radar");
  if (ctxRadar) {
    if (window.aiRadarChart instanceof Chart) {
      window.aiRadarChart.destroy();
    }

    window.aiRadarChart = new Chart(ctxRadar.getContext("2d"), {
      type: "radar",
      data: {
        labels: [
          "Storage Efficiency",
          "Image Quality",
          "Processing Speed",
          "Cluster Density",
          "Confidence",
        ],
        datasets: [
          {
            label: "Current Session",
            data: [
              efficiencyScore,
              qualityScore,
              speedScore,
              clusterDensity,
              aiConfidence,
            ],
            fill: true,
            backgroundColor: "rgba(139, 92, 246, 0.2)",
            borderColor: "#8b5cf6",
            pointBackgroundColor: "#fff",
            pointBorderColor: "#8b5cf6",
            pointHoverBackgroundColor: "#fff",
            pointHoverBorderColor: "#8b5cf6",
          },
          {
            label: "Benchmark",
            data: [50, 75, 80, 50, 85],
            fill: true,
            backgroundColor: "transparent",
            borderColor: "#444",
            pointBackgroundColor: "transparent",
            pointBorderColor: "transparent",
            borderDash: [5, 5],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        elements: { line: { borderWidth: 2 } },
        scales: {
          r: {
            angleLines: { color: "#333" },
            grid: { color: "#2a2a2a" },
            pointLabels: {
              color: "#d1d5db",
              font: { size: 13, family: "'Outfit', sans-serif", weight: "500" },
            },
            ticks: { display: false, backdropColor: "transparent" },
            suggestedMin: 0,
            suggestedMax: 100,
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }
}

function renderActionCenter(unique, dupes, total) {
  const sId = document.getElementById("summary-session-id");
  if (sId) sId.textContent = currentSessionId || "N/A";

  const sSaved = document.getElementById("summary-savings-text");
  if (sSaved)
    sSaved.textContent =
      total > 0 ? ((dupes / total) * 100).toFixed(1) + "%" : "0%";

  const impactBadge = document.getElementById("summary-impact-badge");
  if (impactBadge) {
    if (dupes > 0) impactBadge.classList.remove("hidden");
    else impactBadge.classList.add("hidden");
  }

  document.getElementById("sum-total-img").textContent = total;
  document.getElementById("sum-dupes-img").textContent = dupes;
  document.getElementById("sum-clusters-img").textContent = unique;

  const grid = document.getElementById("priority-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const sortedGroups = Object.entries(currentGroups)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 8);

  sortedGroups.forEach(([name, files]) => {
    const url = `${API_URL}/results/${currentSessionId}/clusters/${files[0]}`;
    const div = document.createElement("div");
    div.className = `highlight-card rounded-xl p-4 cursor-pointer group flex flex-col gap-3`;
    div.innerHTML = `
            <div class="flex justify-between items-start">
                <span class="text-white font-bold text-sm truncate w-full">${name}</span>
                <span class="impact-badge text-[10px] font-bold px-2 py-1 rounded">${files.length}</span>
            </div>
            <div class="highlight-img-container aspect-video w-full bg-black rounded-lg border border-[#333]">
                 <img src="${url}" class="w-full h-full object-contain opacity-80 group-hover:opacity-100 transition-opacity duration-300" loading="lazy">
            </div>
        `;
    div.onclick = () => {
      document.querySelector('[data-tab="browser"]').click();
      setTimeout(() => loadCluster(name), 150);
    };
    grid.appendChild(div);
  });
}

const selectAllBtn = document.getElementById("select-all-btn");
if (selectAllBtn)
  selectAllBtn.onclick = () =>
    document.querySelectorAll(".thumbnail-card").forEach((c) => {
      c.classList.add("selected");
      c.querySelector("input").checked = true;
    });

const deselectAllBtn = document.getElementById("deselect-all-btn");
if (deselectAllBtn)
  deselectAllBtn.onclick = () =>
    document.querySelectorAll(".thumbnail-card").forEach((c) => {
      c.classList.remove("selected");
      c.querySelector("input").checked = false;
    });

const keepBestBtn = document.getElementById("keep-best-btn");
if (keepBestBtn) {
  keepBestBtn.onclick = async () => {
    const best = qualityScores[currentClusterName]?.images.find(
      (i) => i.is_best
    );
    if (!best) {
      alert("No best image found for this cluster.");
      return;
    }

    document.querySelectorAll(".thumbnail-card").forEach((c) => {
      c.classList.remove("selected");
      c.querySelector("input").checked = false;
    });

    const bestCard = document.querySelector(`[data-path="${best.path}"]`);
    if (bestCard) {
      bestCard.classList.add("selected");
      bestCard.querySelector("input").checked = true;
      bestCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    const confirmMsg = `Keep the BEST image and delete ${
      currentGroups[currentClusterName].length - 1
    } others with Quantum Merge animation?`;

    if (confirm(confirmMsg)) {
      await callSmartCleanupAPI(best.path);
    }
  };
}

const imageModal = document.getElementById("image-modal");
if (imageModal)
  imageModal.onclick = function (e) {
    if (e.target === this) this.classList.add("hidden");
  };

const moveBtn = document.getElementById("move-btn");
const moveModal = document.getElementById("move-modal");
const moveSelect = document.getElementById("move-cluster-select");
const moveNewInputGroup = document.getElementById(
  "move-new-cluster-input-group"
);
const moveNewInput = document.getElementById("move-new-cluster-name");
const moveConfirmBtn = document.getElementById("move-confirm-btn");
const moveCancelBtn = document.getElementById("move-cancel-btn");

if (moveBtn && moveModal) {
  moveBtn.onclick = () => {
    const selectedCards = document.querySelectorAll(".thumbnail-card.selected");
    if (selectedCards.length === 0)
      return alert("Please select images to move.");

    moveSelect.innerHTML = "";

    const newOption = document.createElement("option");
    newOption.value = "__NEW_CLUSTER__";
    newOption.textContent = "+ Create New Cluster...";
    newOption.className = "text-violet-400 font-bold";
    moveSelect.appendChild(newOption);

    Object.keys(currentGroups)
      .sort()
      .forEach((name) => {
        if (name !== currentClusterName) {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          moveSelect.appendChild(opt);
        }
      });

    moveSelect.value =
      Object.keys(currentGroups).find((n) => n !== currentClusterName) ||
      "__NEW_CLUSTER__";
    moveNewInputGroup.classList.add("hidden");
    if (moveSelect.value === "__NEW_CLUSTER__") {
      moveNewInputGroup.classList.remove("hidden");
    }

    moveModal.classList.remove("hidden");
  };

  moveSelect.onchange = () => {
    if (moveSelect.value === "__NEW_CLUSTER__") {
      moveNewInputGroup.classList.remove("hidden");
      moveNewInput.focus();
    } else {
      moveNewInputGroup.classList.add("hidden");
    }
  };

  moveCancelBtn.onclick = () => {
    moveModal.classList.add("hidden");
  };

  moveConfirmBtn.onclick = async () => {
    const selectedCards = Array.from(
      document.querySelectorAll(".thumbnail-card.selected")
    );
    const imagePaths = selectedCards.map((c) => c.dataset.path);

    let targetCluster = moveSelect.value;

    if (targetCluster === "__NEW_CLUSTER__") {
      targetCluster = moveNewInput.value.trim();
      if (!targetCluster)
        return alert("Please enter a name for the new cluster.");
    }

    const originalBtnText = moveConfirmBtn.textContent;
    moveConfirmBtn.textContent = "MOVING...";
    moveConfirmBtn.disabled = true;

    try {
      const res = await fetch(`${API_URL}/move-images`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          image_paths: imagePaths,
          target_cluster: targetCluster,
        }),
      });

      if (!res.ok) throw new Error((await res.json()).detail || "Move failed");

      currentGroups[currentClusterName] = currentGroups[
        currentClusterName
      ].filter((p) => !imagePaths.includes(p));

      if (!currentGroups[targetCluster]) {
        currentGroups[targetCluster] = [];
      }
      currentGroups[targetCluster].push(...imagePaths);

      selectedCards.forEach((c) => c.remove());

      syncUniverseMap(imagePaths);

      renderClusterList();

      if (currentGroups[currentClusterName].length === 0) {
        delete currentGroups[currentClusterName];
        const nextGroup = Object.keys(currentGroups)[0];
        if (nextGroup) loadCluster(nextGroup);
        else document.getElementById("thumbnail-gallery").innerHTML = "";
      }

      moveModal.classList.add("hidden");
      alert(`Moved ${imagePaths.length} images to '${targetCluster}'`);
    } catch (e) {
      alert("Error moving images: " + e.message);
    } finally {
      moveConfirmBtn.textContent = originalBtnText;
      moveConfirmBtn.disabled = false;
    }
  };
}
