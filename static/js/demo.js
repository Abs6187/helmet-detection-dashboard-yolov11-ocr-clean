/**
 * static/js/demo.js
 * ==================
 * Demo Lab — Helmet Detection + Speed Estimation interactive demos.
 * Depends on: dashboard (app.css loaded), Bootstrap modal not required.
 */

/* ═══════════════════════════════════════════════════════════════
   Helmet Detection Demo
   Fetches /demo_samples → renders thumbnail buttons.
   Click → POST /demo_detect → shows detection results.
═══════════════════════════════════════════════════════════════ */
(async function initHelmetDemo() {
    const row               = document.getElementById("helmet-sample-row");
    const resultArea        = document.getElementById("helmet-demo-result");
    const resultImg         = document.getElementById("helmet-result-img");
    const resultPlaceholder = document.getElementById("helmet-result-img-placeholder");
    const modeEl            = document.getElementById("demo-mode");
    const violationEl       = document.getElementById("demo-violation");
    const statsEl           = document.getElementById("demo-stats");
    const ocrEl             = document.getElementById("demo-ocr");
    const detTbody          = document.getElementById("demo-det-tbody");

    if (!row) return;   // demo section not present on this page

    const CATEGORY_LABELS = {
        double_riding: "Double Riding",
        single_rider:  "Single Rider",
        triple_riding: "Triple Riding",
    };

    /* ── Load sample thumbnails ─────────────────────────────────────────── */

    async function loadSamples() {
        try {
            const r    = await fetch("/demo_samples");
            if (!r.ok) throw new Error("Failed to load samples");
            const data = await r.json();

            if (!Object.keys(data).length) {
                row.innerHTML = "<div class='demo-placeholder-text'>No sample images found in dataset_samples/</div>";
                return;
            }

            row.innerHTML = "";
            for (const [cat, files] of Object.entries(data)) {
                const group = document.createElement("div");
                group.className = "demo-sample-group";

                const label = document.createElement("div");
                label.className   = "demo-sample-group-label";
                label.textContent = CATEGORY_LABELS[cat] || cat.replace(/_/g, " ");
                group.appendChild(label);

                const thumbRow = document.createElement("div");
                thumbRow.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;";

                for (const filePath of files.slice(0, 4)) {
                    const btn     = document.createElement("button");
                    btn.className = "demo-thumb-btn";
                    btn.title     = `${cat}: ${filePath.split("/").pop()}`;
                    btn.dataset.samplePath = filePath;

                    const img     = document.createElement("img");
                    img.src       = "/" + filePath;
                    img.alt       = cat;
                    img.loading   = "lazy";

                    const overlay = document.createElement("div");
                    overlay.className = "demo-thumb-overlay";

                    btn.appendChild(img);
                    btn.appendChild(overlay);
                    btn.addEventListener("click", () => runDetection(btn, filePath));
                    thumbRow.appendChild(btn);
                }
                group.appendChild(thumbRow);
                row.appendChild(group);
            }
        } catch (e) {
            row.innerHTML = `<div class='demo-placeholder-text'>${e.message}</div>`;
        }
    }

    /* ── Run detection on a sample ──────────────────────────────────────── */

    async function runDetection(btn, samplePath) {
        document.querySelectorAll(".demo-thumb-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        // Reset result area to loading state
        resultArea.style.display          = "";
        resultImg.src                     = "";
        resultImg.style.display           = "none";
        resultPlaceholder.style.display   = "";
        resultPlaceholder.innerHTML       = '<div class="demo-spinner"></div><div>Running detection…</div>';
        modeEl.textContent      = "—";
        violationEl.textContent = "—";
        statsEl.textContent     = "—";
        ocrEl.textContent       = "—";
        detTbody.innerHTML      = "";

        resultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

        try {
            const resp = await fetch("/demo_detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_path: samplePath, extract_text: true }),
            });
            const data = await resp.json();

            if (data.error && !data.mode) throw new Error(data.message || data.error);

            // Mode badge
            modeEl.textContent = data.mode || "unknown";
            modeEl.className   = "demo-badge-sm" + (data.mode === "local" ? " local" : "");

            // Violation
            violationEl.innerHTML = data.has_violation
                ? "<span style='color:#a73b00;font-weight:700;'>Violation detected</span>"
                : "<span style='color:#007a5a;'>No violation</span>";

            statsEl.textContent = data.stats || "—";
            ocrEl.textContent   = data.ocr_text || "—";

            // Detections table
            const dets = data.detections || [];
            if (dets.length) {
                dets.forEach(d => {
                    const tr    = document.createElement("tr");
                    const label = d.label || d.Class || "Unknown";
                    const conf  = d.confidence !== undefined ? d.confidence : d.Confidence;
                    tr.innerHTML = `<td>${label}</td><td>${conf !== undefined ? Number(conf).toFixed(3) : "—"}</td>`;
                    detTbody.appendChild(tr);
                });
            } else {
                detTbody.innerHTML = "<tr><td colspan='2' style='color:var(--muted)'>No detections</td></tr>";
            }

            // Annotated image (remote) or original sample (local fallback)
            const imgSrc = data.annotated_image_url || "/" + samplePath;
            resultImg.src = imgSrc;
            resultImg.onload = () => {
                resultPlaceholder.style.display = "none";
                resultImg.style.display         = "block";
            };
            resultImg.onerror = () => {
                resultPlaceholder.innerHTML = "<div style='color:var(--muted);font-size:0.82rem;padding:16px'>Image not available</div>";
            };

        } catch (err) {
            resultPlaceholder.innerHTML = `<div style='color:#a73b00;font-size:0.82rem;padding:16px'>Error: ${err.message}</div>`;
        }
    }

    loadSamples();
})();


/* ═══════════════════════════════════════════════════════════════
   Speed Estimation Demo
   Fake cinematic loading (6.6 s) → reveals pre-rendered output video.
═══════════════════════════════════════════════════════════════ */
(function initSpeedDemo() {
    const btn       = document.getElementById("speed-demo-btn");
    const idleEl    = document.getElementById("speed-idle");
    const loadingEl = document.getElementById("speed-loading");
    const resultEl  = document.getElementById("speed-result");

    if (!btn) return;

    // Loading step definitions (index 0 starts immediately)
    const STEPS = [
        { id: "sstep-1", delay: 0    },
        { id: "sstep-2", delay: 1400 },
        { id: "sstep-3", delay: 2900 },
        { id: "sstep-4", delay: 4200 },
        { id: "sstep-5", delay: 5400 },
    ];

    // Representative stats from the pre-computed output video
    const DEMO_STATS = { total: 18, avg: 42, max: 71, in: 11, out: 7 };

    function showStep(idx) {
        STEPS.forEach((s, i) => {
            const el = document.getElementById(s.id);
            if (!el) return;
            el.className = "speed-step"
                + (i < idx  ? " speed-step-done"
                 : i === idx ? " speed-step-active"
                 : "");
        });
    }

    btn.addEventListener("click", function () {
        if (btn.disabled) return;
        btn.disabled    = true;
        btn.textContent = "Analysing…";

        idleEl.style.display    = "none";
        loadingEl.style.display = "";
        resultEl.style.display  = "none";
        showStep(0);

        // Advance steps on a staggered timeline
        STEPS.forEach((s, i) => {
            if (i === 0) return;
            setTimeout(() => showStep(i), s.delay);
        });

        // After ~6.6 s reveal the pre-rendered video + stats
        setTimeout(() => {
            loadingEl.style.display = "none";
            resultEl.style.display  = "";

            document.getElementById("sstat-total").textContent = DEMO_STATS.total;
            document.getElementById("sstat-avg").textContent   = DEMO_STATS.avg;
            document.getElementById("sstat-max").textContent   = DEMO_STATS.max;
            document.getElementById("sstat-in").textContent    = DEMO_STATS.in;
            document.getElementById("sstat-out").textContent   = DEMO_STATS.out;

            btn.disabled    = false;
            btn.textContent = "Run Again";
        }, 6600);
    });
})();
