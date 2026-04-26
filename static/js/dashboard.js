/**
 * static/js/dashboard.js
 * ========================
 * Dashboard interactivity: summary widgets, search/filter, and inline editing.
 */

const currencyFormatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
});

function normalize(value) {
    return String(value || "").trim().toLowerCase();
}

/* ── Summary metric widgets ─────────────────────────────────────────────── */

function updateSummaryWidgets() {
    const cards      = [...document.querySelectorAll(".offender-item")];
    const total      = cards.length;
    const applied    = cards.filter(c => c.dataset.cardFineApplied === "true").length;
    const pending    = total - applied;
    const fineTotal  = cards.reduce((sum, c) => {
        const node = c.querySelector("[data-card-fine]");
        const fine = parseInt(node ? node.textContent : "0", 10);
        return sum + (Number.isFinite(fine) ? fine : 0);
    }, 0);

    document.querySelector("[data-summary='total_cases']").textContent    = total;
    document.querySelector("[data-summary='fine_applied']").textContent   = applied;
    document.querySelector("[data-summary='fine_pending']").textContent   = pending;
    document.querySelector("[data-summary='total_fine_amount']").textContent =
        currencyFormatter.format(fineTotal);

    document.querySelectorAll("[data-category-total]").forEach(node => {
        const cat      = node.dataset.categoryTotal;
        const catCards = cards.filter(c => c.dataset.cardCategory === cat);
        const catAppl  = catCards.filter(c => c.dataset.cardFineApplied === "true").length;
        node.textContent = String(catCards.length);
        document.querySelector(`[data-category-applied='${cat}']`).textContent = String(catAppl);
        document.querySelector(`[data-category-pending='${cat}']`).textContent = String(catCards.length - catAppl);
    });
}

/* ── Server update ──────────────────────────────────────────────────────── */

async function postUpdate(category, folder, field, value) {
    const resp = await fetch("/update_offender", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, folder, field, value }),
    });
    const payload = await resp.json();
    if (!resp.ok || payload.status !== "success") {
        throw new Error(payload.message || "Update failed.");
    }
    return payload.record;
}

/* ── Card state helpers ─────────────────────────────────────────────────── */

function setCardFineState(card, fineApplied) {
    card.dataset.cardFineApplied = fineApplied ? "true" : "false";
    const node = card.querySelector("[data-card-status]");
    if (!node) return;
    node.textContent = fineApplied ? "Fine Applied" : "Pending Fine";
    node.classList.toggle("status-applied", fineApplied);
    node.classList.toggle("status-pending", !fineApplied);
}

/* ── Editable field handler ─────────────────────────────────────────────── */

async function handleEditableChange(event) {
    const input    = event.target;
    const { category, folder, field } = input.dataset;
    const value    = input.type === "checkbox" ? input.checked : input.value;
    const prevDisabled = input.disabled;
    input.disabled = true;

    try {
        const record   = await postUpdate(category, folder, field, value);
        const selector = `.offender-item[data-card-category="${category}"][data-card-folder="${folder}"]`;
        const card     = document.querySelector(selector);
        if (!card) { updateSummaryWidgets(); return; }

        if (field === "name") {
            card.dataset.cardName = normalize(record.name);
            card.querySelector("[data-card-name-text]").textContent = record.name || "Unnamed Offender";
        }
        if (field === "number_plate") {
            card.dataset.cardPlate = normalize(record.number_plate);
            card.querySelector("[data-card-plate-text]").textContent = record.number_plate || "Not set";
        }
        if (field === "location") {
            card.dataset.cardLocation = normalize(record.location);
            card.querySelector("[data-card-location-text]").textContent = record.location || "Unknown";
        }
        if (field === "fine") {
            card.querySelector("[data-card-fine]").textContent = String(record.fine || 0);
        }
        if (field === "fine_applied") {
            setCardFineState(card, Boolean(record.fine_applied));
        }
        updateSummaryWidgets();
        applyCategoryFilters(category);
    } catch (err) {
        window.alert(err.message);
        if (input.type === "checkbox") input.checked = !input.checked;
    } finally {
        input.disabled = prevDisabled;
    }
}

/* ── Search / filter ────────────────────────────────────────────────────── */

function applyCategoryFilters(category) {
    const searchNode  = document.querySelector(`[data-search-category="${category}"]`);
    const pendingNode = document.querySelector(`[data-pending-category="${category}"]`);
    const searchTerm  = normalize(searchNode ? searchNode.value : "");
    const pendingOnly = Boolean(pendingNode && pendingNode.checked);

    document.querySelectorAll(`.offender-item[data-card-category="${category}"]`).forEach(card => {
        const haystack = [
            card.dataset.cardName,
            card.dataset.cardPlate,
            card.dataset.cardLocation,
            normalize(card.dataset.cardFolder),
        ].join(" ");
        const matchSearch  = searchTerm.length === 0 || haystack.includes(searchTerm);
        const matchPending = !pendingOnly || card.dataset.cardFineApplied === "false";
        card.hidden = !(matchSearch && matchPending);
    });
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */

document.querySelectorAll(".editable-field").forEach(el =>
    el.addEventListener("change", handleEditableChange));
document.querySelectorAll(".editable-checkbox").forEach(el =>
    el.addEventListener("change", handleEditableChange));
document.querySelectorAll("[data-search-category]").forEach(el =>
    el.addEventListener("input", e => applyCategoryFilters(e.target.dataset.searchCategory)));
document.querySelectorAll("[data-pending-category]").forEach(el =>
    el.addEventListener("change", e => applyCategoryFilters(e.target.dataset.pendingCategory)));

updateSummaryWidgets();
