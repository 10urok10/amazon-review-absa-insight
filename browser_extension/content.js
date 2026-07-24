/*
 * Auto-injected (via manifest content_scripts, alongside scraper.js and
 * render.js) into Amazon pages. Adds a floating "Bu Ürünü Analiz Et" button
 * so analysis doesn't require opening the toolbar popup, plus a small badge
 * that lights up automatically if this product is already cached (cheap
 * SQLite check, no GPU/LLM call). All network calls are relayed through
 * background.js -- see the comment there for why.
 */

(function () {
  const asin = getAsinFromPage();
  if (!asin) return; // not a product page -- don't show the button

  const host = document.createElement("div");
  host.id = "absa-insight-host";
  host.style.cssText = "position:fixed;bottom:0;right:0;z-index:2147483647;";
  document.documentElement.appendChild(host);
  const shadow = host.attachShadow({ mode: "open" });

  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
    .wrap { position: fixed; bottom: 20px; right: 20px; }
    .fab {
      width: 56px; height: 56px; border-radius: 50%;
      background: #d03b3b; color: white; border: none; cursor: pointer;
      box-shadow: 0 2px 10px rgba(0,0,0,0.35); font-size: 24px;
      display: flex; align-items: center; justify-content: center;
    }
    .fab.cached { background: #0ca30c; }
    .fab:disabled { background: #c3c2b7; cursor: not-allowed; }
    .badge {
      position: absolute; top: -2px; right: -2px;
      width: 16px; height: 16px; border-radius: 50%;
      background: #0ca30c; border: 2px solid white;
      display: none;
    }
    .badge.show { display: block; }
    .panel {
      position: fixed; bottom: 84px; right: 20px; width: 360px; max-height: 70vh;
      overflow-y: auto; background: #fcfcfb; border: 1px solid #e1e0d9;
      border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      padding: 14px; font-size: 13px; color: #0b0b0b; display: none;
    }
    .panel.show { display: block; }
    .panel-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 8px;
    }
    .panel-header b { font-size: 13px; }
    .close-btn {
      background: none; border: none; cursor: pointer; color: #52514e;
      font-size: 16px; line-height: 1; padding: 0 0 0 8px;
    }
    .close-btn:hover { color: #0b0b0b; }
    #panelStatus {
      font-size: 12px; color: #52514e; margin-bottom: 8px;
      white-space: pre-wrap; font-family: ui-monospace, monospace;
    }
    ${RESULT_PANEL_CSS}
  `;
  shadow.appendChild(style);

  const wrap = document.createElement("div");
  wrap.className = "wrap";
  wrap.innerHTML = `
    <div style="position:relative;">
      <button class="fab" id="fabBtn" title="Bu Ürünü Analiz Et">🔍</button>
      <div class="badge" id="fabBadge" title="Bu ürün daha önce analiz edildi"></div>
    </div>
    <div class="panel" id="panel">
      <div class="panel-header">
        <b>🔍 Ürün İçgörüsü</b>
        <button class="close-btn" id="closeBtn" title="Kapat">✕</button>
      </div>
      <div id="panelStatus"></div>
      <div id="panelResult"></div>
    </div>
  `;
  shadow.appendChild(wrap);

  const fabBtn = shadow.getElementById("fabBtn");
  const fabBadge = shadow.getElementById("fabBadge");
  const panel = shadow.getElementById("panel");
  const closeBtn = shadow.getElementById("closeBtn");
  const panelStatus = shadow.getElementById("panelStatus");
  const panelResult = shadow.getElementById("panelResult");

  closeBtn.addEventListener("click", () => panel.classList.remove("show"));

  chrome.runtime.sendMessage({ type: "checkCache", asin }, (resp) => {
    if (chrome.runtime.lastError) return; // backend/extension not reachable yet
    if (resp && resp.ok && resp.data && resp.data.cached) {
      fabBtn.classList.add("cached");
      fabBadge.classList.add("show");
    }
  });

  fabBtn.addEventListener("click", async () => {
    // Second click while the panel is already open just closes it, instead
    // of re-scraping and re-analyzing -- click again (or reopen via the
    // badge state) to re-run.
    if (panel.classList.contains("show")) {
      panel.classList.remove("show");
      return;
    }

    panel.classList.add("show");
    panelResult.innerHTML = "";
    fabBtn.disabled = true;
    panelStatus.textContent = "Sayfa taranıyor (daha fazla yorum varsa otomatik yükleniyor)...";
    const startedAt = performance.now();

    const scraped = await scrapeAmazonProduct();
    if (!scraped.asin) {
      panelStatus.textContent = "ASIN bulunamadı.";
      fabBtn.disabled = false;
      return;
    }
    if (!scraped.reviews.length) {
      panelStatus.textContent = "Bu sayfada yorum bulunamadı. Sayfayı yorumlar bölümüne kadar kaydırıp tekrar dene.";
      fabBtn.disabled = false;
      return;
    }
    panelStatus.textContent = `${scraped.reviews.length} yorum bulundu. Analiz ediliyor -- ilk analiz 20-60s sürebilir...`;

    chrome.runtime.sendMessage({ type: "analyze", payload: scraped }, (resp) => {
      fabBtn.disabled = false;
      if (chrome.runtime.lastError) {
        panelStatus.textContent = `Backend'e bağlanılamadı. (${chrome.runtime.lastError.message})`;
        return;
      }
      if (!resp || !resp.ok) {
        const errMsg = resp && resp.data ? resp.data.error : resp && resp.error;
        panelStatus.textContent = `Hata: ${errMsg || "bilinmiyor"}`;
        return;
      }
      const data = resp.data;
      const totalSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      panelStatus.textContent = data.from_cache
        ? `⚡ Cache'ten geldi (ilk analiz: ${data.created_at}) -- ${totalSeconds}s`
        : `✅ Yeni analiz tamamlandı -- ${totalSeconds}s toplam, backend: ${data.backend_seconds}s`;
      renderInsightResult(panelResult, data);
      fabBtn.classList.add("cached");
      fabBadge.classList.add("show");
    });
  });
})();
