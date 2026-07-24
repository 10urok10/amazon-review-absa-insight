const BACKEND_URL = "http://127.0.0.1:5057";

const btn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

btn.addEventListener("click", async () => {
  btn.disabled = true;
  resultEl.innerHTML = "";
  statusEl.textContent = "Sayfa taranıyor (daha fazla yorum varsa otomatik yükleniyor)...";
  const startedAt = performance.now();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // scraper.js defines scrapeAmazonProduct() as a page-global; inject the
    // file first, then invoke it -- chrome.scripting.executeScript's `func`
    // form only serializes the one function passed to it, so it can't
    // reference other top-level functions unless they're already present
    // in the page from a prior file injection.
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["scraper.js"] });
    const [{ result: scraped }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => scrapeAmazonProduct(),
    });

    if (!scraped || !scraped.asin) {
      statusEl.textContent =
        "ASIN bulunamadı. Bir Amazon ürün sayfasında olduğundan emin ol.\n" +
        `Sayfa: ${tab.url}`;
      btn.disabled = false;
      return;
    }
    if (!scraped.reviews || scraped.reviews.length === 0) {
      statusEl.textContent = "Bu sayfada yorum bulunamadı. Sayfayı yorumlar bölümüne kadar kaydırıp tekrar dene.";
      btn.disabled = false;
      return;
    }

    statusEl.textContent = `${scraped.reviews.length} yorum bulundu (ASIN: ${scraped.asin}). Analiz ediliyor -- ilk analiz 20-60s sürebilir...`;

    const res = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scraped),
    });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = `Hata: ${data.error || res.statusText}`;
      btn.disabled = false;
      return;
    }

    const totalSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    const timingNote = data.from_cache
      ? `${totalSeconds}s`
      : `${totalSeconds}s toplam, backend: ${data.backend_seconds}s`;
    statusEl.textContent = data.from_cache
      ? `⚡ Cache'ten geldi (ilk analiz: ${data.created_at}) -- ${timingNote}`
      : `✅ Yeni analiz tamamlandı -- ${timingNote}`;
    renderInsightResult(resultEl, data);
  } catch (err) {
    statusEl.textContent = `Backend'e bağlanılamadı. extension_server.py çalışıyor mu? (${err})`;
  }

  btn.disabled = false;
});
