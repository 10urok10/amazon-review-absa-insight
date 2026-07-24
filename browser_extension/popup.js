const BACKEND_URL = "http://127.0.0.1:5057";

const btn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

// Injected into the active tab via chrome.scripting.executeScript -- runs in
// the page's own context. Only interacts with elements already on the page
// (including clicking Amazon's own "show more reviews" button, exactly what
// a human reading the page would click) -- no requests are made to any
// server other than what the button click itself triggers.
async function scrapeAmazonProduct() {
  const MAX_SHOW_MORE_CLICKS = 10; // bounds total reviews fetched & clicks per analysis

  function countReviewBodies() {
    return document.querySelectorAll('span[data-hook="review-body"]').length;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function waitForGrowth(previousCount, timeoutMs = 4000, intervalMs = 300) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      await sleep(intervalMs);
      if (countReviewBodies() > previousCount) return true;
    }
    return false;
  }

  // Click Amazon's "show N more reviews" button repeatedly (paced, capped)
  // so multi-page review lists get read without the user clicking manually.
  for (let clicks = 0; clicks < MAX_SHOW_MORE_CLICKS; clicks++) {
    const moreBtn = document.querySelector('a[data-hook="show-more-button"], button[data-hook="show-more-button"]');
    if (!moreBtn) break;
    const before = countReviewBodies();
    moreBtn.click();
    const grew = await waitForGrowth(before);
    if (!grew) break;
  }

  let asin = null;

  // 1. URL path, e.g. /dp/B0XXXXXXXX/ or /gp/product/B0XXXXXXXX/
  const urlMatch = window.location.pathname.match(/\/(?:dp|gp\/product|product-reviews)\/([A-Z0-9]{10})(?:[/?]|$)/i);
  if (urlMatch) asin = urlMatch[1].toUpperCase();

  // 2. Hidden form field some product pages carry
  if (!asin) {
    const hidden = document.querySelector("input#ASIN, input[name='ASIN']");
    if (hidden && hidden.value) asin = hidden.value.toUpperCase();
  }

  // 3. data-asin attribute -- present on most Amazon product-page containers
  //    regardless of URL shape (works across regional Amazon TLDs too)
  if (!asin) {
    const withDataAsin = document.querySelector("[data-asin]:not([data-asin=''])");
    if (withDataAsin) asin = withDataAsin.getAttribute("data-asin").toUpperCase();
  }

  // 4. Fallback: scan the whole URL (not just pathname) for a 10-char ASIN token
  if (!asin) {
    const wideMatch = window.location.href.match(/\/([A-Z0-9]{10})(?:[/?]|$)/i);
    if (wideMatch) asin = wideMatch[1].toUpperCase();
  }

  const titleEl = document.querySelector("#productTitle");
  const title = titleEl ? titleEl.textContent.trim() : document.title.trim();

  // Look for review-body spans directly -- newer Amazon review UIs (e.g. the
  // "/portal/customer-reviews/" page) keep the data-hook on the body text but
  // wrap it in <li data-hook="review"> instead of the classic <div>, so an
  // attribute selector anchored to a specific tag name (div[data-hook=...])
  // never matches. Searching for the body span directly sidesteps that.
  const RATING_ICON_SELECTOR =
    'i[data-hook="review-star-rating"], i[data-hook="cmps-review-star-rating"], i[class*="a-star-"]';

  function findNearbyRatingIcon(bodyEl) {
    let el = bodyEl;
    for (let i = 0; i < 8 && el; i++) {
      el = el.parentElement;
      if (!el) break;
      const iconEl = el.querySelector(RATING_ICON_SELECTOR);
      if (iconEl) return iconEl;
    }
    return null;
  }

  function extractRating(iconEl) {
    if (!iconEl) return null;
    // Prefer the a-star-N / a-star-N-M CSS class -- locale-independent,
    // unlike the alt text ("5.0 out of 5 stars" in English puts the rating
    // first, but "5 yıldız üzerinden 5,0" in Turkish puts it last).
    const classMatch = iconEl.className.match(/a-star-(\d+)(?:-(\d+))?/);
    if (classMatch) {
      return classMatch[2] ? parseFloat(`${classMatch[1]}.${classMatch[2]}`) : parseFloat(classMatch[1]);
    }
    const altEl = iconEl.querySelector("span.a-icon-alt") || iconEl;
    const text = altEl.textContent || "";
    const nums = text.match(/\d+[.,]\d+|\d+/g);
    if (!nums || !nums.length) return null;
    const chosen = /üzerinden/i.test(text) ? nums[nums.length - 1] : nums[0];
    return parseFloat(chosen.replace(",", "."));
  }

  const bodyEls = document.querySelectorAll('span[data-hook="review-body"]');
  const reviews = [];
  bodyEls.forEach((bodyEl) => {
    const text = bodyEl.textContent.trim();
    if (!text) return;
    const rating = extractRating(findNearbyRatingIcon(bodyEl));
    reviews.push({ text, rating });
  });

  return { asin, title, reviews, url: window.location.href };
}

function renderResult(data) {
  const entries = Object.entries(data.aspect_stats || {})
    .map(([aspect, c]) => [aspect, c, c.Positive + c.Neutral + c.Negative])
    .sort((a, b) => b[2] - a[2])
    .slice(0, 10);

  const maxTotal = entries.length ? Math.max(...entries.map((e) => e[2])) : 1;

  const bars = entries
    .map(([aspect, c, total]) => {
      const barWidthPct = total > 0 ? (total / maxTotal) * 100 : 0;
      const segments = [
        ["pos", c.Positive],
        ["neu", c.Neutral],
        ["neg", c.Negative],
      ]
        .filter(([, n]) => n > 0)
        .map(([cls, n]) => `<div class="bar-segment ${cls}" style="width:${(n / total) * 100}%"></div>`)
        .join("");
      return `
        <div class="aspect-item">
          <div class="aspect-label" title="${aspect}">${aspect}</div>
          <div class="aspect-track"><div class="aspect-bar" style="width:${barWidthPct}%">${segments}</div></div>
          <div class="aspect-total">${total}</div>
        </div>`;
    })
    .join("");

  const legend = entries.length
    ? `<div class="chart-legend">
        <span><span class="legend-dot" style="background:#0ca30c"></span>Positive</span>
        <span><span class="legend-dot" style="background:#898781"></span>Neutral</span>
        <span><span class="legend-dot" style="background:#d03b3b"></span>Negative</span>
      </div>`
    : "";

  resultEl.innerHTML = `
    <div><b>${data.title || data.asin}</b></div>
    <div class="metrics">
      <div class="metric"><b>${data.avg_rating ? data.avg_rating.toFixed(2) : "N/A"}</b><span>/ 5 puan</span></div>
      <div class="metric"><b>${data.review_count}</b><span>yorum analiz edildi</span></div>
    </div>
    <div class="insight">🇹🇷 ${data.insight_text_tr || "(çeviri yok)"}</div>
    <div class="insight">🇬🇧 ${data.insight_text}</div>
    ${legend}
    <div>${bars}</div>
  `;
}

btn.addEventListener("click", async () => {
  btn.disabled = true;
  resultEl.innerHTML = "";
  statusEl.textContent = "Sayfa taranıyor (daha fazla yorum varsa otomatik yükleniyor)...";
  const startedAt = performance.now();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result: scraped }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: scrapeAmazonProduct,
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
    renderResult(data);
  } catch (err) {
    statusEl.textContent = `Backend'e bağlanılamadı. extension_server.py çalışıyor mu? (${err})`;
  }

  btn.disabled = false;
});
