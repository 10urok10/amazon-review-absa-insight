/*
 * Shared between popup.js (injected into the active tab on demand via
 * chrome.scripting.executeScript) and content.js (loaded automatically as
 * a content script). Reads only what's already rendered on the page,
 * including clicking Amazon's own "show more reviews" button -- exactly
 * what a human reading the page would click. No requests of its own.
 */

function getAsinFromPage() {
  let asin = null;

  const urlMatch = window.location.pathname.match(/\/(?:dp|gp\/product|product-reviews)\/([A-Z0-9]{10})(?:[/?]|$)/i);
  if (urlMatch) asin = urlMatch[1].toUpperCase();

  if (!asin) {
    const hidden = document.querySelector("input#ASIN, input[name='ASIN']");
    if (hidden && hidden.value) asin = hidden.value.toUpperCase();
  }

  // data-asin attribute -- present on most Amazon product-page containers
  // regardless of URL shape (works across regional Amazon TLDs too)
  if (!asin) {
    const withDataAsin = document.querySelector("[data-asin]:not([data-asin=''])");
    if (withDataAsin) asin = withDataAsin.getAttribute("data-asin").toUpperCase();
  }

  if (!asin) {
    const wideMatch = window.location.href.match(/\/([A-Z0-9]{10})(?:[/?]|$)/i);
    if (wideMatch) asin = wideMatch[1].toUpperCase();
  }

  return asin;
}

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

  const asin = getAsinFromPage();
  const titleEl = document.querySelector("#productTitle");
  const title = titleEl ? titleEl.textContent.trim() : document.title.trim();

  // Newer Amazon review UIs (e.g. the "/portal/customer-reviews/" page) keep
  // data-hook="review-body" on the text span but drop the classic enclosing
  // div[data-hook="review"] wrapper (it's an <li> now, or absent) -- so we
  // select the body span directly and walk up parents for the rating icon,
  // rather than anchoring to an outer review container.
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

// Scrolls to and briefly highlights the reviews on the page whose text
// mentions any of `terms` (case-insensitive substring match) -- `terms`
// should include the aspect's canonical name plus any pre-synonym-map raw
// terms folded into it server-side (see aspect_search_terms in the
// /analyze response), since a review's raw text may use the pre-mapping
// word (e.g. "cable") rather than the canonical aspect ("charger").
function scrollToAspectMentions(terms) {
  const lowerTerms = terms.map((t) => t.toLowerCase());
  const matches = Array.from(document.querySelectorAll('span[data-hook="review-body"]')).filter((el) => {
    const text = el.textContent.toLowerCase();
    return lowerTerms.some((t) => text.includes(t));
  });

  matches.forEach((el) => {
    const original = el.style.backgroundColor;
    el.style.transition = "background-color 0.3s";
    el.style.backgroundColor = "#fff3a3";
    setTimeout(() => {
      el.style.backgroundColor = original;
    }, 2500);
  });

  if (matches.length) {
    matches[0].scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return { found: matches.length };
}
