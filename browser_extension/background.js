/*
 * Service worker. Content scripts run inside the Amazon page's own origin,
 * so a fetch() they issue directly is still subject to Amazon's page CSP
 * (connect-src) and can be blocked even though the extension itself has
 * host_permissions for the backend. Routing the actual network calls
 * through this background worker sidesteps that -- it runs in the
 * extension's own context, not the page's.
 */

const BACKEND_URL = "http://127.0.0.1:5057";

async function checkCache(asin) {
  const res = await fetch(`${BACKEND_URL}/cache_status/${encodeURIComponent(asin)}`);
  return res.json();
}

async function analyze(payload) {
  const res = await fetch(`${BACKEND_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  return { httpOk: res.ok, data };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "checkCache") {
    checkCache(message.asin)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async response
  }
  if (message.type === "analyze") {
    analyze(message.payload)
      .then(({ httpOk, data }) => sendResponse({ ok: httpOk, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
  return false;
});
