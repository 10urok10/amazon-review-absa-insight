/*
 * Shared between popup.js (renders into popup.html's own document, which
 * already carries this CSS in a <style> block) and content.js (renders into
 * a Shadow DOM panel injected into the Amazon page, which needs its own
 * copy of the CSS since Shadow DOM isolates styles from the host page).
 */

const RESULT_PANEL_CSS = `
  .metrics { display: flex; gap: 20px; margin: 8px 0 12px 0; }
  .metric b { font-size: 20px; display: block; }
  .metric span { font-size: 11px; color: #52514e; }
  .insight {
    background: #f9f9f7;
    border: 1px solid #e1e0d9;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
  }
  .chart-legend {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: #52514e;
    margin: 10px 0 6px 0;
  }
  .chart-legend span { display: inline-flex; align-items: center; gap: 4px; }
  .legend-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
  .aspect-item { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
  .aspect-item.clickable { cursor: pointer; border-radius: 4px; }
  .aspect-item.clickable:hover { background: rgba(0, 0, 0, 0.05); }
  .aspect-label {
    width: 76px;
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .aspect-track {
    flex: 1;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    background: #e1e0d9;
  }
  .aspect-bar { height: 100%; display: flex; }
  .bar-segment { height: 100%; }
  .bar-segment.pos { background: #0ca30c; }
  .bar-segment.neu { background: #898781; }
  .bar-segment.neg { background: #d03b3b; }
  .bar-segment + .bar-segment { border-left: 1px solid #fcfcfb; }
  .aspect-total {
    width: 18px;
    font-size: 11px;
    color: #52514e;
    text-align: right;
    flex-shrink: 0;
  }
`;

// onAspectClick(aspect, searchTerms), if provided, is called when an aspect
// bar is clicked -- used by content.js to scroll to matching reviews on the
// page. Left undefined for the popup, where there's no visible page to
// scroll to, so bars there stay non-interactive.
function renderInsightResult(container, data, onAspectClick) {
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
      const clickable = onAspectClick ? "clickable" : "";
      return `
        <div class="aspect-item ${clickable}" data-aspect="${aspect}">
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

  container.innerHTML = `
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

  if (onAspectClick) {
    container.querySelectorAll(".aspect-item[data-aspect]").forEach((el) => {
      el.addEventListener("click", () => {
        const aspect = el.getAttribute("data-aspect");
        const terms = (data.aspect_search_terms && data.aspect_search_terms[aspect]) || [aspect];
        onAspectClick(aspect, terms);
      });
    });
  }
}
