"""
Streamlit UI for the on-demand ABSA product insight tool.

Search a product by title, pick it from the results, and get a natural-
language insight into what reviewers actually praise and complain about --
backed by pyabsa ATEPC (aspect extraction + sentiment) and Gemini (insight
generation), with a SQLite cache so repeat lookups are instant.

Run:
    streamlit run app.py
"""

import sys

import altair as alt
import pandas as pd
import streamlit as st

from config import DEVICE, SENTIMENT_COLORS, SENTIMENT_ORDER
from insight_engine import get_or_compute_insight, search_products

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

st.set_page_config(page_title="Amazon Ürün İçgörü Aracı", page_icon="🔍", layout="centered")

st.title("🔍 Amazon Ürün İçgörü Aracı")
st.caption(
    "Bir ürün ara, gerçek yorumlardan çıkarılmış aspect-bazlı olumlu/olumsuz "
    "noktaları ve yıldız puanıyla çelişen gizli sorunları gör."
)

query = st.text_input(
    "Ürün adı ara",
    placeholder="örn. laptop charger, screen protector, bluetooth speaker",
)

if query:
    try:
        results = search_products(query, limit=20)
    except FileNotFoundError as exc:
        st.error(str(exc))
        results = []

    if not results:
        st.info("Eşleşen ürün bulunamadı, farklı bir anahtar kelime dene.")
    else:

        def _label(r):
            title = r["title"] if len(r["title"]) <= 90 else r["title"][:90] + "..."
            return f"{title}  —  {r['review_count']} yorum, {r['avg_rating']:.1f}★"

        options = {_label(r): r for r in results}
        choice_label = st.selectbox(f"{len(results)} sonuç bulundu, birini seç", list(options.keys()))
        chosen = options[choice_label]

        if st.button("Analiz Et", type="primary"):
            log_box = st.empty()
            log_lines = []

            def report_progress(msg: str) -> None:
                log_lines.append(msg)
                log_box.code("\n".join(log_lines))

            try:
                with st.spinner("Analiz çalışıyor (ilk sorguda pyabsa + Gemini çalışır, ~20-60s)..."):
                    result = get_or_compute_insight(chosen["asin"], progress=report_progress)
                log_box.empty()
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Analiz başarısız oldu: {exc}")

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.divider()

    if result.get("from_cache"):
        st.success(f"⚡ Cache'ten getirildi (ilk analiz: {result['created_at']}) — GPU ve LLM çalışmadı")
    else:
        st.success("✅ Yeni analiz tamamlandı ve cache'e kaydedildi")

    st.subheader(result["title"] or result["asin"])
    col1, col2 = st.columns(2)
    col1.metric("Ortalama Puan", f"{result['avg_rating']:.2f} / 5")
    col2.metric("Yorum Sayısı", result["review_count"])

    st.markdown("### 💡 İçgörü")
    tab_tr, tab_en = st.tabs(["Türkçe", "English"])
    with tab_tr:
        st.write(result.get("insight_text_tr") or "(Türkçe çeviri yok -- yeniden analiz gerekiyor)")
    with tab_en:
        st.write(result["insight_text"])

    aspect_stats = result["aspect_stats"]
    if aspect_stats:
        st.markdown("### 📊 Aspect Dağılımı (Top 15)")

        sentiment_rank = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
        rows = [
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "count": counts[sentiment],
                "total": sum(counts.values()),
                "sentiment_rank": sentiment_rank[sentiment],
            }
            for aspect, counts in aspect_stats.items()
            for sentiment in SENTIMENT_ORDER
        ]
        df = pd.DataFrame(rows)
        top_aspects = (
            df.groupby("aspect")["total"].first().sort_values(ascending=False).head(15).index.tolist()
        )
        df = df[df["aspect"].isin(top_aspects)]

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Mention sayısı"),
                y=alt.Y("aspect:N", sort=top_aspects, title=None),
                color=alt.Color(
                    "sentiment:N",
                    scale=alt.Scale(
                        domain=SENTIMENT_ORDER, range=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]
                    ),
                    legend=alt.Legend(title="Sentiment"),
                ),
                order=alt.Order("sentiment_rank:Q"),
                tooltip=[
                    alt.Tooltip("aspect:N", title="Aspect"),
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("count:Q", title="Mentions"),
                ],
            )
            .properties(height=max(300, 24 * len(top_aspects)))
        )
        st.altair_chart(chart, use_container_width=True)

with st.sidebar:
    st.markdown("### Nasıl çalışır")
    st.markdown(
        "1. Ürün adına göre ara (`product_index.parquet` üzerinde)\n"
        "2. Seçilen ASIN daha önce analiz edildiyse cache'ten anında gelir\n"
        "3. İlk analizde: yorumlar filtrelenir → pyabsa ATEPC ile aspect+sentiment "
        "çıkarılır → Gemini ile doğal dil özeti üretilir → sonuç cache'e yazılır"
    )
    st.markdown("---")
    st.caption(f"Model device: {DEVICE}")
