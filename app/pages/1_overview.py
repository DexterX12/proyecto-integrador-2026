import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import get_repo, kpi_row, ms_caption, sidebar_version, ts_to_date  # noqa: E402

st.set_page_config(page_title="Overview · FashRec", page_icon="📊", layout="wide")
repo = get_repo()
version = sidebar_version(repo)
st.title("📊 Overview")

if version is None:
    st.stop()


@st.cache_data(show_spinner="Consultando datos curados...")
def load_overview():
    return {
        "stats": repo.overview_stats(), "ratings": repo.rating_distribution(),
        "monthly": repo.reviews_per_month(), "entities": repo.interactions_per_entity(),
        "cats": repo.category_distribution(),
    }


data = load_overview()
stats, metrics = data["stats"], repo.metrics(version)
cfg, meta = repo.model_config(version), repo.recommendation_meta(version)
tt = metrics.get("splits", {}).get("test", {}).get("two_tower", {}).get("overall", {})

kpi_row([("Usuarios", f"{stats['users']:,}"), ("Ítems (productos)", f"{stats['items']:,}"),
         ("Interacciones", f"{stats['interactions']:,}"), ("Rating promedio", f"{stats['avg_rating']:.2f}")])
kpi_row([("Modelo activo", version), ("Entrenado", cfg.get("created_at", "?")[:10]),
         ("Usuarios con recomendaciones", f"{meta.get('users', 0):,}"),
         ("Recall@20 / NDCG@20 (test)", f"{tt.get('recall@20', 0):.4f} / {tt.get('ndcg@20', 0):.4f}")])
st.caption(f"Datos desde {ts_to_date(stats['ts_min'])} hasta {ts_to_date(stats['ts_max'])} · "
           "Interacciones = una por par (usuario, parent_asin), la reseña más reciente.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribución de ratings")
    r = data["ratings"]
    st.plotly_chart(px.bar(r, x="rating", y="n", labels={"n": "interacciones", "rating": "rating"},
                           text_auto=".2s"), width="stretch")
    pos = r[r["rating"] >= 4]["n"].sum() / r["n"].sum()
    st.caption(f"Relevante = rating ≥ 4 → {pos:.1%} de las interacciones.")
with c2:
    st.subheader("Evolución temporal de reseñas")
    m = data["monthly"].copy()
    m["partición"] = m["split"].map({0: "train", 1: "validation", 2: "test"})
    st.plotly_chart(px.bar(m, x="month", y="n", color="partición",
                           color_discrete_map={"train": "#1976d2", "validation": "#f9a825", "test": "#c62828"},
                           labels={"n": "reseñas", "month": ""}), width="stretch")
    cut = repo.manifest().get("cutoffs", {})
    st.caption(f"Cortes temporales: validation desde {cut.get('val_start')}, test desde {cut.get('test_start')}.")

c3, c4 = st.columns(2)
with c3:
    st.subheader("Interacciones por usuario (log-log)")
    u = data["entities"]["users"]
    st.plotly_chart(px.scatter(u, x="c", y="n", log_x=True, log_y=True,
                               labels={"c": "interacciones del usuario", "n": "nº de usuarios"}),
                    width="stretch")
    st.caption(f"{u.loc[u['c'] == 1, 'n'].sum() / u['n'].sum():.1%} de los usuarios tiene una sola interacción.")
with c4:
    st.subheader("Interacciones por ítem (log-log)")
    i = data["entities"]["items"]
    st.plotly_chart(px.scatter(i, x="c", y="n", log_x=True, log_y=True,
                               labels={"c": "interacciones del ítem", "n": "nº de ítems"}), width="stretch")

c5, c6 = st.columns(2)
with c5:
    st.subheader("Categorías de producto (derivadas del título)")
    cats = data["cats"].groupby("product_type", as_index=False)["n"].sum().sort_values("n")
    st.plotly_chart(px.bar(cats, x="n", y="product_type", orientation="h", labels={"n": "productos", "product_type": ""}),
                    width="stretch")
with c6:
    st.subheader("Cobertura del catálogo")
    cov = {s: metrics["splits"][s]["two_tower"]["beyond_accuracy"]["@20"].get("coverage")
           for s in metrics.get("splits", {}) if "two_tower" in metrics["splits"][s]}
    uni = metrics.get("meta", {}).get("candidate_universe_items", 0)
    rec = repo.recommendation_meta(version)
    if rec:
        st.metric("Ítems distintos recomendados (Top-50, todos los usuarios)",
                  f"{rec['distinct_items_recommended']:,}", f"{rec['distinct_items_recommended'] / max(uni, 1):.1%} del universo")
    for s, v in cov.items():
        st.write(f"Cobertura@20 en {s}: **{v:.2%}** del universo candidato ({uni:,} ítems)")
    k = repo.kcore_report()
    if k:
        st.info(f"Con 3-core (solo positivos) sobrevive el {k['positives_only_k3']['pct_interactions_kept']:.2f}% "
                "de las interacciones, por eso no se aplica k-core como filtro principal.")
ms_caption(repo, "overview_stats", version)
