import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import fmt_price, get_repo, ms_caption, product_cards, sidebar_version  # noqa: E402

st.set_page_config(page_title="Productos · FashRec", page_icon="🔎", layout="wide")
repo = get_repo()
version = sidebar_version(repo)
st.title("🔎 Explorador de productos y embeddings")
if version is None:
    st.stop()

query = st.text_input("Buscar por título o `parent_asin`", placeholder="p. ej. leather watch, B08BHN9PK5")
selected = None
if query.strip():
    found = repo.search_products(query, limit=30)
    ms_caption(repo, "search_products")
    if found.empty:
        st.warning("Sin resultados.")
    else:
        labels = {r.parent_asin: f"{(r.title or '')[:90]} · {r.parent_asin} · {r.train_count} int." for r in found.itertuples()}
        selected = st.selectbox("Resultados (ordenados por popularidad en train)", list(labels), format_func=labels.get)

if selected:
    prod = repo.product(selected)
    left, right = st.columns([1, 2])
    with left:
        product_cards([{"title": prod["title"], "image_url": prod["image_url"],
                        "badges": [(prod["product_type"], "#455a64"), (prod["audience"], "#6a1b9a")],
                        "lines": [fmt_price(prod["price"]), prod["store"]]}], columns=1, height_per_row=340)
    with right:
        st.markdown(f"**parent_asin:** `{prod['parent_asin']}`  \n**Tienda:** {prod['store'] or 's/d'}  \n"
                    f"**Rating promedio (metadata):** {prod['meta_average_rating']} ({prod['meta_rating_number']} calificaciones)  \n"
                    f"**Interacciones en train:** {int(prod['train_count'])}  \n"
                    f"**En el universo candidato:** {'sí' if prod['in_universe'] else 'no (producto nuevo: cold start de ítem)'}")
        if prod["features_text"]:
            with st.expander("Features"):
                st.write(prod["features_text"])
        if prod["description_text"]:
            with st.expander("Descripción"):
                st.write(prod["description_text"])
    st.subheader("Productos más cercanos (FAISS, similitud coseno entre embeddings)")
    if not prod["in_universe"]:
        st.info("Este producto no está en el índice (sin interacciones en train). Se necesitaría codificarlo con "
                "la torre de producto usando solo contenido; ver comparación de cold start.")
    else:
        sim = repo.similar_items(version, int(prod["item_idx"]), k=10)
        ms_caption(repo, "similar_items", version)
        product_cards([{"title": r.title, "image_url": r.image_url,
                        "badges": [(f"sim {r.similarity:.3f}", "#1565c0")],
                        "lines": [r.product_type, fmt_price(r.price)]} for r in sim.itertuples()], columns=5)

st.divider()
st.subheader("Proyección UMAP de embeddings de producto")
proj = repo.projection(version)
if proj.empty:
    st.info("No hay proyección para esta versión (recommend_job --skip-umap).")
else:
    st.caption("Exploratoria: muestra de productos ponderada por popularidad. Que las categorías formen zonas indica "
               "que el embedding captura contenido; **no** es una prueba de la calidad del recomendador.")
    fig = px.scatter(proj, x="x", y="y", color="product_type", hover_name="title", hover_data=["parent_asin"],
                     opacity=0.75, height=650)
    fig.update_traces(marker=dict(size=5))
    if selected and (proj["parent_asin"] == selected).any():
        pt = proj[proj["parent_asin"] == selected].iloc[0]
        fig.add_scatter(x=[pt["x"]], y=[pt["y"]], mode="markers", name="seleccionado",
                        marker=dict(size=16, color="black", symbol="star"))
    fig.update_layout(xaxis_title="", yaxis_title="", xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, width="stretch")
