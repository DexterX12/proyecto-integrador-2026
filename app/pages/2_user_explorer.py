import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.components.common import (fmt_date, fmt_price, get_repo, kpi_row, ms_caption,  # noqa: E402
                                   product_cards, sidebar_version)

st.set_page_config(page_title="Usuarios · FashRec", page_icon="🧑", layout="wide")
repo = get_repo()
version = sidebar_version(repo)
st.title("🧑 Explorador de usuarios")
if version is None:
    st.stop()

meta = repo.recommendation_meta(version)
st.caption(f"Recomendaciones del modelo `{version}` generadas el {meta.get('generated_at', '?')[:16].replace('T', ' ')} UTC "
           f"con el historial hasta **{meta.get('as_of', '?')}**. Los productos ya vistos antes de esa fecha no se recomiendan.")

with st.sidebar:
    st.header("Controles")
    n_recs = st.select_slider("Número de recomendaciones (N)", options=[5, 10, 20], value=10)
    only_hits = st.checkbox("Solo usuarios con compras posteriores (permite comprobar aciertos)", value=True)
    search = st.text_input("Filtrar user_id (contiene)", "")

users = repo.list_users(version, search=search, only_with_targets=only_hits, limit=300)
if users.empty:
    st.warning("Ningún usuario coincide con el filtro.")
    st.stop()

labels = {r.user_id: f"{r.user_id[:14]}… · {r.n_hist_pos} positivos previos · {r.n_test_positives} posteriores"
          for r in users.itertuples()}
user_id = st.selectbox("Usuario", list(labels), format_func=labels.get)
manual = st.text_input("…o escriba un user_id exacto (usuario nuevo → fallback por popularidad)", "")
if manual.strip():
    user_id = manual.strip()

recs = repo.recommendations(user_id, version, n=n_recs)
history = repo.user_history(user_id, version, limit=15)
seen = repo.user_seen_asins(user_id)
future = repo.user_future_positives(user_id, version)
future_asins = set(future["parent_asin"])

if recs.empty:
    st.warning("Este usuario no tiene recomendaciones personalizadas (sin historial positivo previo). "
               "Se muestra el **fallback por popularidad** (cold start de usuario).")
    pop = repo.popularity_fallback(version, n_recs)
    product_cards([{"title": r.title, "image_url": r.image_url,
                    "badges": [(f"#{r.rank}", "#455a64")],
                    "lines": [r.product_type, fmt_price(r.price)]} for r in pop.itertuples()])
    ms_caption(repo, "popularity_fallback", version)
    st.stop()

row = users[users["user_id"] == user_id]
kpi_row([("Positivos previos", f"{int(row['n_hist_pos'].iloc[0]) if len(row) else len(history)}"),
         ("Productos vistos (excluidos)", f"{len(seen)}"),
         ("Compras posteriores (rating ≥ 4)", f"{len(future)}"),
         ("Aciertos en el Top-N", f"{len(set(recs['parent_asin']) & future_asins)}")])

st.subheader("Historial reciente")
if history.empty:
    st.info("Sin historial.")
else:
    product_cards([{
        "title": r.title, "image_url": r.image_url,
        "badges": [(f"★ {r.rating:.0f}", "#2e7d32" if r.rating >= 4 else "#c62828")],
        "lines": [fmt_date(r.date), r.product_type, fmt_price(r.price)]} for r in history.itertuples()], columns=5)
ms_caption(repo, "user_history", version)

st.subheader(f"Top-{n_recs} recomendado")
cards = []
for r in recs.itertuples():
    badges = [(f"#{r.rank}", "#455a64"), (f"score {r.score:.3f}", "#1565c0")]
    if r.parent_asin in future_asins:
        badges.append(("✔ acierto", "#2e7d32"))
    if r.parent_asin in seen:
        badges.append(("⚠ ya visto", "#c62828"))
    rating = f"★ {r.avg_rating:.1f} ({int(r.n_ratings):,})" if r.avg_rating == r.avg_rating and r.n_ratings == r.n_ratings else "★ s/d"
    cards.append({"title": r.title, "image_url": r.image_url, "badges": badges,
                  "lines": [f"{r.product_type} · {r.audience}", fmt_price(r.price), rating]})
product_cards(cards, columns=5)
ms_caption(repo, "recommendations", version)
violations = [r.parent_asin for r in recs.itertuples() if r.parent_asin in seen]
if violations:
    st.error(f"{len(violations)} recomendación(es) pertenecen al historial (no debería ocurrir).")
else:
    st.success("Ninguna recomendación pertenece al historial del usuario.")

if len(future):
    with st.expander(f"Compras reales posteriores al corte ({len(future)})"):
        product_cards([{"title": r.title, "image_url": r.image_url,
                        "badges": [(f"★ {r.rating:.0f}", "#2e7d32")] + ([] if r.in_universe else [("fuera del universo", "#6d4c41")]),
                        "lines": [fmt_date(r.date)]} for r in future.itertuples()], columns=5)
